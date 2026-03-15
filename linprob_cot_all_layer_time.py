"""
Token-Level Linear Probing on Pre-Generated CoT Traces - TIME TARGET (All Layers Concatenated)

This script is similar to linprob_cot_all_layer.py but trains probes to predict TIME instead of velocity.
We concatenate activations from ALL layers at each token position and train a single probe per token.

Each probe takes input of shape (num_layers * hidden_dim) and predicts the time value.

Workflow:
1. Load pre-generated CoT traces from disk
2. For selected traces, truncate at the point where TIME value appears (end of reasoning)
3. Generate many variations by substituting numbers while keeping token count identical  
4. Train linear probes (Ridge regression) for each token position starting from "Answer"
5. Each probe operates on concatenated activations from ALL layers

Example: "A 17 kg runner has 2.388e+04 Joules... Answer (step-by-step): ... time = 42.5 seconds"
We truncate at "time = " and train probes to predict the time value.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
from pathlib import Path
import json
import re
from collections import defaultdict
import joblib

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment Configuration
EXPERIMENT = "velocity_uniform_t"  # Experiment type (determines which traces to load)
TARGET_VARIABLE = "time"  # What we're probing for
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
TRACES_DIR = Path(f"/home/wuroderi/scratch/reasoning_traces/Qwen2.5-32B/{EXPERIMENT}")
TRACES_METADATA_FILE = TRACES_DIR / "traces_metadata.json"
PLOTS_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_linprob_cot_all_layer_{EXPERIMENT}_time")
PLOTS_DIR.mkdir(exist_ok=True)
PROBES_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/probes_linprob_cot_all_layer_{EXPERIMENT}_time")
PROBES_DIR.mkdir(exist_ok=True)

# Data Configuration
TRACE_INDICES = [0, 1, 2, 3, 4]  # Which pre-generated traces to use as base CoT outputs
TRAIN_RATIO = 0.8  # 80% train, 20% validation
NUM_VARIATIONS_PER_TRACE = 200  # Target number of synthetic variations per base trace
TRUNCATE_AT_TOKEN_INDEX = None  # Token indices to truncate at (exclusive). None = auto-detect time variable

# Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis Configuration
LAYERS_TO_PROBE = list(range(64))  # Use ALL 64 layers (concatenated)
RIDGE_ALPHA = 1.0  # Ridge regression regularization strength

print(f"="*80)
print(f"TOKEN-LEVEL LINEAR PROBING (ALL LAYERS CONCATENATED): {EXPERIMENT.upper()} - TARGET: {TARGET_VARIABLE.upper()}")
print(f"="*80)
print(f"Model: {MODEL_PATH}")
print(f"Traces dir: {TRACES_DIR}")
print(f"Device: {device}")
print(f"Plots directory: {PLOTS_DIR}")
print(f"Trace indices: {TRACE_INDICES}")
print(f"Target variable: {TARGET_VARIABLE}")
print(f"Using {len(LAYERS_TO_PROBE)} layers concatenated (input dim: {len(LAYERS_TO_PROBE)} × d_model)")
if TRUNCATE_AT_TOKEN_INDEX is not None:
    if isinstance(TRUNCATE_AT_TOKEN_INDEX, list):
        print(f"Truncation mode: TOKEN INDEX LIST (per-trace truncation)")
        if len(TRUNCATE_AT_TOKEN_INDEX) != len(TRACE_INDICES):
            raise ValueError(f"TRUNCATE_AT_TOKEN_INDEX list length ({len(TRUNCATE_AT_TOKEN_INDEX)}) must match TRACE_INDICES length ({len(TRACE_INDICES)})")
        for i, (trace_idx, token_idx) in enumerate(zip(TRACE_INDICES, TRUNCATE_AT_TOKEN_INDEX)):
            print(f"  Trace {trace_idx}: truncate at token {token_idx} (exclusive)")
    else:
        print(f"Truncation mode: TOKEN INDEX (truncating at token {TRUNCATE_AT_TOKEN_INDEX}, exclusive)")
else:
    print(f"Truncation mode: AUTO-DETECT (searching for time variable pattern)")
print()

# ==========================================
# LOAD MODEL AND TRACES
# ==========================================

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-32B",
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
    fold_value_biases=False,
    move_to_device=False,
    load_state_dict=False
)

# Ensure embedding layer is on a GPU device
if model.embed.W_E.device.type == 'cpu':
    model.embed = model.embed.to('cuda:0')
    print("Moved embedding layer to cuda:0")

if hasattr(model, 'pos_embed') and model.pos_embed.W_pos.device.type == 'cpu':
    model.pos_embed = model.pos_embed.to('cuda:0')
    print("Moved positional embedding to cuda:0")

print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dimensions")
print(f"Concatenated input dimension: {len(LAYERS_TO_PROBE) * model.cfg.d_model}")
print(f"Embedding device: {model.embed.W_E.device}\n")

# Load pre-generated traces
print("Loading pre-generated traces...")
with open(TRACES_METADATA_FILE, 'r') as f:
    all_traces = json.load(f)
print(f"Loaded {len(all_traces)} traces from {TRACES_METADATA_FILE}")
print(f"Will run {len(TRACE_INDICES)} separate experiments, one for each base trace CoT output")
print()

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def find_number_in_scientific_notation(text, number):
    """Find scientific notation representation of a number in text."""
    sci_patterns = [
        f"{number:.3e}",
        f"{number:.2e}",
        f"{number:.4e}",
        f"{number:.1e}",
    ]
    
    for pattern in sci_patterns:
        if pattern in text:
            return pattern
    
    for pattern in sci_patterns:
        pattern_no_plus = pattern.replace('+', '')
        if pattern_no_plus in text:
            return pattern_no_plus
            
    return None

def truncate_at_time_variable(generated_text, model=None, token_index=None):
    """
    Truncate at the point where TIME value appears or at specified token index.
    
    Args:
        generated_text: Text to truncate
        model: TransformerLens model (required if token_index is specified)
        token_index: Token index to truncate at (exclusive). None = auto-detect time variable
    
    Returns:
        Truncated text
    """
    # If token_index is specified, truncate at that exact token position
    if token_index is not None:
        if model is None:
            raise ValueError("model must be provided when using token_index")
        
        # Tokenize the text
        tokens = model.to_tokens(generated_text, prepend_bos=True)[0]
        
        # Truncate at the specified index (exclusive)
        if token_index >= len(tokens):
            return generated_text  # Index out of range, return full text
        
        truncated_tokens = tokens[:token_index]
        
        # Decode back to text
        truncated_text = model.to_string(truncated_tokens)
        return truncated_text
    
    # Otherwise, use pattern matching (original behavior)
    # Common patterns where time value appears
    patterns = [
        r't\s*=\s*',  # "t = "
        r'time\s*=\s*',  # "time = "
        r'Time\s*=\s*',  # "Time = "
        r'answer\s*(?:is\s*)?=?\s*',  # "answer is" or "answer ="
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            # Include the pattern itself but stop there
            truncation_point = match.end()
            truncated = generated_text[:truncation_point]
            return truncated
    
    # If no pattern found, return original (no CoT generated)
    return generated_text

def create_variations_from_traces(trace, all_traces_pool, tokenizer, model, truncate_token_index=None):
    """Create synthetic variations of a trace by substituting numbers."""
    variations = []
    
    original_full_text = trace.get('generated_text', trace['prompt'])
    original_text = truncate_at_time_variable(original_full_text, model, truncate_token_index)
    
    original_m = trace['m']
    original_ke = trace['ke']
    original_v = trace['v']
    original_d = trace['d']
    original_t = trace.get('expected_time', trace['d'] / trace['v'])  # t = d/v
    original_format_id = trace['format_id']
    
    original_ke_str = find_number_in_scientific_notation(original_text, original_ke)
    if original_ke_str is None:
        original_ke_str = f"{original_ke:.3e}"
    
    original_tokens_obj = model.to_tokens(original_text, prepend_bos=True)
    original_n_tokens = original_tokens_obj.shape[1]
    
    same_format_traces = [t for t in all_traces_pool if t['format_id'] == original_format_id and t['id'] != trace['id']]
    
    print(f"  Found {len(same_format_traces)} traces with same format_id {original_format_id}")
    
    successful_variations = 0
    
    for other_trace in same_format_traces:
        m_new = other_trace['m']
        v_new = other_trace['v']
        d_new = other_trace['d']
        ke_new = other_trace['ke']
        t_new = other_trace.get('expected_time', d_new / v_new)  # t = d/v
        
        other_full_text = other_trace.get('generated_text', other_trace['prompt'])
        other_text = truncate_at_time_variable(other_full_text, model, truncate_token_index)
        
        ke_new_str = find_number_in_scientific_notation(other_text, ke_new)
        if ke_new_str is None:
            ke_new_str = f"{ke_new:.3e}"
        
        new_text = original_text
        new_text = new_text.replace(f" {original_m} kg", f" {m_new} kg")
        new_text = new_text.replace(original_ke_str, ke_new_str)
        new_text = new_text.replace(f" {original_d} m", f" {d_new} m")
        
        new_tokens_obj = model.to_tokens(new_text, prepend_bos=True)
        new_n_tokens = new_tokens_obj.shape[1]
        
        if new_n_tokens != original_n_tokens:
            continue
            
        if m_new == original_m and ke_new == original_ke and d_new == original_d:
            continue
        
        variations.append({
            'prompt': new_text,
            'm': m_new,
            'ke': ke_new,
            'v': v_new,
            'd': d_new,
            't': t_new,  # TARGET VARIABLE
            'original_trace_id': trace['id'],
            'source_trace_id': other_trace['id'],
            'n_tokens': new_n_tokens
        })
        successful_variations += 1
    
    print(f"  Generated {successful_variations} valid variations")
    
    return variations

def extract_activations_all_layers_concatenated(prompts, model, layers, batch_size=8):
    """
    Extract activations from all layers and concatenate them for each token.
    Returns:
        - concatenated_activations: numpy array [total_tokens, len(layers) * d_model]
        - all_token_counts: list of token counts per prompt
    """
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    all_concatenated_activations = []
    all_token_counts = []
    
    embed_device = model.embed.W_E.device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize individually
        batch_tokens_list = []
        batch_token_lengths = []
        max_len = 0
        
        for prompt in batch_prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            batch_tokens_list.append(tokens)
            batch_token_lengths.append(tokens.shape[1])
            max_len = max(max_len, tokens.shape[1])
        
        # Pad to same length
        padded_tokens = []
        for tokens in batch_tokens_list:
            if tokens.shape[1] < max_len:
                padding = torch.zeros((1, max_len - tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=1)
            padded_tokens.append(tokens)
        
        batch_tokens = torch.cat(padded_tokens, dim=0).to(embed_device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda name: name in hook_names
            )
        
        # For each prompt in batch, concatenate all layer activations
        for j in range(len(batch_prompts)):
            n_tokens = batch_token_lengths[j]
            
            # Collect activations from all layers for this prompt
            layer_activations = []
            for layer in layers:
                hook_name = f"blocks.{layer}.hook_resid_post"
                layer_acts = cache[hook_name][j, :n_tokens].cpu().float()  # [n_tokens, d_model]
                layer_activations.append(layer_acts)
            
            # Concatenate along feature dimension: [n_tokens, len(layers) * d_model]
            concatenated = torch.cat(layer_activations, dim=1)
            all_concatenated_activations.append(concatenated)
        
        all_token_counts.extend(batch_token_lengths)
    
    # Concatenate all prompts: [total_tokens, len(layers) * d_model]
    concatenated_activations = torch.cat(all_concatenated_activations, dim=0).numpy()
    
    return concatenated_activations, all_token_counts

def find_answer_token_position(prompt, tokenizer, model):
    """Find the token position where 'Answer' starts."""
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    
    for i in range(len(tokens)):
        token_str = model.to_string(tokens[i])
        if "Answer" in token_str or "answer" in token_str:
            return i
    
    return len(tokens) // 2

print("Generating synthetic variations for each trace...")
print("="*80)

# ==========================================
# MAIN EXPERIMENT LOOP
# ==========================================

for i, trace_idx in enumerate(TRACE_INDICES):
    if trace_idx >= len(all_traces):
        print(f"WARNING: Trace index {trace_idx} out of range, skipping")
        continue
    
    trace = all_traces[trace_idx]
    
    # Get the truncation index for this specific trace
    if TRUNCATE_AT_TOKEN_INDEX is not None:
        if isinstance(TRUNCATE_AT_TOKEN_INDEX, list):
            current_truncate_index = TRUNCATE_AT_TOKEN_INDEX[i]
        else:
            current_truncate_index = TRUNCATE_AT_TOKEN_INDEX
    else:
        current_truncate_index = None
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: TRACE {trace['id']} (Index {trace_idx}) - Format {trace['format_id']}")
    print(f"{'='*80}")
    
    # Calculate expected time
    expected_time = trace.get('expected_time', trace['d'] / trace['v'])
    print(f"Original values: m={trace['m']} kg, ke={trace['ke']:.3e} J, v={trace['v']} m/s, d={trace['d']} m")
    print(f"Expected time: t={expected_time:.3f} seconds")
    
    # Create trace-specific directories
    trace_plots_dir = PLOTS_DIR / f"trace_{trace_idx}"
    trace_plots_dir.mkdir(exist_ok=True)
    trace_probes_dir = PROBES_DIR / f"trace_{trace_idx}"
    trace_probes_dir.mkdir(exist_ok=True)
    
    # Get full generated text and truncate at time variable
    original_full_text = trace.get('generated_text', trace['prompt'])
    original_truncated = truncate_at_time_variable(original_full_text, model, current_truncate_index)
    
    if current_truncate_index is not None:
        print(f"\nOriginal text truncated at token index {current_truncate_index} (exclusive):")
    else:
        print(f"\nOriginal text truncated at time variable (pattern matching):")
    print(f"  {original_truncated[-200:]}" if len(original_truncated) > 200 else f"  {original_truncated}")
    
    # Tokenize original truncated text
    original_tokens_obj = model.to_tokens(original_truncated, prepend_bos=True)
    original_token_strs = model.to_str_tokens(original_truncated, prepend_bos=True)
    print(f"\nOriginal tokenization ({original_tokens_obj.shape[1]} tokens)")
    
    # Generate variations
    print(f"\nGenerating variations using all {len(all_traces)} traces...")
    if current_truncate_index is not None:
        print(f"Using token index truncation: truncating at token {current_truncate_index} (exclusive)")
    else:
        print(f"Using automatic truncation: searching for time variable pattern")
    variations = create_variations_from_traces(trace, all_traces, tokenizer, model, current_truncate_index)
    
    print(f"\n{'-'*80}")
    print(f"GENERATED VARIATIONS ({len(variations)} total)")
    print(f"{'-'*80}")
    
    for var_idx, var in enumerate(variations[:5]):
        print(f"\nVariation {var_idx + 1} (from trace {var['source_trace_id']}):")
        print(f"  Values: m={var['m']} kg, ke={var['ke']:.3e} J, v={var['v']} m/s, d={var['d']} m")
        print(f"  Expected time: t={var['t']:.3f} seconds")
        print(f"  Token count: {var['n_tokens']}")
    
    if len(variations) > 5:
        print(f"\n... and {len(variations) - 5} more variations")
    
    # ==========================================
    # EXTRACT ACTIVATIONS (CONCATENATED)
    # ==========================================

    print("\nExtracting concatenated activations from all variations...")
    print("This may take a while...")

    # Split into train and validation
    train_variations, val_variations = train_test_split(
        variations, 
        train_size=TRAIN_RATIO, 
        random_state=42
    )

    print(f"Train variations: {len(train_variations)}")
    print(f"Val variations: {len(val_variations)}")

    # Extract activations for training data
    train_prompts = [v['prompt'] for v in train_variations]
    train_times = np.array([v['t'] for v in train_variations])  # TARGET: time values

    print("\nExtracting training activations (all layers concatenated)...")
    train_activations, train_token_counts = extract_activations_all_layers_concatenated(
        train_prompts, model, LAYERS_TO_PROBE, batch_size=4
    )

    print(f"Extracted training activations: {train_activations.shape}")
    print(f"  Shape: [total_tokens={train_activations.shape[0]}, layers*d_model={train_activations.shape[1]}]")

    # Extract activations for validation data  
    val_prompts = [v['prompt'] for v in val_variations]
    val_times = np.array([v['t'] for v in val_variations])  # TARGET: time values

    print("\nExtracting validation activations (all layers concatenated)...")
    val_activations, val_token_counts = extract_activations_all_layers_concatenated(
        val_prompts, model, LAYERS_TO_PROBE, batch_size=4
    )

    print(f"Extracted validation activations: {val_activations.shape}")

    # ==========================================
    # TRAIN PER-TOKEN LINEAR PROBES
    # ==========================================

    print("\n" + "="*80)
    print("TRAINING LINEAR PROBES FOR EACH TOKEN POSITION")
    print(f"(Each probe uses concatenated activations to predict TIME)")
    print("="*80)

    # Find "Answer" position
    example_prompt = train_prompts[0]
    answer_token_pos = find_answer_token_position(example_prompt, tokenizer, model)
    print(f"Starting probes from token position {answer_token_pos} (Answer keyword)")
    print()

    # Determine max sequence length
    max_seq_len = max(train_token_counts + val_token_counts)
    print(f"Maximum sequence length: {max_seq_len} tokens")

    # Get token strings for plotting
    example_token_strs = model.to_str_tokens(example_prompt, prepend_bos=True)
    print(f"Example tokens (last 20): ...{example_token_strs[-20:]}")
    print()

    # Initialize storage
    probes = {}  # probes[token_pos] = Ridge probe
    results = {}  # results[token_pos] = {train_r2, val_r2, ...}

    # Train probes for each token position
    for token_pos in range(answer_token_pos, max_seq_len):
        if token_pos % 10 == 0:
            print(f"\nTraining probe for token position {token_pos}...")
        
        # Collect concatenated activations for this token position
        train_acts_at_pos = []
        train_labels_at_pos = []
        
        current_idx = 0
        for i, n_tokens in enumerate(train_token_counts):
            if token_pos < n_tokens:
                # Get concatenated activation at this token position
                act = train_activations[current_idx + token_pos]
                train_acts_at_pos.append(act)
                train_labels_at_pos.append(train_times[i])
            current_idx += n_tokens
        
        n_samples = len(train_labels_at_pos)
        if n_samples < 10:
            if token_pos % 10 == 0:
                print(f"  Skipping: only {n_samples} samples at this position")
            continue
        
        train_acts_at_pos = np.array(train_acts_at_pos)
        train_labels_at_pos = np.array(train_labels_at_pos)
        
        # Collect validation data
        val_acts_at_pos = []
        val_labels_at_pos = []
        
        current_idx = 0
        for i, n_tokens in enumerate(val_token_counts):
            if token_pos < n_tokens:
                act = val_activations[current_idx + token_pos]
                val_acts_at_pos.append(act)
                val_labels_at_pos.append(val_times[i])
            current_idx += n_tokens
        
        val_acts_at_pos = np.array(val_acts_at_pos)
        val_labels_at_pos = np.array(val_labels_at_pos)
        
        # Train Ridge regression
        probe = Ridge(alpha=RIDGE_ALPHA)
        probe.fit(train_acts_at_pos, train_labels_at_pos)
        
        probes[token_pos] = probe
        
        # Evaluate on training data
        train_preds = probe.predict(train_acts_at_pos)
        train_r2 = r2_score(train_labels_at_pos, train_preds)
        train_mae = mean_absolute_error(train_labels_at_pos, train_preds)
        train_mpe = np.mean(np.abs((train_preds - train_labels_at_pos) / train_labels_at_pos)) * 100
        
        # Evaluate on validation data
        if len(val_labels_at_pos) > 0:
            val_preds = probe.predict(val_acts_at_pos)
            val_r2 = r2_score(val_labels_at_pos, val_preds)
            val_mae = mean_absolute_error(val_labels_at_pos, val_preds)
            val_mpe = np.mean(np.abs((val_preds - val_labels_at_pos) / val_labels_at_pos)) * 100
        else:
            val_r2 = 0.0
            val_mae = float('inf')
            val_mpe = float('inf')
        
        results[token_pos] = {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mpe': train_mpe,
            'val_mpe': val_mpe,
            'n_train': n_samples,
            'n_val': len(val_labels_at_pos)
        }
        
        if token_pos % 10 == 0:
            print(f"  Token {token_pos}: Val R²={val_r2:.3f}, MAE={val_mae:.2f}, MPE={val_mpe:.1f}%")

    print(f"\n{'='*80}")
    print("PROBE TRAINING COMPLETE")
    print(f"{'='*80}")
    print()

    # ==========================================
    # COLLECT TOP 5 PREDICTIONS
    # ==========================================

    print("Collecting top 5 predictions...")
    
    all_predictions = []
    for token_pos in results:
        all_predictions.append({
            'token_pos': token_pos,
            'val_r2': results[token_pos]['val_r2'],
            'val_mae': results[token_pos]['val_mae'],
            'val_mpe': results[token_pos]['val_mpe']
        })
    
    all_predictions.sort(key=lambda x: x['val_r2'], reverse=True)
    top_5_configs = all_predictions[:5]
    
    top_5_predictions = []
    for config in top_5_configs:
        token_pos = config['token_pos']
        
        token_str = example_token_strs[token_pos] if token_pos < len(example_token_strs) else f"pos{token_pos}"
        
        # Get validation predictions
        val_acts_list = []
        val_labels_list = []
        val_indices = []
        
        current_idx = 0
        for i, n_tokens in enumerate(val_token_counts):
            if token_pos < n_tokens:
                act = val_activations[current_idx + token_pos]
                val_acts_list.append(act)
                val_labels_list.append(val_times[i])
                val_indices.append(i)
            current_idx += n_tokens
        
        val_acts_array = np.array(val_acts_list)
        val_labels_array = np.array(val_labels_list)
        
        probe = probes[token_pos]
        preds = probe.predict(val_acts_array)
        
        top_5_predictions.append({
            'token_pos': token_pos,
            'token': token_str,
            'val_r2': config['val_r2'],
            'val_mae': config['val_mae'],
            'val_mpe': config['val_mpe'],
            'predictions': preds.tolist(),
            'true_values': val_labels_array.tolist(),
            'validation_indices': val_indices
        })
    
    # Save validation data
    print("\nSaving validation data to JSON...")
    validation_data = {
        'trace_id': trace['id'],
        'trace_idx': trace_idx,
        'target_variable': TARGET_VARIABLE,
        'truncation_mode': 'token_index' if current_truncate_index is not None else 'auto_detect',
        'truncation_token_index': current_truncate_index,
        'original_values': {
            'm': trace['m'],
            'ke': trace['ke'],
            'v': trace['v'],
            'd': trace['d'],
            't': expected_time
        },
        'variations': val_variations,
        'times': val_times.tolist(),
        'n_samples': len(val_variations),
        'top_5_predictions': top_5_predictions,
        'probe_type': 'all_layers_concatenated',
        'concatenated_dim': len(LAYERS_TO_PROBE) * model.cfg.d_model
    }
    val_data_file = trace_plots_dir / 'validation_data.json'
    with open(val_data_file, 'w') as f:
        json.dump(validation_data, f, indent=2)
    print(f"Saved validation data to: {val_data_file}")
    
    print("\nTop 5 Predictions:")
    for i, pred_info in enumerate(top_5_predictions):
        print(f"  {i+1}. Token {pred_info['token_pos']} ('{pred_info['token']}')")
        print(f"     R²={pred_info['val_r2']:.4f}, MAE={pred_info['val_mae']:.2f}, MPE={pred_info['val_mpe']:.1f}%")
    print()

    # ==========================================
    # SAVE PROBES
    # ==========================================

    print("Saving trained probes...")
    for token_pos in probes:
        probe_filename = trace_probes_dir / f"probe_token{token_pos}_all_layers_time.joblib"
        joblib.dump(probes[token_pos], probe_filename)
    print(f"Saved {len(probes)} probes to: {trace_probes_dir}")
    print()

    # ==========================================
    # VISUALIZATION
    # ==========================================

    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    token_positions = sorted(probes.keys())
    
    # Helper function for token labels
    def is_numeric_token(token_str):
        """Check if token represents a number."""
        cleaned = token_str.strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            if any(c in cleaned for c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                numeric_chars = sum(c.isdigit() for c in cleaned)
                if numeric_chars / max(len(cleaned), 1) > 0.5:
                    return True
            return False

    token_labels = []
    for pos in token_positions:
        if pos < len(example_token_strs):
            token_str = example_token_strs[pos]
            if len(token_str) > 8:
                token_str = token_str[:6] + '..'
            if is_numeric_token(token_str):
                token_labels.append(f'[NUM]')
            else:
                token_labels.append(token_str)
        else:
            token_labels.append(f'{pos}')

    # Plot 1: R² across token positions
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    val_r2_values = [results[pos]['val_r2'] for pos in token_positions]
    
    ax.plot(token_positions, val_r2_values, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Token Position', fontsize=14)
    ax.set_ylabel('Validation R²', fontsize=14)
    ax.set_title(f'Linear Probe Performance (All Layers, Target: TIME) - Trace {trace_idx}\n{EXPERIMENT.capitalize()}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Set x-axis labels
    ax.set_xticks(token_positions[::max(1, len(token_positions)//20)])
    ax.set_xticklabels([token_labels[i] for i in range(0, len(token_positions), max(1, len(token_positions)//20))], 
                       rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'r2_per_token_time.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'r2_per_token_time.png'}")
    plt.close()

    # Plot 2: MAE across token positions
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    val_mae_values = [results[pos]['val_mae'] for pos in token_positions]
    
    ax.plot(token_positions, val_mae_values, 'r-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Token Position', fontsize=14)
    ax.set_ylabel('Validation MAE (seconds)', fontsize=14)
    ax.set_title(f'Linear Probe MAE (All Layers, Target: TIME) - Trace {trace_idx}\n{EXPERIMENT.capitalize()}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.set_xticks(token_positions[::max(1, len(token_positions)//20)])
    ax.set_xticklabels([token_labels[i] for i in range(0, len(token_positions), max(1, len(token_positions)//20))], 
                       rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'mae_per_token_time.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'mae_per_token_time.png'}")
    plt.close()

    # Plot 3: MPE across token positions
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    val_mpe_values = [results[pos]['val_mpe'] for pos in token_positions]
    
    ax.plot(token_positions, val_mpe_values, 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Token Position', fontsize=14)
    ax.set_ylabel('Validation MPE (%)', fontsize=14)
    ax.set_title(f'Linear Probe Mean Percent Error (All Layers, Target: TIME) - Trace {trace_idx}\n{EXPERIMENT.capitalize()}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.set_xticks(token_positions[::max(1, len(token_positions)//20)])
    ax.set_xticklabels([token_labels[i] for i in range(0, len(token_positions), max(1, len(token_positions)//20))], 
                       rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'mpe_per_token_time.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'mpe_per_token_time.png'}")
    plt.close()

    # Plot 4: Combined metrics
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

    ax1.plot(token_positions, val_r2_values, 'b-', linewidth=2)
    ax1.set_ylabel('Validation R²', fontsize=12)
    ax1.set_title(f'Probe Performance for TIME Prediction (All Layers) - Trace {trace_idx}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(token_positions, val_mae_values, 'r-', linewidth=2)
    ax2.set_ylabel('Validation MAE (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax3.plot(token_positions, val_mpe_values, 'g-', linewidth=2)
    ax3.set_xlabel('Token Position', fontsize=12)
    ax3.set_ylabel('Validation MPE (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Set x-axis labels on bottom plot only
    ax3.set_xticks(token_positions[::max(1, len(token_positions)//20)])
    ax3.set_xticklabels([token_labels[i] for i in range(0, len(token_positions), max(1, len(token_positions)//20))], 
                        rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'combined_metrics_time.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'combined_metrics_time.png'}")
    plt.close()

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    # Find best performance
    best_token_pos = max(token_positions, key=lambda pos: results[pos]['val_r2'])
    best_r2 = results[best_token_pos]['val_r2']
    best_mae = results[best_token_pos]['val_mae']
    best_mpe = results[best_token_pos]['val_mpe']

    summary = {
        'experiment': EXPERIMENT,
        'target_variable': TARGET_VARIABLE,
        'trace_id': trace['id'],
        'trace_idx': trace_idx,
        'probe_type': 'all_layers_concatenated',
        'truncation_mode': 'token_index' if current_truncate_index is not None else 'auto_detect',
        'truncation_token_index': current_truncate_index,
        'n_layers': len(LAYERS_TO_PROBE),
        'concatenated_dim': len(LAYERS_TO_PROBE) * model.cfg.d_model,
        'n_total_traces': len(all_traces),
        'n_variations': len(variations),
        'n_train': len(train_variations),
        'n_val': len(val_variations),
        'token_positions': token_positions,
        'answer_token_position': answer_token_pos,
        'best_results': {
            'token_pos': best_token_pos,
            'val_r2': best_r2,
            'val_mae': best_mae,
            'val_mpe': best_mpe,
            'train_r2': results[best_token_pos]['train_r2']
        }
    }

    results_file = trace_plots_dir / 'probe_results_summary_time.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {results_file}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE FOR TRACE {trace_idx}")
    print(f"{'='*80}")
    print(f"Best performance (predicting TIME):")
    print(f"  Token position: {best_token_pos}")
    print(f"  Token: {example_token_strs[best_token_pos] if best_token_pos < len(example_token_strs) else '???'}")
    print(f"  Validation R²: {best_r2:.4f}")
    print(f"  Validation MAE: {best_mae:.2f} seconds")
    print(f"  Validation MPE: {best_mpe:.1f}%")
    print(f"  (Using concatenated activations from {len(LAYERS_TO_PROBE)} layers)")
    print(f"\nAll visualizations saved to: {trace_plots_dir}")
    print(f"All probes saved to: {trace_probes_dir}")
    print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"{'='*80}")
print(f"Processed {len(TRACE_INDICES)} trace(s)")
print(f"Target variable: {TARGET_VARIABLE}")
print(f"Results saved to: {PLOTS_DIR}")
print(f"Probes saved to: {PROBES_DIR}")
print(f"{'='*80}")
