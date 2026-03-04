"""
Token-Level Linear Probing on Pre-Generated CoT Traces

This script analyzes where latent variables emerge during chain-of-thought reasoning
by training linear probes on synthetic variations of pre-generated CoT traces.

Workflow:
1. Load pre-generated CoT traces from disk
2. For selected traces, truncate at the point where hidden variable appears
3. Generate many variations by substituting numbers while keeping token count identical  
4. Train linear probes (Ridge regression) for each token position starting from "Answer"
5. Probe ALL layer activations to find where the hidden variable emerges

Example: "A 17 kg runner has 2.388e+04 Joules... Answer (step-by-step): 1/2mv^2 = 2.388e+04 J v = "
We swap 17 and 2.388e+04 (ensuring same tokenization) to create training examples.
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
EXPERIMENT = "velocity"  # Options: "velocity", "current"
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
TRACES_DIR = Path(f"/home/wuroderi/scratch/reasoning_traces/Qwen2.5-32B/{EXPERIMENT}")
TRACES_METADATA_FILE = TRACES_DIR / "traces_metadata.json"
PLOTS_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_linprob_cot_{EXPERIMENT}")
PLOTS_DIR.mkdir(exist_ok=True)
PROBES_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/probes_linprob_cot_{EXPERIMENT}")
PROBES_DIR.mkdir(exist_ok=True)

# Data Configuration
TRACE_INDICES = [0, 1, 2, 3, 4]  # Which pre-generated traces to use as base CoT outputs (will run separate experiments for each)
TRAIN_RATIO = 0.8  # 80% train, 20% validation
NUM_VARIATIONS_PER_TRACE = 200  # Target number of synthetic variations per base trace

# Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis Configuration
LAYERS_TO_PROBE = list(range(64))  # Probe ALL 64 layers
RIDGE_ALPHA = 1.0  # Ridge regression regularization strength

print(f"="*80)
print(f"TOKEN-LEVEL LINEAR PROBING ON PRE-GENERATED COT: {EXPERIMENT.upper()}")
print(f"="*80)
print(f"Model: {MODEL_PATH}")
print(f"Traces dir: {TRACES_DIR}")
print(f"Device: {device}")
print(f"Plots directory: {PLOTS_DIR}")
print(f"Trace indices: {TRACE_INDICES}")
print(f"Probing layers: {len(LAYERS_TO_PROBE)} layers")
print()

# ==========================================
# LOAD MODEL AND TRACES
# ==========================================

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Automatically distribute across GPUs
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load with move_to_device=False to prevent TransformerLens from moving the distributed model
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-32B",
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    fold_ln=False,  # Disable folding for multi-GPU compatibility
    center_writing_weights=False,  # Disable centering for multi-GPU compatibility
    fold_value_biases=False,  # Disable value bias folding for multi-GPU compatibility
    move_to_device=False,  # Don't move model - it's already distributed
    load_state_dict=False  # Don't reload weights - already loaded in hf_model
)

# Ensure embedding layer is on a GPU device for multi-GPU setup
if model.embed.W_E.device.type == 'cpu':
    # Move embedding to first available GPU
    model.embed = model.embed.to('cuda:0')
    print("Moved embedding layer to cuda:0")

# Also move positional embedding if it exists
if hasattr(model, 'pos_embed') and model.pos_embed.W_pos.device.type == 'cpu':
    model.pos_embed = model.pos_embed.to('cuda:0')
    print("Moved positional embedding to cuda:0")

print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dimensions")
print(f"Embedding device: {model.embed.W_E.device}\n")

# Load pre-generated traces
print("Loading pre-generated traces...")
with open(TRACES_METADATA_FILE, 'r') as f:
    all_traces = json.load(f)
print(f"Loaded {len(all_traces)} traces from {TRACES_METADATA_FILE}")
print(f"Will run {len(TRACE_INDICES)} separate experiments, one for each base trace CoT output")
print(f"Each experiment will use all {len(all_traces)} traces to generate synthetic variations")
print()

# ==========================================
# GENERATE SYNTHETIC VARIATIONS
# ==========================================

def find_number_in_scientific_notation(text, number):
    """Find scientific notation representation of a number in text."""
    # Try various scientific notation formats
    sci_patterns = [
        f"{number:.3e}",  # Standard: 2.388e+04
        f"{number:.2e}",  # Two decimals
        f"{number:.4e}",  # Four decimals
        f"{number:.1e}",  # One decimal
    ]
    
    for pattern in sci_patterns:
        if pattern in text:
            return pattern
    
    # Also try without the + sign
    for pattern in sci_patterns:
        pattern_no_plus = pattern.replace('+', '')
        if pattern_no_plus in text:
            return pattern_no_plus
            
    return None

def truncate_at_velocity_variable(generated_text):
    """
    Truncate the generated text at the point where the velocity value appears.
    Keeps everything up to and including "v = " but stops before the actual number.
    Example: "...1/2mv^2 = 2.388e+04 J v = 52.8 m/s..." -> "...1/2mv^2 = 2.388e+04 J v = "
    """
    # Common patterns where velocity value appears
    patterns = [
        r'v\s*=\s*',  # "v = "
        r'velocity\s*=\s*',  # "velocity = "
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            # Include the pattern itself (e.g., "v = ") but stop there
            truncation_point = match.end()
            truncated = generated_text[:truncation_point]
            return truncated
    
    # If no pattern found, return original (no CoT generated)
    return generated_text

def create_variations_from_traces(trace, all_traces_pool, tokenizer, model):
    """
    Create synthetic variations of a trace by substituting numbers from other traces
    while ensuring tokenization stays identical.
    """
    variations = []
    
    # Use generated_text (includes CoT) and truncate at velocity
    original_full_text = trace.get('generated_text', trace['prompt'])
    original_text = truncate_at_velocity_variable(original_full_text)
    
    original_m = trace['m']
    original_ke = trace['ke']
    original_v = trace['v']
    original_d = trace['d']
    original_format_id = trace['format_id']
    
    # Find the original KE string in scientific notation
    original_ke_str = find_number_in_scientific_notation(original_text, original_ke)
    if original_ke_str is None:
        original_ke_str = f"{original_ke:.3e}"
    
    # Count original tokens for validation
    original_tokens_obj = model.to_tokens(original_text, prepend_bos=True)
    original_n_tokens = original_tokens_obj.shape[1]
    
    # Filter to only use traces with the same format_id (same prompt structure)
    same_format_traces = [t for t in all_traces_pool if t['format_id'] == original_format_id and t['id'] != trace['id']]
    
    print(f"  Found {len(same_format_traces)} traces with same format_id {original_format_id}")
    
    successful_variations = 0
    
    for other_trace in same_format_traces:
        # Get numbers from other trace
        m_new = other_trace['m']
        v_new = other_trace['v']
        d_new = other_trace['d']
        ke_new = other_trace['ke']
        
        # Get the other trace's truncated text to extract its KE format
        other_full_text = other_trace.get('generated_text', other_trace['prompt'])
        other_text = truncate_at_velocity_variable(other_full_text)
        
        # Find KE string format from other trace
        ke_new_str = find_number_in_scientific_notation(other_text, ke_new)
        if ke_new_str is None:
            ke_new_str = f"{ke_new:.3e}"
        
        # Create new text by substitution (in both question and CoT parts)
        new_text = original_text
        new_text = new_text.replace(f" {original_m} kg", f" {m_new} kg")
        new_text = new_text.replace(original_ke_str, ke_new_str)
        new_text = new_text.replace(f" {original_d} m", f" {d_new} m")
        
        # Verify token count stayed the same
        new_tokens_obj = model.to_tokens(new_text, prepend_bos=True)
        new_n_tokens = new_tokens_obj.shape[1]
        
        if new_n_tokens != original_n_tokens:
            print(f"    SKIP: Token count mismatch! Original={original_n_tokens}, New={new_n_tokens}")
            print(f"          m: {original_m}->{m_new}, ke: {original_ke_str}->{ke_new_str}, d: {original_d}->{d_new}")
            continue
            
        # Skip if values didn't actually change
        if m_new == original_m and ke_new == original_ke and d_new == original_d:
            continue
        
        variations.append({
            'prompt': new_text,  # This now includes CoT up to "v = "
            'm': m_new,
            'ke': ke_new,
            'v': v_new,  # This is our target variable!
            'd': d_new,
            'original_trace_id': trace['id'],
            'source_trace_id': other_trace['id'],
            'n_tokens': new_n_tokens
        })
        successful_variations += 1
    
    print(f"  Generated {successful_variations} valid variations")
    
    return variations

def extract_activations_all_layers(prompts, model, layers, batch_size=8):
    """
    Extract activations from all tokens for all specified layers.
    Returns dict mapping layer -> numpy array [total_tokens, d_model]
    """
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    all_layer_activations = {layer: [] for layer in layers}
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
        
        # Stack and move to device
        batch_tokens = torch.cat(padded_tokens, dim=0).to(embed_device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda name: name in hook_names
            )
        
        # Extract activations for each layer
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            batch_acts = cache[hook_name]  # [batch_size, seq_len, d_model]
            
            # Collect only non-padded tokens
            for j in range(len(batch_prompts)):
                n_tokens = batch_token_lengths[j]
                prompt_acts = batch_acts[j, :n_tokens].cpu().float()
                all_layer_activations[layer].append(prompt_acts)
        
        # Track token counts for first layer (same for all)
        if i == 0 or len(all_token_counts) < len(prompts):
            all_token_counts.extend(batch_token_lengths)
    
    # Concatenate activations
    activations_dict = {
        layer: torch.cat(all_layer_activations[layer], dim=0).numpy()
        for layer in layers
    }
    
    return activations_dict, all_token_counts

def find_answer_token_position(prompt, tokenizer, model):
    """Find the token position where 'Answer' starts."""
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    
    # Look for "Answer" keyword
    for i in range(len(tokens)):
        token_str = model.to_string(tokens[i])
        if "Answer" in token_str or "answer" in token_str:
            return i
    
    # Fallback: return middle of sequence
    return len(tokens) // 2

print("Generating synthetic variations for each trace...")
print("="*80)

# ==========================================
# MAIN EXPERIMENT LOOP
# ==========================================

for trace_idx in TRACE_INDICES:
    if trace_idx >= len(all_traces):
        print(f"WARNING: Trace index {trace_idx} out of range, skipping")
        continue
    
    trace = all_traces[trace_idx]
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: TRACE {trace['id']} (Index {trace_idx}) - Format {trace['format_id']}")
    print(f"{'='*80}")
    print(f"Original values: m={trace['m']} kg, ke={trace['ke']:.3e} J, v={trace['v']} m/s, d={trace['d']} m")
    
    # Create trace-specific directories
    trace_plots_dir = PLOTS_DIR / f"trace_{trace_idx}"
    trace_plots_dir.mkdir(exist_ok=True)
    trace_probes_dir = PROBES_DIR / f"trace_{trace_idx}"
    trace_probes_dir.mkdir(exist_ok=True)
    
    # Get full generated text and truncate at velocity variable
    original_full_text = trace.get('generated_text', trace['prompt'])
    original_truncated = truncate_at_velocity_variable(original_full_text)
    
    print(f"\nOriginal full text (truncated at 'v = '):")
    print(f"  {original_truncated[:200]}..." if len(original_truncated) > 200 else f"  {original_truncated}")
    
    # Tokenize original truncated text
    original_tokens_obj = model.to_tokens(original_truncated, prepend_bos=True)
    original_token_strs = model.to_str_tokens(original_truncated, prepend_bos=True)
    print(f"\nOriginal tokenization ({original_tokens_obj.shape[1]} tokens)")
    
    # Generate variations using numbers from ALL traces
    print(f"\nGenerating variations using all {len(all_traces)} traces...")
    variations = create_variations_from_traces(trace, all_traces, tokenizer, model)
    
    print(f"\n{'-'*80}")
    print(f"GENERATED VARIATIONS ({len(variations)} total)")
    print(f"{'-'*80}")
    
    # Show a few example variations
    for var_idx, var in enumerate(variations[:5]):  # Show first 5 for brevity
        print(f"\nVariation {var_idx + 1} (from trace {var['source_trace_id']}):")
        print(f"  Values: m={var['m']} kg, ke={var['ke']:.3e} J, v={var['v']} m/s")
        print(f"  Token count: {var['n_tokens']}")
    
    if len(variations) > 5:
        print(f"\n... and {len(variations) - 5} more variations")
    
    # ==========================================
    # EXTRACT ACTIVATIONS
    # ==========================================

    print("\nExtracting activations from all variations...")
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
    train_velocities = np.array([v['v'] for v in train_variations])

    print("\nExtracting training activations...")
    train_activations_dict, train_token_counts = extract_activations_all_layers(
        train_prompts, model, LAYERS_TO_PROBE, batch_size=4
    )

    print(f"Extracted training activations: {train_activations_dict[0].shape}")

    # Extract activations for validation data  
    val_prompts = [v['prompt'] for v in val_variations]
    val_velocities = np.array([v['v'] for v in val_variations])

    print("\nExtracting validation activations...")
    val_activations_dict, val_token_counts = extract_activations_all_layers(
        val_prompts, model, LAYERS_TO_PROBE, batch_size=4
    )

    print(f"Extracted validation activations: {val_activations_dict[0].shape}")

    # ==========================================
    # TRAIN PER-TOKEN LINEAR PROBES
    # ==========================================

    print("\n" + "="*80)
    print("TRAINING LINEAR PROBES FOR EACH TOKEN POSITION")
    print("="*80)

    # Find "Answer" position in first prompt to determine where to start probing
    example_prompt = train_prompts[0]
    answer_token_pos = find_answer_token_position(example_prompt, tokenizer, model)
    print(f"Starting probes from token position {answer_token_pos} (Answer keyword)")
    print(f"Example prompt tokens: {model.to_str_tokens(example_prompt, prepend_bos=True)[:answer_token_pos+5]}")
    print()

    # Determine max sequence length across all prompts
    max_seq_len = max(train_token_counts + val_token_counts)
    print(f"Maximum sequence length: {max_seq_len} tokens")

    # Get token strings for plotting (from first example)
    example_token_strs = model.to_str_tokens(example_prompt, prepend_bos=True)
    print(f"Example tokens: {example_token_strs[:20]}...")
    print()

    # Initialize storage for probes and results
    # probes[token_pos][layer] = Ridge probe
    probes = defaultdict(dict)

    # results[token_pos][layer] = {train_r2, val_r2, train_mae, val_mae}
    results = defaultdict(dict)

    # Train probes for each token position starting from "Answer"
    for token_pos in range(answer_token_pos, max_seq_len):
        print(f"\n{'='*80}")
        print(f"TOKEN POSITION {token_pos}")
        print(f"{'='*80}")
        
        # Collect activations and labels for this token position across all sequences
        train_acts_at_pos = {layer: [] for layer in LAYERS_TO_PROBE}
        train_labels_at_pos = []
        
        # Track which training examples have this token position
        current_idx = 0
        for i, n_tokens in enumerate(train_token_counts):
            if token_pos < n_tokens:
                # This sequence has this token position
                for layer in LAYERS_TO_PROBE:
                    # Extract activation at this specific token position
                    act = train_activations_dict[layer][current_idx + token_pos]
                    train_acts_at_pos[layer].append(act)
                train_labels_at_pos.append(train_velocities[i])
            current_idx += n_tokens
        
        # Convert to arrays
        n_samples = len(train_labels_at_pos)
        if n_samples < 10:  # Need minimum samples to train
            print(f"  Skipping: only {n_samples} samples at this position")
            continue
        
        train_labels_at_pos = np.array(train_labels_at_pos)
        print(f"  Training samples at this position: {n_samples}")
        
        # Do the same for validation data
        val_acts_at_pos = {layer: [] for layer in LAYERS_TO_PROBE}
        val_labels_at_pos = []
        
        current_idx = 0
        for i, n_tokens in enumerate(val_token_counts):
            if token_pos < n_tokens:
                for layer in LAYERS_TO_PROBE:
                    act = val_activations_dict[layer][current_idx + token_pos]
                    val_acts_at_pos[layer].append(act)
                val_labels_at_pos.append(val_velocities[i])
            current_idx += n_tokens
        
        val_labels_at_pos = np.array(val_labels_at_pos)
        print(f"  Validation samples at this position: {len(val_labels_at_pos)}")
        
        # Train a probe for each layer at this token position
        for layer in LAYERS_TO_PROBE:
            train_acts_layer = np.array(train_acts_at_pos[layer])
            
            # Train Ridge regression
            probe = Ridge(alpha=RIDGE_ALPHA)
            probe.fit(train_acts_layer, train_labels_at_pos)
            
            # Store probe
            probes[token_pos][layer] = probe
            
            # Evaluate on training data
            train_preds = probe.predict(train_acts_layer)
            train_r2 = r2_score(train_labels_at_pos, train_preds)
            train_mae = mean_absolute_error(train_labels_at_pos, train_preds)
            train_mpe = np.mean(np.abs((train_preds - train_labels_at_pos) / train_labels_at_pos)) * 100
            
            # Evaluate on validation data
            if len(val_labels_at_pos) > 0:
                val_acts_layer = np.array(val_acts_at_pos[layer])
                val_preds = probe.predict(val_acts_layer)
                val_r2 = r2_score(val_labels_at_pos, val_preds)
                val_mae = mean_absolute_error(val_labels_at_pos, val_preds)
                # Calculate mean percent error: mean(|pred - true| / |true|) * 100
                val_mpe = np.mean(np.abs((val_preds - val_labels_at_pos) / val_labels_at_pos)) * 100
            else:
                val_r2 = 0.0
                val_mae = float('inf')
                val_mpe = float('inf')
            
            # Store results
            results[token_pos][layer] = {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_mpe': train_mpe,
                'val_mpe': val_mpe,
                'n_train': n_samples,
                'n_val': len(val_labels_at_pos)
            }
        
        # Print summary for this token position
        best_layer = max(LAYERS_TO_PROBE, key=lambda l: results[token_pos][l]['val_r2'])
        best_r2 = results[token_pos][best_layer]['val_r2']
        best_mae = results[token_pos][best_layer]['val_mae']
        print(f"  Best layer: {best_layer} (Val R²: {best_r2:.3f}, MAE: {best_mae:.2f})")

    print(f"\n{'='*80}")
    print("PROBE TRAINING COMPLETE")
    print(f"{'='*80}")
    print()

    # ==========================================
    # COLLECT TOP 5 PREDICTIONS & SAVE VALIDATION DATA
    # ==========================================

    print("Collecting top 5 predictions...")
    # Find top 5 best predictions across all layers and token positions
    all_predictions = []
    for token_pos in results:
        for layer in results[token_pos]:
            all_predictions.append({
                'token_pos': token_pos,
                'layer': layer,
                'val_r2': results[token_pos][layer]['val_r2'],
                'val_mae': results[token_pos][layer]['val_mae'],
                'val_mpe': results[token_pos][layer]['val_mpe']
            })
    
    # Sort by R² (descending) and take top 5
    all_predictions.sort(key=lambda x: x['val_r2'], reverse=True)
    top_5_configs = all_predictions[:5]
    
    # For each top 5 config, get actual predictions
    top_5_predictions = []
    for config in top_5_configs:
        token_pos = config['token_pos']
        layer = config['layer']
        
        # Get token string
        token_str = example_token_strs[token_pos] if token_pos < len(example_token_strs) else f"pos{token_pos}"
        
        # Get validation predictions for this config
        current_idx = 0
        val_acts_layer = []
        val_labels_layer = []
        val_indices = []
        
        for i, n_tokens in enumerate(val_token_counts):
            if token_pos < n_tokens:
                act = val_activations_dict[layer][current_idx + token_pos]
                val_acts_layer.append(act)
                val_labels_layer.append(val_velocities[i])
                val_indices.append(i)
            current_idx += n_tokens
        
        val_acts_layer = np.array(val_acts_layer)
        val_labels_layer = np.array(val_labels_layer)
        
        # Get predictions from the trained probe
        probe = probes[token_pos][layer]
        preds = probe.predict(val_acts_layer)
        
        top_5_predictions.append({
            'token_pos': token_pos,
            'token': token_str,
            'layer': layer,
            'val_r2': config['val_r2'],
            'val_mae': config['val_mae'],
            'val_mpe': config['val_mpe'],
            'predictions': preds.tolist(),
            'true_values': val_labels_layer.tolist(),
            'validation_indices': val_indices
        })
    
    # Save validation data to JSON with top 5 predictions
    print("\nSaving validation data to JSON...")
    validation_data = {
        'trace_id': trace['id'],
        'trace_idx': trace_idx,
        'original_values': {
            'm': trace['m'],
            'ke': trace['ke'],
            'v': trace['v'],
            'd': trace['d']
        },
        'variations': val_variations,
        'velocities': val_velocities.tolist(),
        'n_samples': len(val_variations),
        'top_5_predictions': top_5_predictions
    }
    val_data_file = trace_plots_dir / 'validation_data.json'
    with open(val_data_file, 'w') as f:
        json.dump(validation_data, f, indent=2)
    print(f"Saved validation data to: {val_data_file}")
    
    print("\nTop 5 Predictions:")
    for i, pred_info in enumerate(top_5_predictions):
        print(f"  {i+1}. Token {pred_info['token_pos']} ('{pred_info['token']}'), Layer {pred_info['layer']}")
        print(f"     R²={pred_info['val_r2']:.4f}, MAE={pred_info['val_mae']:.2f}, MPE={pred_info['val_mpe']:.1f}%")
    print()

    # ==========================================
    # SAVE PROBES
    # ==========================================

    print("Saving trained probes...")
    for token_pos in probes:
        for layer in probes[token_pos]:
            probe_filename = trace_probes_dir / f"probe_token{token_pos}_layer{layer}.joblib"
            joblib.dump(probes[token_pos][layer], probe_filename)
    print(f"Saved {sum(len(probes[tp]) for tp in probes)} probes to: {trace_probes_dir}")
    print()

    # ==========================================
    # VISUALIZATION
    # ==========================================

    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Create heatmap showing validation R² across token positions and layers
    token_positions = sorted(probes.keys())
    r2_matrix = np.zeros((len(LAYERS_TO_PROBE), len(token_positions)))
    mae_matrix = np.zeros((len(LAYERS_TO_PROBE), len(token_positions)))
    mpe_matrix = np.zeros((len(LAYERS_TO_PROBE), len(token_positions)))

    for i, layer in enumerate(LAYERS_TO_PROBE):
        for j, token_pos in enumerate(token_positions):
            if layer in results[token_pos]:
                r2_matrix[i, j] = results[token_pos][layer]['val_r2']
                mae_matrix[i, j] = results[token_pos][layer]['val_mae']
                mpe_matrix[i, j] = results[token_pos][layer]['val_mpe']
            else:
                r2_matrix[i, j] = np.nan
                mae_matrix[i, j] = np.nan
                mpe_matrix[i, j] = np.nan

    # Create token labels for x-axis
    def is_numeric_token(token_str):
        """Check if token represents a number."""
        # Remove spaces and check if it's a number
        cleaned = token_str.strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            # Check for scientific notation parts
            if any(c in cleaned for c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                # Check if it's mostly numeric
                numeric_chars = sum(c.isdigit() for c in cleaned)
                if numeric_chars / max(len(cleaned), 1) > 0.5:
                    return True
            return False

    token_labels = []
    for pos in token_positions:
        if pos < len(example_token_strs):
            token_str = example_token_strs[pos]
            # Truncate long tokens
            if len(token_str) > 8:
                token_str = token_str[:6] + '..'
            # Tag numeric tokens
            if is_numeric_token(token_str):
                token_labels.append(f'[NUM]')
            else:
                token_labels.append(token_str)
        else:
            token_labels.append(f'{pos}')

    # Plot 1: R² heatmap
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    im = ax.imshow(r2_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Token Position', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title(f'Linear Probe Performance (Val R²) - Trace {trace_idx}\n{EXPERIMENT.capitalize()}', 
                 fontsize=16, fontweight='bold')

    # Set ticks
    layer_tick_step = max(1, len(LAYERS_TO_PROBE) // 20)
    ax.set_yticks(range(0, len(LAYERS_TO_PROBE), layer_tick_step))
    ax.set_yticklabels([f'L{LAYERS_TO_PROBE[i]}' for i in range(0, len(LAYERS_TO_PROBE), layer_tick_step)])

    token_tick_step = max(1, len(token_positions) // 20)
    ax.set_xticks(range(0, len(token_positions), token_tick_step))
    ax.set_xticklabels([f'{token_labels[i]}' for i in range(0, len(token_positions), token_tick_step)], rotation=45, ha='right')

    # Add text annotations with values on ALL layers and ALL tokens
    text_sample_layer = 1
    text_sample_token = 1
    for i in range(0, len(LAYERS_TO_PROBE), text_sample_layer):
        for j in range(0, len(token_positions), text_sample_token):
            if not np.isnan(r2_matrix[i, j]):
                text_color = 'white' if r2_matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{r2_matrix[i, j]:.2f}', ha='center', va='center',
                       color=text_color, fontsize=5, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation R²', fontsize=13)

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'r2_heatmap_all_layers.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'r2_heatmap_all_layers.png'}")
    plt.close()

    # Plot 2: MAE heatmap
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Use log scale for MAE to see variation better
    mae_matrix_masked = np.where(np.isnan(mae_matrix), np.inf, mae_matrix)
    vmax = np.percentile(mae_matrix_masked[mae_matrix_masked < np.inf], 95)

    im = ax.imshow(mae_matrix, aspect='auto', cmap='viridis_r', vmin=0, vmax=vmax)
    ax.set_xlabel('Token Position', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title(f'Linear Probe MAE Across Layers and Token Positions\n{EXPERIMENT.capitalize()}', 
             fontsize=16, fontweight='bold')

    ax.set_yticks(range(0, len(LAYERS_TO_PROBE), layer_tick_step))
    ax.set_yticklabels([f'L{LAYERS_TO_PROBE[i]}' for i in range(0, len(LAYERS_TO_PROBE), layer_tick_step)])

    ax.set_xticks(range(0, len(token_positions), token_tick_step))
    ax.set_xticklabels([f'{token_labels[i]}' for i in range(0, len(token_positions), token_tick_step)], rotation=45, ha='right')

    # Add text annotations with values on ALL layers and ALL tokens
    for i in range(0, len(LAYERS_TO_PROBE), text_sample_layer):
        for j in range(0, len(token_positions), text_sample_token):
            if not np.isnan(mae_matrix[i, j]) and mae_matrix[i, j] < np.inf:
                text_color = 'white' if mae_matrix[i, j] > vmax * 0.5 else 'black'
                ax.text(j, i, f'{mae_matrix[i, j]:.1f}', ha='center', va='center',
                       color=text_color, fontsize=5, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Absolute Error', fontsize=13)

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'mae_heatmap_all_layers.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'mae_heatmap_all_layers.png'}")
    plt.close()

    # Plot 2b: MPE heatmap (NEW)
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Use percentile for MPE to see variation better
    mpe_matrix_masked = np.where(np.isnan(mpe_matrix), np.inf, mpe_matrix)
    vmax_mpe = np.percentile(mpe_matrix_masked[mpe_matrix_masked < np.inf], 95)

    im = ax.imshow(mpe_matrix, aspect='auto', cmap='viridis_r', vmin=0, vmax=vmax_mpe)
    ax.set_xlabel('Token Position', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title(f'Linear Probe Mean Percent Error Across Layers and Token Positions\n{EXPERIMENT.capitalize()}', 
             fontsize=16, fontweight='bold')

    ax.set_yticks(range(0, len(LAYERS_TO_PROBE), layer_tick_step))
    ax.set_yticklabels([f'L{LAYERS_TO_PROBE[i]}' for i in range(0, len(LAYERS_TO_PROBE), layer_tick_step)])

    ax.set_xticks(range(0, len(token_positions), token_tick_step))
    ax.set_xticklabels([f'{token_labels[i]}' for i in range(0, len(token_positions), token_tick_step)], rotation=45, ha='right')

    # Add text annotations with values on ALL layers and ALL tokens
    for i in range(0, len(LAYERS_TO_PROBE), text_sample_layer):
        for j in range(0, len(token_positions), text_sample_token):
            if not np.isnan(mpe_matrix[i, j]) and mpe_matrix[i, j] < np.inf:
                text_color = 'white' if mpe_matrix[i, j] > vmax_mpe * 0.5 else 'black'
                ax.text(j, i, f'{mpe_matrix[i, j]:.0f}%', ha='center', va='center',
                       color=text_color, fontsize=5, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Percent Error (%)', fontsize=13)

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'mpe_heatmap_all_layers.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'mpe_heatmap_all_layers.png'}")
    plt.close()

    # Plot 3: Max R² across layers for each token position
    max_r2_per_token = []
    max_layer_per_token = []

    for token_pos in token_positions:
        valid_r2s = [(layer, results[token_pos][layer]['val_r2']) 
                     for layer in LAYERS_TO_PROBE if layer in results[token_pos]]
        if valid_r2s:
            best_layer, best_r2 = max(valid_r2s, key=lambda x: x[1])
            max_r2_per_token.append(best_r2)
            max_layer_per_token.append(best_layer)
        else:
            max_r2_per_token.append(0)
            max_layer_per_token.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Top plot: Max R² per token
    ax1.plot(token_positions, max_r2_per_token, 'b-', linewidth=2)
    ax1.set_xlabel('Token Position', fontsize=13)
    ax1.set_ylabel('Max Validation R² (across layers)', fontsize=13)
    ax1.set_title('Best Probe Performance at Each Token Position', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Which layer achieves max R²
    ax2.scatter(token_positions, max_layer_per_token, c=max_r2_per_token, cmap='viridis', s=50, alpha=0.7)
    ax2.set_xlabel('Token Position', fontsize=13)
    ax2.set_ylabel('Best Layer', fontsize=13)
    ax2.set_title('Which Layer Achieves Best Performance at Each Position', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Validation R²', fontsize=12)

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'max_r2_per_token.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'max_r2_per_token.png'}")
    plt.close()

    # Plot 4: Selected layers comparison
    selected_layers_to_plot = [0, 15, 31, 47, 63]  # Early, middle, late layers
    selected_layers_to_plot = [l for l in selected_layers_to_plot if l in LAYERS_TO_PROBE]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for layer in selected_layers_to_plot:
        r2_values = [results[tok][layer]['val_r2'] if layer in results[tok] else 0 
                     for tok in token_positions]
        ax.plot(token_positions, r2_values, linewidth=2, label=f'Layer {layer}', alpha=0.8)

    ax.set_xlabel('Token Position', fontsize=13)
    ax.set_ylabel('Validation R²', fontsize=13)
    ax.set_title('Probe Performance Across Token Positions (Selected Layers)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(trace_plots_dir / 'r2_selected_layers.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {trace_plots_dir / 'r2_selected_layers.png'}")
    plt.close()

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    # Save summary statistics
    summary = {
        'experiment': EXPERIMENT,
        'trace_id': trace['id'],
        'trace_idx': trace_idx,
        'n_total_traces': len(all_traces),
        'n_variations': len(variations),
        'n_train': len(train_variations),
        'n_val': len(val_variations),
        'layers_probed': LAYERS_TO_PROBE,
        'token_positions': token_positions,
        'answer_token_position': answer_token_pos,
        'best_results': {}
    }

    # Find overall best performance
    best_overall_r2 = -1
    best_overall_config = None

    for token_pos in token_positions:
        for layer in LAYERS_TO_PROBE:
            if layer in results[token_pos]:
                r2 = results[token_pos][layer]['val_r2']
                if r2 > best_overall_r2:
                    best_overall_r2 = r2
                    best_overall_config = {
                        'token_pos': token_pos,
                        'layer': layer,
                        'val_r2': r2,
                        'val_mae': results[token_pos][layer]['val_mae'],
                        'train_r2': results[token_pos][layer]['train_r2']
                    }

    summary['best_results']['overall'] = best_overall_config

    # Save to JSON
    results_file = trace_plots_dir / 'probe_results_summary.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {results_file}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE FOR TRACE {trace_idx}")
    print(f"{'='*80}")
    print(f"Best overall performance:")
    print(f"  Token position: {best_overall_config['token_pos']}")
    print(f"  Layer: {best_overall_config['layer']}")
    print(f"  Validation R²: {best_overall_config['val_r2']:.4f}")
    print(f"  Validation MAE: {best_overall_config['val_mae']:.2f}")
    print(f"  ('Answer' starts at token {answer_token_pos})")
    print(f"\nAll visualizations saved to: {trace_plots_dir}")
    print(f"\nAll probes saved to: {trace_probes_dir}")
    print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"{'='*80}")
print(f"Processed {len(TRACE_INDICES)} trace(s)")
print(f"Results saved to: {PLOTS_DIR}")
print(f"Probes saved to: {PROBES_DIR}")
print(f"{'='*80}")

