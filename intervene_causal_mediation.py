"""
Causal Mediation Analysis for Velocity Representations

This script performs causal mediation analysis to identify which activations
are most causal for determining velocity values in the model's reasoning.

Methodology:
1. Pair prompts with matching format_id (source and base)
2. Truncate base prompt at location where velocity is output in CoT
3. Create counterfactual source by replacing base's mass/KE values with source values
4. For each (layer, token) position:
   - Patch activation from counterfactual source into base
   - Measure log probability difference: log(P(source_velocity_token)) - log(P(base_velocity_token))
5. Generate heatmap showing which positions are most causal

The heatmap shows which layer/token combinations most influence the model's
prediction of velocity values.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B"
TRACES_DIR = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity")
TRACES_METADATA_FILE = TRACES_DIR / "traces_metadata.json"
OUTPUT_DIR = Path("/home/wuroderi/links/scratch/causal_plots")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Experiment Configuration
N_PROMPT_PAIRS = None  # None = use all available pairs
#LAYERS_TO_TEST = list(range(0, 64, 4))  # Test every 4th layer to reduce computation
LAYERS_TO_TEST = list(range(0, 64))

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"="*80)
print(f"CAUSAL MEDIATION ANALYSIS")
print(f"="*80)
print(f"Model: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Prompt pairs: {N_PROMPT_PAIRS}")
print(f"Layers to test: {len(LAYERS_TO_TEST)}")
print(f"Output: {OUTPUT_DIR}")
print()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {model.config.num_hidden_layers} layers")
print()

# ==========================================
# LOAD TRACES
# ==========================================

print("Loading traces...")
with open(TRACES_METADATA_FILE, 'r') as f:
    traces = json.load(f)

print(f"Loaded {len(traces)} traces")

# Group traces by format_id
traces_by_format = defaultdict(list)
for trace in traces:
    traces_by_format[trace['format_id']].append(trace)

min_traces_per_format = min(len(traces) for traces in traces_by_format.values())
n_pairs_per_format = min_traces_per_format // 2

# Create prompt pairs
prompt_pairs = []
for format_id, format_traces in traces_by_format.items():
    pairs_to_create = n_pairs_per_format if N_PROMPT_PAIRS is None else min(n_pairs_per_format, N_PROMPT_PAIRS // len(traces_by_format) + 1)
    for i in range(pairs_to_create):
        if N_PROMPT_PAIRS is not None and len(prompt_pairs) >= N_PROMPT_PAIRS:
            break
        source_trace = format_traces[i]
        base_trace = format_traces[i + n_pairs_per_format]
        
        prompt_pairs.append({
            'source_trace': source_trace,
            'base_trace': base_trace,
            'format_id': format_id
        })
    if N_PROMPT_PAIRS is not None and len(prompt_pairs) >= N_PROMPT_PAIRS:
        break

print(f"Created {len(prompt_pairs)} prompt pairs")
print()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_prompt_from_trace(trace_text):
    """Extract just the prompt (ends with 'Answer (step-by-step): ')"""
    marker = "Answer (step-by-step): "
    if marker in trace_text:
        end_pos = trace_text.find(marker) + len(marker)
        return trace_text[:end_pos]
    return trace_text

def find_velocity_in_text(text: str, velocity: float) -> Optional[int]:
    """Find position where velocity value appears in text."""
    # Convert velocity to string representations
    velocity_strs = [
        str(int(velocity)),
        f"{velocity:.1f}",
        f"{velocity:.2f}",
    ]
    
    for vel_str in velocity_strs:
        # Find first digit of velocity
        pattern = rf'\b{vel_str[0]}'
        match = re.search(pattern, text)
        if match:
            # Verify the full number follows
            remaining = text[match.start():]
            if remaining.startswith(vel_str):
                return match.start()
    
    return None

def format_value_like_reference(reference: str, value: float) -> str:
    """Format a numeric value using the same textual style as a reference string."""
    if 'e' in reference.lower():
        mantissa, exponent = re.split(r'[eE]', reference)
        decimal_places = len(mantissa.split('.')[1]) if '.' in mantissa else 0
        formatted = f"{value:.{decimal_places}e}"
        return formatted.upper() if 'E' in reference else formatted

    if '.' in reference:
        decimal_places = len(reference.split('.')[1])
        return f"{value:.{decimal_places}f}"

    return f"{value:.0f}"

def replace_formatted_value_occurrences(text: str, old_value: float, new_value: float,
                                       offset: int = 0) -> Tuple[str, List[Tuple[int, int, str, str]]]:
    """Replace numeric substrings that match a value when rendered in their existing format."""
    numeric_pattern = re.compile(r'(?<![\w.])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])')

    replacements = []
    result_parts = []
    last_end = 0

    for match in numeric_pattern.finditer(text):
        token = match.group()
        if format_value_like_reference(token, old_value) != token:
            continue

        replacement = format_value_like_reference(token, new_value)

        result_parts.append(text[last_end:match.start()])
        result_parts.append(replacement)
        replacements.append((offset + match.start(), offset + match.end(), token, replacement))
        last_end = match.end()

    if not replacements:
        return text, []

    result_parts.append(text[last_end:])
    return ''.join(result_parts), replacements

def replace_all_numeric_occurrences(text: str, old_ke: float, new_ke: float, 
                                   old_mass: float, new_mass: float) -> Tuple[str, List[Tuple[int, int, str, str]]]:
    """
    Replace all occurrences of KE and mass values in text, plus derived v^2 values in the CoT.
    Handles both scientific notation and decimal representations.
    First replaces in prompt, then only searches in CoT (after 'Answer (step-by-step): ').
    
    Returns:
        Tuple of (modified_text, list of (start_pos, end_pos, old_value, new_value))
    """
    # Find where CoT starts
    answer_marker = "Answer (step-by-step): "
    if answer_marker in text:
        marker_pos = text.find(answer_marker) + len(answer_marker)
        prompt_part = text[:marker_pos]
        cot_part = text[marker_pos:]
    else:
        prompt_part = text
        cot_part = ""
        marker_pos = len(text)
    
    replacements = []  # Track what was replaced
    
    # Convert to different representations
    old_ke_formats = [
        f"{old_ke:.3e}",
        str(int(old_ke)) if old_ke == int(old_ke) else str(old_ke),
    ]
    new_ke_formats = [
        f"{new_ke:.3e}",
        str(int(new_ke)) if new_ke == int(new_ke) else str(new_ke),
    ]

    old_v_squared = (2 * old_ke) / old_mass
    new_v_squared = (2 * new_ke) / new_mass
    
    old_mass_str = str(int(old_mass))
    new_mass_str = str(int(new_mass))
    
    # Replace in prompt part (all occurrences)
    result_prompt = prompt_part
    for old_fmt, new_fmt in zip(old_ke_formats, new_ke_formats):
        if old_fmt in result_prompt:
            # Find all occurrences
            start = 0
            while True:
                pos = result_prompt.find(old_fmt, start)
                if pos == -1:
                    break
                replacements.append((pos, pos + len(old_fmt), old_fmt, new_fmt))
                result_prompt = result_prompt[:pos] + new_fmt + result_prompt[pos + len(old_fmt):]
                start = pos + len(new_fmt)
    
    # Replace mass in prompt
    for match in re.finditer(rf'\b{old_mass_str}\b', result_prompt):
        replacements.append((match.start(), match.end(), old_mass_str, new_mass_str))
    result_prompt = re.sub(rf'\b{old_mass_str}\b', new_mass_str, result_prompt)
    
    # Replace in CoT part (all occurrences)
    result_cot = cot_part
    cot_offset = marker_pos
    
    for old_fmt, new_fmt in zip(old_ke_formats, new_ke_formats):
        if old_fmt in result_cot:
            start = 0
            while True:
                pos = result_cot.find(old_fmt, start)
                if pos == -1:
                    break
                replacements.append((cot_offset + pos, cot_offset + pos + len(old_fmt), old_fmt, new_fmt))
                result_cot = result_cot[:pos] + new_fmt + result_cot[pos + len(old_fmt):]
                start = pos + len(new_fmt)
    
    # Replace mass in CoT
    for match in re.finditer(rf'\b{old_mass_str}\b', result_cot):
        replacements.append((cot_offset + match.start(), cot_offset + match.end(), old_mass_str, new_mass_str))
    result_cot = re.sub(rf'\b{old_mass_str}\b', new_mass_str, result_cot)

    # Replace explicit derived v^2 values in CoT while preserving the original number format.
    result_cot, v_squared_replacements = replace_formatted_value_occurrences(
        result_cot,
        old_value=old_v_squared,
        new_value=new_v_squared,
        offset=cot_offset
    )
    replacements.extend(v_squared_replacements)
    
    return result_prompt + result_cot, replacements

def get_next_token_logprobs(model, tokenizer, text: str, target_tokens: List[str]) -> Dict[str, float]:
    """
    Get log probabilities for target tokens as the next token.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        target_tokens: List of token strings to get probabilities for
    
    Returns:
        Dictionary mapping token string to log probability
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    results = {}
    for token_str in target_tokens:
        # Tokenize single character
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = token_ids[0]
            results[token_str] = log_probs[token_id].item()
        else:
            # If it tokenizes to multiple tokens, take the first
            token_id = token_ids[0]
            results[token_str] = log_probs[token_id].item()
    
    return results

def extract_and_patch_activation(model, tokenizer, source_text: str, base_text: str,
                                layer: int, token_pos: int) -> str:
    """
    Extract activation from source at (layer, token_pos) and patch into base.
    Returns the modified base text for probability computation.
    
    Note: We don't actually modify text, just return base_text.
    The patching happens during the forward pass.
    """
    # This function will be used differently - we'll use hooks during forward pass
    return base_text

def compute_causal_effect(model, tokenizer, source_text: str, base_text: str,
                         layer: int, token_pos: int, 
                         source_velocity_digit: str, base_velocity_digit: str) -> float:
    """
    Compute causal effect of patching (layer, token_pos) from source to base.
    
    Returns:
        log(P(source_digit)) - log(P(base_digit)) after patching
    """
    # First, get source activation
    source_inputs = tokenizer(source_text, return_tensors="pt").to(model.device)
    source_input_ids = source_inputs['input_ids']
    
    activation_storage = {}
    
    def source_hook(module, input, output):
        hidden_states = output[0]
        if token_pos < hidden_states.shape[1]:
            activation_storage['activation'] = hidden_states[0, token_pos].detach().cpu()
    
    # Extract activation from source
    handle = model.model.layers[layer].register_forward_hook(source_hook)
    with torch.no_grad():
        _ = model(source_input_ids)
    handle.remove()
    
    if 'activation' not in activation_storage:
        return 0.0
    
    source_activation = activation_storage['activation']
    
    # Now patch into base and get probabilities
    base_inputs = tokenizer(base_text, return_tensors="pt").to(model.device)
    base_input_ids = base_inputs['input_ids']
    
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
        else:
            hidden_states = output.clone()
        
        if token_pos < hidden_states.shape[1]:
            hidden_states[0, token_pos] = source_activation.to(hidden_states.device)
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    handle = model.model.layers[layer].register_forward_hook(patch_hook)
    
    with torch.no_grad():
        outputs = model(base_input_ids)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    handle.remove()
    
    # Get log probs for target digits
    source_token_ids = tokenizer.encode(source_velocity_digit, add_special_tokens=False)
    base_token_ids = tokenizer.encode(base_velocity_digit, add_special_tokens=False)
    
    if len(source_token_ids) == 0 or len(base_token_ids) == 0:
        return 0.0
    
    source_log_prob = log_probs[source_token_ids[0]].item()
    base_log_prob = log_probs[base_token_ids[0]].item()
    
    # Clean up
    del source_input_ids, base_input_ids, source_activation
    torch.cuda.empty_cache()
    
    return source_log_prob - base_log_prob

# ==========================================
# MAIN ANALYSIS
# ==========================================

print("Running causal mediation analysis...")
print("="*80)

all_metadata = []

for pair_idx, pair in enumerate(prompt_pairs):
    source_trace = pair['source_trace']
    base_trace = pair['base_trace']
    
    print(f"\n{'='*80}")
    print(f"PAIR {pair_idx + 1}/{len(prompt_pairs)}")
    print(f"{'='*80}")
    print(f"Source: trace {source_trace['id']}, velocity={source_trace['v']:.1f}, mass={source_trace['m']}, KE={source_trace['ke']:.3e}")
    print(f"Base:   trace {base_trace['id']}, velocity={base_trace['v']:.1f}, mass={base_trace['m']}, KE={base_trace['ke']:.3e}")
    print(f"Format ID: {pair['format_id']}")
    print()
    
    # Extract prompt
    base_prompt = extract_prompt_from_trace(base_trace['generated_text'])
    
    # Find where velocity appears in base CoT
    base_full_text = base_trace['generated_text']
    velocity_pos = find_velocity_in_text(base_full_text, base_trace['v'])
    
    if velocity_pos is None:
        print("  WARNING: Could not find velocity in base text. Skipping.")
        continue
    
    # Truncate base text up to velocity
    base_truncated = base_full_text[:velocity_pos]
    print(f"  Truncated base at character position {velocity_pos}")
    print(f"  Base truncated text:\n    ...{base_truncated}") # Want to see all of it
    print()
    
    # Create counterfactual source by replacing values
    source_counterfactual, replacements = replace_all_numeric_occurrences(
        base_truncated,
        old_ke=base_trace['ke'],
        new_ke=source_trace['ke'],
        old_mass=base_trace['m'],
        new_mass=source_trace['m']
    )
    
    print(f"  Created counterfactual source with {len(replacements)} substitutions:")
    for start, end, old_val, new_val in replacements:
        print(f"    Position {start}-{end}: '{old_val}' → '{new_val}'")
    print()
    print(f"  Source counterfactual text:\n    ...{source_counterfactual}")
    print()
    
    # Get first digits of velocities
    source_velocity_digit = str(int(source_trace['v']))[0]
    base_velocity_digit = str(int(base_trace['v']))[0]
    
    print(f"  Target velocity digits: source='{source_velocity_digit}' (from {source_trace['v']:.1f}), base='{base_velocity_digit}' (from {base_trace['v']:.1f})")
    print()
    
    # Tokenize both to get token-level substitution info
    base_tokens = tokenizer(base_truncated, return_tensors="pt", add_special_tokens=True)
    source_tokens = tokenizer(source_counterfactual, return_tensors="pt", add_special_tokens=True)
    
    base_token_ids = base_tokens['input_ids'][0]
    source_token_ids = source_tokens['input_ids'][0]
    n_tokens = base_token_ids.shape[0]
    
    # Get token strings
    base_token_strings = [tokenizer.decode([tid.item()]) for tid in base_token_ids]
    source_token_strings = [tokenizer.decode([tid.item()]) for tid in source_token_ids]
    
    # Find which tokens differ
    differing_tokens = []
    for i in range(min(len(base_token_ids), len(source_token_ids))):
        if base_token_ids[i] != source_token_ids[i]:
            differing_tokens.append({
                'position': i,
                'base_token': base_token_strings[i],
                'source_token': source_token_strings[i],
                'base_token_id': base_token_ids[i].item(),
                'source_token_id': source_token_ids[i].item()
            })
    
    print(f"  Tokenization:")
    print(f"    Base:   {n_tokens} tokens")
    print(f"    Source: {len(source_token_ids)} tokens")
    print(f"    Differing tokens: {len(differing_tokens)} positions")
    if differing_tokens:
        print(f"    Substituted token positions and values:")
        for diff in differing_tokens[:10]:  # Show first 10
            print(f"      Token {diff['position']}: '{diff['base_token']}' → '{diff['source_token']}'")
        if len(differing_tokens) > 10:
            print(f"      ... and {len(differing_tokens) - 10} more")
    print()
    
    print(f"  Analyzing {n_tokens} tokens × {len(LAYERS_TO_TEST)} layers...")
    print()
    
    # Initialize heatmap data
    heatmap_data = np.zeros((len(LAYERS_TO_TEST), n_tokens))
    
    # Compute causal effects
    for layer_idx, layer in enumerate(LAYERS_TO_TEST):
        print(f"    Layer {layer}...", end='', flush=True)
        for token_idx in range(n_tokens):
            causal_effect = compute_causal_effect(
                model, tokenizer,
                source_counterfactual, base_truncated,
                layer, token_idx,
                source_velocity_digit, base_velocity_digit
            )
            heatmap_data[layer_idx, token_idx] = causal_effect
        print(" done")
    
    # Create heatmap
    plt.figure(figsize=(max(20, n_tokens * 0.3), len(LAYERS_TO_TEST) * 0.3))
    
    sns.heatmap(
        heatmap_data,
        xticklabels=base_token_strings,
        yticklabels=LAYERS_TO_TEST,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'log(P(source)) - log(P(base))'}
    )
    
    plt.xlabel('Token')
    plt.ylabel('Layer')
    plt.title(f'Causal Mediation Analysis: Pair {pair_idx + 1}\n'
              f'Source v={source_trace["v"]}, Base v={base_trace["v"]}')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    
    # Save plot
    plot_file = OUTPUT_DIR / f"causal_heatmap_pair_{pair_idx + 1:03d}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmap to {plot_file}")
    
    # Also save data as numpy array
    data_file = OUTPUT_DIR / f"causal_data_pair_{pair_idx + 1:03d}.npz"
    np.savez(
        data_file,
        heatmap_data=heatmap_data,
        layers=LAYERS_TO_TEST,
        token_strings=base_token_strings,
        source_velocity=source_trace['v'],
        base_velocity=base_trace['v']
    )
    
    # Save metadata
    pair_metadata = {
        'pair_idx': pair_idx + 1,
        'source_trace_id': source_trace['id'],
        'base_trace_id': base_trace['id'],
        'format_id': pair['format_id'],
        'source_velocity': source_trace['v'],
        'base_velocity': base_trace['v'],
        'source_mass': source_trace['m'],
        'base_mass': base_trace['m'],
        'source_ke': source_trace['ke'],
        'base_ke': base_trace['ke'],
        'source_distance': source_trace['d'],
        'base_distance': base_trace['d'],
        'target_digits': {
            'source': source_velocity_digit,
            'base': base_velocity_digit
        },
        'truncation_position': velocity_pos,
        'n_tokens': n_tokens,
        'n_layers_tested': len(LAYERS_TO_TEST),
        'layers_tested': LAYERS_TO_TEST,
        'n_substitutions': len(replacements),
        'substitutions': [{'start': s, 'end': e, 'old': o, 'new': n} for s, e, o, n in replacements],
        'differing_tokens': differing_tokens,
        'base_text_truncated': base_truncated,
        'source_counterfactual': source_counterfactual,
        'plot_file': str(plot_file),
        'data_file': str(data_file)
    }
    all_metadata.append(pair_metadata)

# Save all metadata to JSON
metadata_file = OUTPUT_DIR / "causal_mediation_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(all_metadata, f, indent=2)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Analyzed {len(prompt_pairs)} pairs")
print(f"Generated {len(prompt_pairs)} heatmaps")
print(f"Plots saved to: {OUTPUT_DIR}")
print(f"Metadata saved to: {metadata_file}")
print("="*80)
