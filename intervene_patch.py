"""
Activation Patching for Latent Reasoning Features

This script performs activation patching experiments to test whether specific
layer/token positions encode latent reasoning variables (like velocity).

Methodology:
1. Generate pairs of prompts (source and base) with different hidden velocities
2. Extract activations from source prompt at specific (layer, token) positions
3. Patch those activations into base prompt
4. Generate completions and observe if the intervention changes the output
5. If patching source activations into base causes base to output source's velocity,
   this suggests that position encodes the velocity representation

Inspired by: "Unveiling LLMs: The Evolution of Latent Representations" (Bronzini et al., COLM 2024)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
OUTPUT_DIR = Path("/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/intervention_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Traces configuration
TRACES_DIR = Path("/home/wuroderi/scratch/reasoning_traces/Qwen2.5-32B/velocity")
TRACES_METADATA_FILE = TRACES_DIR / "traces_metadata.json"

# Intervention Configuration
N_PROMPT_PAIRS = 30  # Number of pairs to test
#LAYERS_TO_TEST = [23, 31, 47]  # Middle layers
LAYERS_TO_TEST = [0, 5, 10, 15]
QUESTION_START_KEYWORDS = ["What", "calculate", "Determine", "Find", "How"]  # Words indicating question start

# Generation Configuration
MAX_NEW_TOKENS = 256  # Reduced to prevent OOM
TEMPERATURE = 0.4 

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"="*80)
print(f"ACTIVATION PATCHING EXPERIMENT")
print(f"="*80)
print(f"Model: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Prompt pairs: {N_PROMPT_PAIRS}")
print(f"Layers: {LAYERS_TO_TEST}")
print(f"Question start keywords: {QUESTION_START_KEYWORDS}")
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

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} dimensions")
print()

# ==========================================
# LOAD TRACES AND CREATE PROMPT PAIRS
# ==========================================

print("Loading traces from file...")
with open(TRACES_METADATA_FILE, 'r') as f:
    traces = json.load(f)

print(f"Loaded {len(traces)} traces")

def extract_prompt_from_trace(trace_text):
    """
    Extract just the prompt from a trace that includes prompt + generation.
    The prompt ends with 'Answer (step-by-step): '
    """
    marker = "Answer (step-by-step): "
    if marker in trace_text:
        # Find the end of the marker and return everything up to and including it
        end_pos = trace_text.find(marker) + len(marker)
        return trace_text[:end_pos]
    else:
        # Fallback: try to find just the prompt before any reasoning
        # If the model started generating, it might have "Let" or "First" or numbers
        return trace_text  # Return as-is if we can't find the marker

print("Creating prompt pairs...")

# Extract all information from JSON and group prompts by format_id
traces_by_format = defaultdict(list)
for trace in traces:
    traces_by_format[trace['format_id']].append(trace)

# Ensure each format has the same number of traces
min_traces_per_format = min(len(traces) for traces in traces_by_format.values())

# Define n_pairs based on the number of traces and the pairing logic
n_pairs = min_traces_per_format // 2

# Create prompt pairs by pairing traces within the same format_id
prompt_pairs = []
for format_id, format_traces in traces_by_format.items():
    for i in range(n_pairs):
        source_trace = format_traces[i]
        base_trace = format_traces[i + n_pairs]

        # Extract just the prompts (without model generations)
        source_prompt = extract_prompt_from_trace(source_trace['generated_text'])
        base_prompt = extract_prompt_from_trace(base_trace['generated_text'])

        pair = {
            'source_prompt': source_prompt,
            'source_velocity': source_trace['v'],
            'source_trace_id': source_trace['id'],
            'base_prompt': base_prompt,
            'base_velocity': base_trace['v'],
            'base_trace_id': base_trace['id'],
            'format_id': format_id
        }
        prompt_pairs.append(pair)

print(f"Created {len(prompt_pairs)} prompt pairs grouped by format_id.")

# Test question detection on example prompt
print("Testing question detection...")
example_tokens = tokenizer(prompt_pairs[0]['base_prompt'], return_tensors="pt", add_special_tokens=True)
example_token_ids = example_tokens['input_ids'][0]
print(f"Example prompt has {len(example_token_ids)} tokens")
print(f"Example prompt: {prompt_pairs[0]['base_prompt'][:100]}...")
print()

# Ensure paired prompts have the same format_id by taking it from the JSON
for i, pair in enumerate(prompt_pairs):
    source_format_id = traces[i]['format_id']
    base_format_id = traces[i + n_pairs]['format_id']

    # Assign format_id from the JSON
    pair['format_id'] = source_format_id

    # Ensure source and base prompts have the same format_id
    pair['source_format_id'] = source_format_id
    pair['base_format_id'] = base_format_id

print("Assigned format IDs to paired prompts from JSON.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def find_question_start_token(tokenizer, prompt, keywords):
    """
    Find the token position where the question starts by looking for specific keywords.
    
    Args:
        tokenizer: Tokenizer
        prompt: Text prompt
        keywords: List of words that indicate question start
    
    Returns:
        Token position where question starts, or -1 #if not found
    """
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    token_ids = tokens['input_ids'][0]
    
    # Decode each token and check for keywords
    for i in range(len(token_ids)):
        token_str = tokenizer.decode([token_ids[i].item()])
        
        # Check if any keyword appears in this token (handle spaces)
        for keyword in keywords:
            # Check both with and without spaces
            if keyword in token_str or keyword.lower() in token_str.lower():
                return i
            # Also check if keyword appears in combination with previous token
            if i > 0:
                prev_token_str = tokenizer.decode([token_ids[i-1].item()])
                combined = prev_token_str + token_str
                if keyword in combined or keyword.lower() in combined.lower():
                    return i-1  # Return the start of the keyword
    
    return -1  # Not found

def get_token_string(tokenizer, prompt, token_pos):
    """
    Get the actual token string at a specific position in the tokenized prompt.
    
    Args:
        tokenizer: Tokenizer
        prompt: Text prompt
        token_pos: Token position index
    
    Returns:
        String representation of the token at that position
    """
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    token_ids = tokens['input_ids'][0]
    
    if token_pos >= len(token_ids):
        return f"[OUT_OF_BOUNDS:{token_pos}/{len(token_ids)}]"
    
    token_id = token_ids[token_pos].item()
    token_str = tokenizer.decode([token_id])
    return token_str

def find_numerical_value_tokens(tokenizer, prompt, value, value_name="value"):
    """
    Find token positions where a numerical value appears in the prompt.
    
    Args:
        tokenizer: Tokenizer
        prompt: Text prompt
        value: Numerical value to find (e.g., 23876.5)
        value_name: Name of the value for debugging
    
    Returns:
        List of token positions where this value appears
    """
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    token_ids = tokens['input_ids'][0]
    
    # Convert value to string and try different formats
    value_str = str(value)
    value_formats = []
    
    # For large values, try scientific notation with 4 significant figures
    # This matches the format in the traces (e.g., 23876.5 -> 2.388e+04)
    if abs(float(value)) >= 1000:
        value_formats.append(f"{float(value):.3e}")  # 4 sig figs: 2.388e+04
        value_formats.append(f"{float(value):.4e}")  # 5 sig figs: 2.3877e+04
        value_formats.append(f"{float(value):.2e}")  # 3 sig figs: 2.39e+04
    
    # Add standard formats
    value_formats.append(value_str)
    if '.' in value_str:
        value_formats.append(value_str.rstrip('0').rstrip('.'))
    else:
        value_formats.extend([str(int(value)), f"{float(value):.1f}", f"{float(value):.2f}"])
    
    positions = []
    
    # Decode full text to find where the value appears
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    print(f"  Searching for {value_name}={value}")
    print(f"    Formats to try: {value_formats[:3]}")
    
    # Find value in text - try each format
    found_format = None
    for value_fmt in value_formats:
        # Check if format appears in text (case insensitive for 'e' vs 'E')
        if value_fmt.lower() in full_text.lower():
            found_format = value_fmt
            print(f"    Found '{found_format}' in full text")
            
            # Now find which token span contains this exact value
            # We need to find the tightest span that contains only the value
            best_span = None
            best_span_len = 999
            
            for i in range(len(token_ids)):
                # Try spans from 1 to 10 tokens
                for span_len in range(1, 11):
                    if i + span_len > len(token_ids):
                        break
                    
                    span_token_ids = [token_ids[i+j].item() for j in range(span_len)]
                    span_str = tokenizer.decode(span_token_ids).strip()
                    
                    # Check if this span contains our value
                    if value_fmt.lower() in span_str.lower():
                        # Check if it's the exact value or has minimal extra characters
                        # We want the smallest span that contains the value
                        if span_len < best_span_len:
                            # Verify this span actually represents just the number
                            # by checking it doesn't have lots of extra text
                            if len(span_str) <= len(value_fmt) + 10:  # Allow some tokenization artifacts
                                best_span = list(range(i, i + span_len))
                                best_span_len = span_len
            
            if best_span:
                positions = best_span
                # Get actual token strings for debug output
                token_strings = [tokenizer.decode([token_ids[p].item()]) for p in positions]
                print(f"    Best span at positions {positions}: {token_strings}")
                break
    
    if not positions:
        print(f"    WARNING: Could not find {value_name}={value} in prompt")
        print(f"    Full text sample: ...{full_text[50:250]}...")
    
    # Sort positions
    positions = sorted(positions)
    
    return positions

def extract_activation(model, tokenizer, prompt, layer, token_pos):
    """
    Extract activation from a specific layer and token position.
    
    Returns:
        Activation tensor of shape [d_model]
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs['input_ids'].to(model.device)
    
    # Storage for activation
    activation_storage = {}
    
    # Hook to capture activation
    def hook_fn(module, input, output):
        # output is a tuple (hidden_states,) for decoder layers
        hidden_states = output[0]  # [batch, seq_len, hidden_size]
        activation_storage['activation'] = hidden_states[0, token_pos].detach().cpu()
    
    # Register hook on the specific layer
    # For Qwen2, layers are in model.model.layers[layer]
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    # Remove hook
    handle.remove()
    
    # Clean up GPU memory
    del input_ids, inputs
    torch.cuda.empty_cache()
    
    return activation_storage['activation']

def generate_with_intervention(model, tokenizer, prompt, patches):
    """
    Generate text from prompt with interventions at specific layer/token positions.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Text prompt
        patches: List of dicts with keys 'layer', 'token_pos', 'activation'
                 Each activation is a tensor of shape [hidden_size]
    
    Returns:
        Generated text (full completion including prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs['input_ids'].to(model.device)
    
    # Group patches by layer for efficiency
    patches_by_layer = {}
    for patch in patches:
        layer = patch['layer']
        if layer not in patches_by_layer:
            patches_by_layer[layer] = []
        patches_by_layer[layer].append(patch)
    
    # Create hook functions for each layer that has patches
    def create_patch_hook(layer_patches):
        def patch_hook(module, input, output):
            # Handle different output structures (tuple vs single tensor)
            if isinstance(output, tuple):
                hidden_states = output[0].clone()  # [batch, seq_len, hidden_size]
            else:
                hidden_states = output.clone()  # [batch, seq_len, hidden_size]
            
            # Apply all patches for this layer
            for patch in layer_patches:
                token_pos = patch['token_pos']
                activation = patch['activation']
                if token_pos < hidden_states.shape[1]:
                    hidden_states[0, token_pos] = activation.to(hidden_states.device)
            
            # Return in the same structure as the original output
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        return patch_hook
    
    # Register hooks for all layers that need patching
    handles = []
    for layer, layer_patches in patches_by_layer.items():
        handle = model.model.layers[layer].register_forward_hook(create_patch_hook(layer_patches))
        handles.append(handle)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True if TEMPERATURE > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    # Move output to CPU immediately to free GPU memory
    output_cpu = output.cpu()
    del output, input_ids, inputs
    torch.cuda.empty_cache()
    
    # Decode output
    generated_text = tokenizer.decode(output_cpu[0], skip_special_tokens=False)
    
    return generated_text

def generate_baseline(model, tokenizer, prompt):
    """Generate text without intervention (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs['input_ids'].to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True if TEMPERATURE > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Move output to CPU immediately to free GPU memory
    output_cpu = output.cpu()
    del output, input_ids, inputs
    torch.cuda.empty_cache()
    
    return tokenizer.decode(output_cpu[0], skip_special_tokens=False)

# ==========================================
# RUN EXPERIMENTS
# ==========================================

print("Running activation patching experiments...")
print("="*80)

results = []

for pair_idx, pair in enumerate(prompt_pairs):
    print(f"\n{'='*80}")
    print(f"PAIR {pair_idx + 1}/{N_PROMPT_PAIRS}")
    print(f"{'='*80}")
    print(f"Source velocity: {pair['source_velocity']:.1f} m/s")
    print(f"Base velocity:   {pair['base_velocity']:.1f} m/s")
    print()
    
    # Generate baseline (no intervention) for base prompt
    print("Generating baseline (base prompt, no intervention)...")
    baseline_output = generate_baseline(model, tokenizer, pair['base_prompt'])
    # Find where prompt ends in the tokenized version
    baseline_prompt_length = len(tokenizer(pair['base_prompt'], add_special_tokens=True)['input_ids'])
    baseline_tokens = tokenizer(baseline_output, add_special_tokens=False)['input_ids']
    generated_part_baseline = tokenizer.decode(baseline_tokens[baseline_prompt_length:], skip_special_tokens=True)
    print(f"Baseline output: {generated_part_baseline[:100]}...")
    print()
    
    # Store baseline
    pair_results = {
        'pair_idx': pair_idx,
        'source_trace_id': pair['source_trace_id'],
        'base_trace_id': pair['base_trace_id'],
        'source_velocity': pair['source_velocity'],
        'base_velocity': pair['base_velocity'],
        'source_prompt': pair['source_prompt'],
        'base_prompt': pair['base_prompt'],
        'baseline_output': baseline_output,
        'interventions': []
    }
    
    # Get numerical values from trace metadata
    source_ke = traces[pair_idx]['ke']  # kinetic energy
    source_mass = traces[pair_idx]['m']  # mass
    base_ke = traces[pair_idx + n_pairs]['ke']
    base_mass = traces[pair_idx + n_pairs]['m']
    
    print(f"Source KE={source_ke}, mass={source_mass}")
    print(f"Base KE={base_ke}, mass={base_mass}")
    
    # Find token positions of numerical values in the prompts
    print("\nFinding token positions of numerical values...")
    
    # Find KE and mass positions in source prompt
    source_ke_positions = find_numerical_value_tokens(tokenizer, pair['source_prompt'], source_ke, "source_KE")
    source_mass_positions = find_numerical_value_tokens(tokenizer, pair['source_prompt'], source_mass, "source_mass")
    
    # Find KE and mass positions in base prompt
    base_ke_positions = find_numerical_value_tokens(tokenizer, pair['base_prompt'], base_ke, "base_KE")
    base_mass_positions = find_numerical_value_tokens(tokenizer, pair['base_prompt'], base_mass, "base_mass")
    
    print(f"Source prompt - KE at tokens {source_ke_positions}, mass at tokens {source_mass_positions}")
    print(f"Base prompt - KE at tokens {base_ke_positions}, mass at tokens {base_mass_positions}")
    
    # Combine all numerical value positions
    source_value_positions = sorted(source_ke_positions + source_mass_positions)
    base_value_positions = sorted(base_ke_positions + base_mass_positions)
    
    if not source_value_positions or not base_value_positions:
        print("WARNING: Could not find numerical values in one or both prompts. Skipping this pair.")
        continue
    
    print(f"\nWill intervene on {len(base_value_positions)} numerical value tokens: {base_value_positions}")
    print()
    
    # Test interventions per layer (all numerical value tokens at once)
    for layer in LAYERS_TO_TEST:
        print(f"\nLayer {layer}: Extracting activations for numerical value tokens {source_value_positions}...")
        
        # Extract activations from source prompt for numerical value positions
        patches = []
        for src_pos, base_pos in zip(source_value_positions, base_value_positions):
            source_activation = extract_activation(
                model,
                tokenizer,
                pair['source_prompt'], 
                layer, 
                src_pos
            )
            patches.append({
                'layer': layer,
                'token_pos': base_pos,  # Patch into base prompt at corresponding position
                'activation': source_activation
            })
        
        print(f"Layer {layer}: Extracted {len(patches)} activations from source numerical values. Now patching into base and generating...")
        
        # Get token strings for the patched positions (base and source)
        base_token_strings = [get_token_string(tokenizer, pair['base_prompt'], pos) for pos in base_value_positions]
        source_token_strings = [get_token_string(tokenizer, pair['source_prompt'], pos) for pos in source_value_positions]
        print(f"  Patching base tokens: {base_token_strings}")
        print(f"  Using source tokens: {source_token_strings}")

        # Patch ALL tokens at once and generate
        intervened_output = generate_with_intervention(
            model,
            tokenizer,
            pair['base_prompt'],
            patches
        )

        # Extract just the generated part (after prompt)
        intervened_tokens = tokenizer(intervened_output, add_special_tokens=False)['input_ids']
        generated_part = tokenizer.decode(intervened_tokens[baseline_prompt_length:], skip_special_tokens=True)
        print(f"Layer {layer} Generated: {generated_part[:80]}...")

        # Store result
        intervention_result = {
            'layer': layer,
            'num_tokens_patched': len(patches),
            'patched_positions': base_value_positions,
            'patched_tokens': base_token_strings,
            'source_patched_positions': source_value_positions,
            'source_patched_tokens': source_token_strings,
            'patch_description': f"KE tokens: {base_ke_positions}, mass tokens: {base_mass_positions}",
            'intervened_output': intervened_output,
            'generated_part': generated_part
        }
        pair_results['interventions'].append(intervention_result)
        print()
    
    results.append(pair_results)

# ==========================================
# SAVE AND SUMMARIZE RESULTS
# ==========================================

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)

# Save detailed results
output_file = OUTPUT_DIR / "intervention_results.txt"
with open(output_file, 'w') as f:
    f.write("ACTIVATION PATCHING EXPERIMENT RESULTS\n")
    f.write("="*80 + "\n\n")
    
    for pair_result in results:
        f.write(f"\nPAIR {pair_result['pair_idx'] + 1}\n")
        f.write("-"*80 + "\n")
        f.write(f"Source trace: {pair_result['source_trace_id']} (velocity: {pair_result['source_velocity']:.1f} m/s)\n")
        f.write(f"Base trace:   {pair_result['base_trace_id']} (velocity: {pair_result['base_velocity']:.1f} m/s)\n")
        f.write(f"\nSource prompt:\n{pair_result['source_prompt']}\n")
        f.write(f"\nBase prompt:\n{pair_result['base_prompt']}\n")
        f.write(f"\nBaseline output (no intervention):\n{pair_result['baseline_output']}\n")
        f.write("\n" + "-"*80 + "\n")
        f.write("INTERVENTIONS:\n")
        f.write("-"*80 + "\n")
        
        for intervention in pair_result['interventions']:
            f.write(f"\nLayer {intervention['layer']}: Patched {intervention['num_tokens_patched']} numerical value tokens\n")
            f.write(f"Base positions: {intervention['patched_positions']} ({intervention['patch_description']})\n")
            f.write(f"Base tokens: {intervention.get('patched_tokens', [])}\n")
            f.write(f"Source positions: {intervention.get('source_patched_positions', [])}\n")
            f.write(f"Source tokens: {intervention.get('source_patched_tokens', [])}\n")
            f.write(f"{intervention['intervened_output']}\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")

print(f"\nDetailed results saved to: {output_file}")

# Print summary
print("\nSUMMARY:")
print(f"Tested {N_PROMPT_PAIRS} prompt pairs")
print(f"Interventions: {len(LAYERS_TO_TEST)} layers × tokens from question start to end")
total_interventions = sum(len(result['interventions']) for result in results)
print(f"Total interventions: {total_interventions}")
print(f"\nLook for cases where patching source activations causes base to generate source velocity!")
print("="*80)