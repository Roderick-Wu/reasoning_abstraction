"""
Token-Level Intervention for Causal Testing

This script performs interventions at the token level during chain-of-thought (CoT) generation.
It swaps computed hidden variable values (e.g., velocity) with values from a source example
to test whether the model's reasoning causally depends on these intermediate values.

Workflow:
1. Generate CoT response from source example (with known correct hidden value)
2. Generate CoT response from target example until hidden value is computed
3. Swap the hidden value tokens from source to target
4. Continue generation and evaluate final answer
5. Compare swapped vs. baseline to measure causal dependence

Key Challenge: Multi-token numbers require careful token alignment
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import prompt_functions
import json
import re
from typing import List, Tuple, Optional, Dict

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment configuration
EXPERIMENT = "velocity"  # Options: 'velocity', 'current', 'radius', etc.

# Model configuration
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"

# Generation configuration
MAX_TOKENS_BEFORE_VALUE = 300  # Max tokens to generate before expecting the value
MAX_TOKENS_AFTER_VALUE = 200   # Max tokens to continue after intervention
TEMPERATURE = 0.7
TOP_P = 0.9

# Intervention configuration
N_SOURCE_SAMPLES = 10   # Number of source examples to use
N_TARGET_SAMPLES = 10   # Number of target examples to test
SWAP_STRATEGY = "exact"  # Options: 'exact' (swap all value tokens), 'first' (swap first occurrence)

# Output configuration
OUTPUT_DIR = Path("/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/intervention_token_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("TOKEN-LEVEL INTERVENTION FOR CAUSAL TESTING")
print("="*70)
print(f"Experiment: {EXPERIMENT}")
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# ==========================================
# LOAD MODEL
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
    move_to_device=False
)

if model.embed.W_E.device.type == 'cpu':
    model.embed = model.embed.to('cuda:0')
if hasattr(model, 'pos_embed') and model.pos_embed.W_pos.device.type == 'cpu':
    model.pos_embed = model.pos_embed.to('cuda:0')

print(f"Model loaded: {model.cfg.n_layers} layers\n")

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def extract_number_from_text(text: str) -> Optional[float]:
    """Extract the first numerical value from generated text."""
    # Look for numbers in various formats
    patterns = [
        r'velocity.*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'current.*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'speed.*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*m/s',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*(?:amperes?|A\b)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return None

def extract_final_answer(text: str) -> Optional[float]:
    """Extract the final numerical answer from generated text."""
    # Look for final answer patterns
    patterns = [
        r'(?:answer|result|final|therefore).*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*(?:seconds?|meters?|m\b|s\b)',
    ]
    
    # Try to get the last number that looks like an answer
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            try:
                return float(matches[-1].group(1))
            except:
                continue
    
    # Fallback: get any number
    numbers = re.findall(r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass
    
    return None

def tokenize_number(number: float, tokenizer) -> List[int]:
    """
    Tokenize a number to see how it's represented.
    Important: Numbers can be split into multiple tokens!
    """
    # Try different representations
    representations = [
        str(int(number)) if number == int(number) else str(number),
        f" {int(number)}" if number == int(number) else f" {number}",
        f" {int(number)} " if number == int(number) else f" {number} ",
    ]
    
    # Return the tokenization of the most common representation
    tokens = tokenizer(representations[1], add_special_tokens=False)['input_ids']
    return tokens

def find_value_in_tokens(token_ids: List[int], value: float, tokenizer, 
                         context_window: int = 5) -> Optional[Tuple[int, int]]:
    """
    Find where a value appears in a token sequence.
    Returns (start_idx, end_idx) if found, else None.
    
    Args:
        token_ids: List of token IDs
        value: Numerical value to find
        tokenizer: Tokenizer
        context_window: How many tokens before/after to check for the value
    
    Returns:
        Tuple of (start_idx, end_idx) where the value tokens are, or None
    """
    value_tokens = tokenize_number(value, tokenizer)
    
    # Search for the value token sequence
    for i in range(len(token_ids) - len(value_tokens) + 1):
        if token_ids[i:i+len(value_tokens)] == value_tokens:
            return (i, i + len(value_tokens))
    
    # If exact match fails, try fuzzy matching by decoding
    # (in case of different tokenization with spaces)
    value_str = str(int(value)) if value == int(value) else str(value)
    
    for i in range(len(token_ids)):
        for length in range(1, min(5, len(token_ids) - i + 1)):
            decoded = tokenizer.decode(token_ids[i:i+length]).strip()
            if value_str in decoded or decoded in value_str:
                return (i, i + length)
    
    return None

def generate_until_value(model, prompt: str, expected_value: float, 
                         max_tokens: int = 300) -> Optional[Tuple[torch.Tensor, int, int]]:
    """
    Generate tokens until the expected value appears, or max_tokens is reached.
    
    Returns:
        Tuple of (all_tokens, value_start_idx, value_end_idx) or None if value not found
    """
    embed_device = model.embed.W_E.device
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(embed_device)
    all_tokens = prompt_tokens.clone()
    prompt_length = prompt_tokens.shape[1]
    
    for step in range(max_tokens):
        # Generate next token
        with torch.no_grad():
            logits = model(all_tokens)[0, -1]
        
        # Sample with temperature
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum_probs > TOP_P
        mask[0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        
        next_token_idx = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
        next_token = torch.tensor([[next_token_idx]], dtype=all_tokens.dtype, device=embed_device)
        
        # Check for EOS
        if next_token_idx == model.tokenizer.eos_token_id:
            break
        
        all_tokens = torch.cat([all_tokens, next_token], dim=1)
        
        # Check if value has appeared in generated text
        generated_ids = all_tokens[0, prompt_length:].cpu().tolist()
        value_position = find_value_in_tokens(generated_ids, expected_value, model.tokenizer)
        
        if value_position is not None:
            start_idx = prompt_length + value_position[0]
            end_idx = prompt_length + value_position[1]
            return all_tokens, start_idx, end_idx
    
    return None

def continue_generation(model, tokens: torch.Tensor, max_tokens: int = 200) -> torch.Tensor:
    """Continue generation from current token sequence."""
    embed_device = model.embed.W_E.device
    all_tokens = tokens.clone()
    
    for step in range(max_tokens):
        with torch.no_grad():
            logits = model(all_tokens)[0, -1]
        
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum_probs > TOP_P
        mask[0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        
        next_token_idx = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
        next_token = torch.tensor([[next_token_idx]], dtype=all_tokens.dtype, device=embed_device)
        
        if next_token_idx == model.tokenizer.eos_token_id:
            break
        
        all_tokens = torch.cat([all_tokens, next_token], dim=1)
    
    return all_tokens

def swap_value_tokens(target_tokens: torch.Tensor, source_value: float, 
                      target_value_start: int, target_value_end: int,
                      tokenizer) -> torch.Tensor:
    """
    Swap the target value tokens with source value tokens.
    
    Args:
        target_tokens: Target token sequence [1, seq_len]
        source_value: The value to inject
        target_value_start: Start index of value in target
        target_value_end: End index of value in target
        tokenizer: Tokenizer
    
    Returns:
        Modified token sequence with swapped value
    """
    # Get source value tokens
    source_tokens = tokenize_number(source_value, tokenizer)
    source_tensor = torch.tensor([source_tokens], dtype=target_tokens.dtype, 
                                 device=target_tokens.device)
    
    # Replace target value tokens with source value tokens
    before = target_tokens[:, :target_value_start]
    after = target_tokens[:, target_value_end:]
    
    swapped_tokens = torch.cat([before, source_tensor, after], dim=1)
    
    return swapped_tokens

# ==========================================
# INTERVENTION EXPERIMENT
# ==========================================

def run_intervention_experiment(source_prompt: str, source_value: float,
                                target_prompt: str, target_value: float,
                                expected_answer_with_source: float,
                                expected_answer_with_target: float) -> Dict:
    """
    Run a single intervention experiment.
    
    Args:
        source_prompt: Source example prompt
        source_value: Hidden value from source (to inject)
        target_prompt: Target example prompt
        target_value: True hidden value in target
        expected_answer_with_source: Expected final answer if using source value
        expected_answer_with_target: Expected final answer if using target value
    
    Returns:
        Dictionary with results
    """
    result = {
        'source_value': source_value,
        'target_value': target_value,
        'expected_answer_with_source': expected_answer_with_source,
        'expected_answer_with_target': expected_answer_with_target,
    }
    
    # BASELINE: Generate target without intervention
    print(f"  Generating baseline (target only)...")
    baseline_result = generate_until_value(model, target_prompt, target_value, 
                                          MAX_TOKENS_BEFORE_VALUE + MAX_TOKENS_AFTER_VALUE)
    
    if baseline_result is None:
        print(f"    WARNING: Could not find target value in baseline generation")
        result['baseline_success'] = False
        return result
    
    baseline_tokens, _, _ = baseline_result
    baseline_text = model.to_string(baseline_tokens[0])
    result['baseline_text'] = baseline_text
    result['baseline_final_answer'] = extract_final_answer(baseline_text)
    result['baseline_success'] = True
    
    # INTERVENTION: Generate until value, swap, then continue
    print(f"  Generating target until value appears...")
    target_partial = generate_until_value(model, target_prompt, target_value, 
                                         MAX_TOKENS_BEFORE_VALUE)
    
    if target_partial is None:
        print(f"    WARNING: Could not find target value in generation")
        result['intervention_success'] = False
        return result
    
    target_tokens, value_start, value_end = target_partial
    
    print(f"    Found value at tokens [{value_start}:{value_end}]")
    print(f"    Swapping {target_value} → {source_value}")
    
    # Swap tokens
    swapped_tokens = swap_value_tokens(target_tokens, source_value, 
                                      value_start, value_end, model.tokenizer)
    
    print(f"  Continuing generation after swap...")
    final_tokens = continue_generation(model, swapped_tokens, MAX_TOKENS_AFTER_VALUE)
    
    intervention_text = model.to_string(final_tokens[0])
    result['intervention_text'] = intervention_text
    result['intervention_final_answer'] = extract_final_answer(intervention_text)
    result['intervention_success'] = True
    
    # Check if intervention changed the answer
    if result['baseline_final_answer'] is not None and result['intervention_final_answer'] is not None:
        result['answer_changed'] = abs(result['baseline_final_answer'] - 
                                      result['intervention_final_answer']) > 0.5
        
        # Check if answer moved toward expected value with source
        baseline_error = abs(result['baseline_final_answer'] - expected_answer_with_target)
        intervention_error = abs(result['intervention_final_answer'] - expected_answer_with_source)
        result['intervention_improved_toward_source'] = intervention_error < baseline_error
    
    return result

# ==========================================
# GENERATE DATA AND RUN EXPERIMENTS
# ==========================================

print("Generating test data...")

# Select appropriate prompt generation function
if EXPERIMENT == "velocity":
    gen_implicit = lambda: prompt_functions.gen_implicit_velocity(samples_per_prompt=5)
elif EXPERIMENT == "current":
    gen_implicit = lambda: prompt_functions.gen_implicit_current(samples_per_prompt=5)
else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")

# Generate examples
prompts, prompt_ids, true_values = gen_implicit()

# For velocity: compute expected final answers (time = distance / velocity)
# Extract additional problem parameters from prompts for answer calculation
def extract_problem_params(prompt: str, experiment: str) -> Dict:
    """Extract problem parameters from prompt text."""
    if experiment == "velocity":
        # Extract distance from "travel {d} m"
        match = re.search(r'travel\s+(\d+)\s*m', prompt)
        if match:
            distance = float(match.group(1))
            return {'distance': distance}
    elif experiment == "current":
        # Extract time from "after {t} seconds"
        match = re.search(r'after\s+(\d+)\s*seconds', prompt)
        if match:
            time = float(match.group(1))
            return {'time': time}
    return {}

print(f"Generated {len(prompts)} test prompts")
print(f"  Example: '{prompts[0][:100]}...'")
print()

# Run experiments
print(f"Running {N_SOURCE_SAMPLES * N_TARGET_SAMPLES} intervention experiments...")
print()

all_results = []

for i in range(min(N_SOURCE_SAMPLES, len(prompts))):
    source_prompt = prompts[i]
    source_value = true_values[i]
    source_params = extract_problem_params(source_prompt, EXPERIMENT)
    
    for j in range(min(N_TARGET_SAMPLES, len(prompts))):
        if i == j:
            continue
        
        target_prompt = prompts[j]
        target_value = true_values[j]
        target_params = extract_problem_params(target_prompt, EXPERIMENT)
        
        # Compute expected answers
        if EXPERIMENT == "velocity" and 'distance' in target_params:
            expected_with_source = target_params['distance'] / source_value
            expected_with_target = target_params['distance'] / target_value
        elif EXPERIMENT == "current" and 'time' in target_params:
            expected_with_source = source_value * target_params['time']
            expected_with_target = target_value * target_params['time']
        else:
            continue
        
        print(f"Experiment {len(all_results) + 1}: Source value={source_value:.1f}, Target value={target_value:.1f}")
        
        result = run_intervention_experiment(
            source_prompt, source_value,
            target_prompt, target_value,
            expected_with_source, expected_with_target
        )
        
        all_results.append(result)
        
        # Save intermediate results
        if len(all_results) % 5 == 0:
            output_file = OUTPUT_DIR / f"intervention_token_{EXPERIMENT}_results.json"
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved intermediate results to {output_file}")
        
        print()

# ==========================================
# SAVE FINAL RESULTS
# ==========================================

output_file = OUTPUT_DIR / f"intervention_token_{EXPERIMENT}_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print(f"Total experiments: {len(all_results)}")
print(f"Results saved to: {output_file}")

# Compute summary statistics
successful_interventions = [r for r in all_results if r.get('intervention_success', False)]
print(f"\nSuccessful interventions: {len(successful_interventions)}/{len(all_results)}")

if successful_interventions:
    changed_answers = [r for r in successful_interventions if r.get('answer_changed', False)]
    print(f"Interventions that changed answer: {len(changed_answers)}/{len(successful_interventions)}")
    
    improved_toward_source = [r for r in successful_interventions 
                             if r.get('intervention_improved_toward_source', False)]
    print(f"Answers moved toward source expectation: {len(improved_toward_source)}/{len(successful_interventions)}")

print()
