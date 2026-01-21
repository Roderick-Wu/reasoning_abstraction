"""
Latent Space Intervention for Causal Testing

This script performs interventions at the latent space (activation) level during 
chain-of-thought (CoT) generation. It modifies model activations at specified 
layers and token positions to test causal relationships between latent representations
and model outputs.

Workflow:
1. Load trained probe that predicts hidden variable from activations
2. Generate CoT for source example, extract activations at specified position
3. Generate CoT for target example, intervene on activations at specified layer/token
4. Replace target activations with source activations (or probe-guided modifications)
5. Continue generation and compare outputs to baseline

Intervention Strategies:
- 'direct': Directly replace target activations with source activations
- 'residual': Add the difference (source - baseline) to target activations
- 'probe_guided': Use probe to compute target activation for desired value
"""

import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import prompt_functions
import json
import re
from typing import List, Tuple, Optional, Dict, Callable
import joblib

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment configuration
EXPERIMENT = "velocity"  # Options: 'velocity', 'current', 'radius', etc.

# Model configuration
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"

# Probe configuration (optional - for probe-guided interventions)
PROBE_PATH = None  # Set to probe path for probe-guided interventions
PROBE_TYPE = "mlp"  # Options: 'mlp', 'linear'

# Generation configuration
MAX_TOKENS_BEFORE_INTERVENTION = 300  # Tokens to generate before intervention
MAX_TOKENS_AFTER_INTERVENTION = 200   # Tokens to continue after intervention
TEMPERATURE = 0.7
TOP_P = 0.9

# Intervention configuration
INTERVENTION_LAYERS = [23, 31, 47]  # Layers to test interventions on
INTERVENTION_STRATEGY = "direct"  # Options: 'direct', 'residual', 'probe_guided'
INTERVENTION_STRENGTH = 1.0  # Scaling factor for intervention (0=none, 1=full)

# Token position configuration
INTERVENTION_TOKEN_MODE = "value"  # Options: 'value' (at value output), 'before_value', 'last'
INTERVENTION_TOKEN_OFFSET = 0  # Offset from the reference position (e.g., -1 for one token before)

# Data configuration
N_SOURCE_SAMPLES = 5   # Number of source examples
N_TARGET_SAMPLES = 5   # Number of target examples per source
MAX_TOTAL_EXPERIMENTS = 25  # Maximum total experiments to run

# Output configuration
OUTPUT_DIR = Path("/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/intervention_latent_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("LATENT SPACE INTERVENTION FOR CAUSAL TESTING")
print("="*70)
print(f"Experiment: {EXPERIMENT}")
print(f"Model: {MODEL_PATH}")
print(f"Intervention Strategy: {INTERVENTION_STRATEGY}")
print(f"Intervention Layers: {INTERVENTION_LAYERS}")
print(f"Token Mode: {INTERVENTION_TOKEN_MODE}")
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
# PROBE DEFINITION (if using probe-guided)
# ==========================================

class MLPProbe(nn.Module):
    """MLP probe for non-linear probing (matches linprob.py)."""
    def __init__(self, input_dim, hidden_dim=128):
        super(MLPProbe, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)

# Load probe if specified
probe = None
if PROBE_PATH is not None and INTERVENTION_STRATEGY == "probe_guided":
    print(f"Loading probe from {PROBE_PATH}...")
    if PROBE_TYPE == "mlp":
        probe = torch.load(PROBE_PATH)
        probe.eval()
        print(f"  Loaded MLP probe")
    elif PROBE_TYPE == "linear":
        probe = joblib.load(PROBE_PATH)
        print(f"  Loaded linear probe")
    print()

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def extract_number_from_text(text: str, search_terms: List[str] = None) -> Optional[float]:
    """Extract numerical value from text, optionally near specific terms."""
    if search_terms is None:
        search_terms = ['velocity', 'current', 'speed', '=']
    
    patterns = []
    for term in search_terms:
        patterns.append(rf'{term}.*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)')
    
    patterns.extend([
        r'=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*m/s',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*(?:amperes?|A\b)',
    ])
    
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
    patterns = [
        r'(?:answer|result|final|therefore|takes).*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*(?:seconds?|meters?|coulombs?|m\b|s\b|C\b)',
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            try:
                return float(matches[-1].group(1))
            except:
                continue
    
    # Fallback: get last number
    numbers = re.findall(r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass
    
    return None

def find_value_token_position(tokens: torch.Tensor, value: float, 
                              tokenizer, prompt_length: int) -> Optional[int]:
    """
    Find the token position where a value is generated.
    Returns the position of the last token of the value.
    """
    # Generate string representation
    full_text = tokenizer.decode(tokens[0].cpu().tolist())
    generated_text = tokenizer.decode(tokens[0, prompt_length:].cpu().tolist())
    
    # Try to find the value in the generated text
    extracted = extract_number_from_text(generated_text)
    
    if extracted is None or abs(extracted - value) > 0.1:
        return None
    
    # Find where in the token sequence this appears
    # This is approximate - we find the last token before the value string ends
    value_str = str(int(value)) if value == int(value) else str(value)
    
    for i in range(prompt_length, tokens.shape[1]):
        partial_text = tokenizer.decode(tokens[0, prompt_length:i+1].cpu().tolist())
        if value_str in partial_text:
            return i
    
    return None

def generate_and_cache_activations(model, prompt: str, layer: int, 
                                   max_tokens: int = 300) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Generate tokens and cache activations at specified layer.
    
    Returns:
        Tuple of (tokens, activations, generated_text)
        activations shape: [seq_len, d_model]
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    embed_device = model.embed.W_E.device
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(embed_device)
    all_tokens = prompt_tokens.clone()
    all_activations = []
    
    for step in range(max_tokens):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                all_tokens,
                names_filter=lambda name: name == hook_name
            )
        
        # Cache last token activation
        last_act = cache[hook_name][0, -1].cpu().float()
        all_activations.append(last_act)
        
        del cache
        if step % 10 == 0:
            torch.cuda.empty_cache()
        
        # Generate next token
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
    
    activations = torch.stack(all_activations, dim=0)
    generated_text = model.to_string(all_tokens[0])
    
    return all_tokens, activations, generated_text

def create_intervention_hook(source_activation: torch.Tensor, 
                            intervention_token: int,
                            intervention_strength: float = 1.0,
                            strategy: str = "direct",
                            baseline_activation: Optional[torch.Tensor] = None) -> Callable:
    """
    Create a hook function for activation intervention.
    
    Args:
        source_activation: Activation to inject [d_model]
        intervention_token: Token position to intervene on
        intervention_strength: Strength of intervention (0-1)
        strategy: 'direct', 'residual', or 'probe_guided'
        baseline_activation: Original activation for residual strategy [d_model]
    
    Returns:
        Hook function
    """
    def intervention_hook(activation, hook):
        """
        activation: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = activation.shape
        
        # Only intervene if token position exists
        if intervention_token >= seq_len:
            return activation
        
        # Get original activation
        original = activation[:, intervention_token, :].clone()
        
        # Apply intervention based on strategy
        if strategy == "direct":
            # Direct replacement
            target = source_activation.to(activation.device).unsqueeze(0)
            
        elif strategy == "residual":
            # Add residual: target = original + (source - baseline)
            if baseline_activation is None:
                print("Warning: baseline_activation required for residual strategy")
                return activation
            
            residual = source_activation - baseline_activation
            target = original + residual.to(activation.device).unsqueeze(0)
            
        else:  # probe_guided would need additional implementation
            return activation
        
        # Blend original and target based on strength
        activation[:, intervention_token, :] = (
            (1 - intervention_strength) * original + 
            intervention_strength * target
        )
        
        return activation
    
    return intervention_hook

def generate_with_intervention(model, prompt: str, layer: int, 
                               intervention_hook: Callable,
                               max_tokens: int = 200) -> Tuple[str, torch.Tensor]:
    """
    Generate text with intervention applied at specified layer.
    
    Returns:
        Tuple of (generated_text, final_tokens)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    embed_device = model.embed.W_E.device
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(embed_device)
    
    # Generate with intervention
    with model.hooks(fwd_hooks=[(hook_name, intervention_hook)]):
        output_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stop_at_eos=True,
            eos_token_id=model.tokenizer.eos_token_id,
            prepend_bos=False
        )
    
    generated_text = model.to_string(output_tokens[0])
    
    return generated_text, output_tokens

# ==========================================
# INTERVENTION EXPERIMENT
# ==========================================

def run_intervention_experiment(source_prompt: str, source_value: float,
                                target_prompt: str, target_value: float,
                                intervention_layer: int,
                                expected_answer_with_source: float,
                                expected_answer_with_target: float) -> Dict:
    """
    Run a single latent space intervention experiment.
    
    Args:
        source_prompt: Source example prompt
        source_value: Hidden value from source
        target_prompt: Target example prompt
        target_value: True hidden value in target
        intervention_layer: Layer to intervene on
        expected_answer_with_source: Expected answer if using source value
        expected_answer_with_target: Expected answer if using target value
    
    Returns:
        Dictionary with results
    """
    result = {
        'source_value': source_value,
        'target_value': target_value,
        'intervention_layer': intervention_layer,
        'expected_answer_with_source': expected_answer_with_source,
        'expected_answer_with_target': expected_answer_with_target,
    }
    
    # STEP 1: Generate source and extract activation
    print(f"    Generating source CoT...")
    source_tokens, source_acts, source_text = generate_and_cache_activations(
        model, source_prompt, intervention_layer, MAX_TOKENS_BEFORE_INTERVENTION
    )
    
    result['source_text'] = source_text
    prompt_length_source = len(model.to_tokens(source_prompt, prepend_bos=True)[0])
    
    # Find where source value appears
    source_value_pos = find_value_token_position(source_tokens, source_value, 
                                                 model.tokenizer, prompt_length_source)
    
    if source_value_pos is None:
        print(f"    WARNING: Could not find source value in generation")
        result['success'] = False
        return result
    
    print(f"    Found source value at position {source_value_pos}")
    
    # Extract source activation
    source_activation = source_acts[source_value_pos - prompt_length_source]
    result['source_value_position'] = source_value_pos
    
    # STEP 2: Generate baseline target (no intervention)
    print(f"    Generating baseline target...")
    baseline_tokens, baseline_acts, baseline_text = generate_and_cache_activations(
        model, target_prompt, intervention_layer,
        MAX_TOKENS_BEFORE_INTERVENTION + MAX_TOKENS_AFTER_INTERVENTION
    )
    
    result['baseline_text'] = baseline_text
    result['baseline_final_answer'] = extract_final_answer(baseline_text)
    
    prompt_length_target = len(model.to_tokens(target_prompt, prepend_bos=True)[0])
    target_value_pos = find_value_token_position(baseline_tokens, target_value,
                                                 model.tokenizer, prompt_length_target)
    
    # STEP 3: Determine intervention position
    if INTERVENTION_TOKEN_MODE == "value" and target_value_pos is not None:
        intervention_pos = target_value_pos + INTERVENTION_TOKEN_OFFSET
    elif INTERVENTION_TOKEN_MODE == "before_value" and target_value_pos is not None:
        intervention_pos = max(prompt_length_target, target_value_pos - 5) + INTERVENTION_TOKEN_OFFSET
    elif INTERVENTION_TOKEN_MODE == "last":
        intervention_pos = prompt_length_target + 50  # Arbitrary position in generation
    else:
        # Default to middle of generation
        intervention_pos = prompt_length_target + 20
    
    result['intervention_position'] = intervention_pos
    print(f"    Intervention position: {intervention_pos}")
    
    # Get baseline activation at intervention position (for residual strategy)
    baseline_activation = None
    if INTERVENTION_STRATEGY == "residual" and intervention_pos < len(baseline_acts) + prompt_length_target:
        baseline_activation = baseline_acts[intervention_pos - prompt_length_target]
    
    # STEP 4: Create intervention hook and generate
    print(f"    Generating with intervention...")
    intervention_hook = create_intervention_hook(
        source_activation,
        intervention_pos,
        INTERVENTION_STRENGTH,
        INTERVENTION_STRATEGY,
        baseline_activation
    )
    
    intervention_text, intervention_tokens = generate_with_intervention(
        model, target_prompt, intervention_layer, intervention_hook,
        MAX_TOKENS_BEFORE_INTERVENTION + MAX_TOKENS_AFTER_INTERVENTION
    )
    
    result['intervention_text'] = intervention_text
    result['intervention_final_answer'] = extract_final_answer(intervention_text)
    result['success'] = True
    
    # STEP 5: Analyze results
    if result['baseline_final_answer'] is not None and result['intervention_final_answer'] is not None:
        result['answer_changed'] = abs(result['baseline_final_answer'] - 
                                      result['intervention_final_answer']) > 0.5
        
        baseline_error = abs(result['baseline_final_answer'] - expected_answer_with_target)
        intervention_error_to_source = abs(result['intervention_final_answer'] - 
                                          expected_answer_with_source)
        
        result['intervention_improved_toward_source'] = (
            intervention_error_to_source < baseline_error
        )
        
        result['baseline_error'] = baseline_error
        result['intervention_error_to_source'] = intervention_error_to_source
    
    return result

# ==========================================
# GENERATE DATA AND RUN EXPERIMENTS
# ==========================================

print("Generating test data...")

# Select appropriate prompt generation function
if EXPERIMENT == "velocity":
    gen_implicit = lambda: prompt_functions.gen_implicit_velocity(samples_per_prompt=3)
elif EXPERIMENT == "current":
    gen_implicit = lambda: prompt_functions.gen_implicit_current(samples_per_prompt=3)
else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")

# Generate examples
prompts, prompt_ids, true_values = gen_implicit()

def extract_problem_params(prompt: str, experiment: str) -> Dict:
    """Extract problem parameters from prompt text."""
    if experiment == "velocity":
        match = re.search(r'travel\s+(\d+)\s*m', prompt)
        if match:
            return {'distance': float(match.group(1))}
    elif experiment == "current":
        match = re.search(r'after\s+(\d+)\s*seconds', prompt)
        if match:
            return {'time': float(match.group(1))}
    return {}

print(f"Generated {len(prompts)} test prompts")
print()

# Run experiments
all_results = []
experiment_count = 0

for layer in INTERVENTION_LAYERS:
    print(f"\nTesting Layer {layer}")
    print("="*70)
    
    for i in range(min(N_SOURCE_SAMPLES, len(prompts))):
        if experiment_count >= MAX_TOTAL_EXPERIMENTS:
            break
        
        source_prompt = prompts[i]
        source_value = true_values[i]
        source_params = extract_problem_params(source_prompt, EXPERIMENT)
        
        for j in range(min(N_TARGET_SAMPLES, len(prompts))):
            if i == j or experiment_count >= MAX_TOTAL_EXPERIMENTS:
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
            
            print(f"\n  Experiment {experiment_count + 1}: Source={source_value:.1f}, Target={target_value:.1f}")
            
            result = run_intervention_experiment(
                source_prompt, source_value,
                target_prompt, target_value,
                layer,
                expected_with_source, expected_with_target
            )
            
            all_results.append(result)
            experiment_count += 1
            
            # Save intermediate results
            if len(all_results) % 3 == 0:
                output_file = OUTPUT_DIR / f"intervention_latent_{EXPERIMENT}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
    
    if experiment_count >= MAX_TOTAL_EXPERIMENTS:
        break

# ==========================================
# SAVE FINAL RESULTS AND SUMMARY
# ==========================================

output_file = OUTPUT_DIR / f"intervention_latent_{EXPERIMENT}_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print(f"Total experiments: {len(all_results)}")
print(f"Results saved to: {output_file}")

# Compute summary statistics
successful = [r for r in all_results if r.get('success', False)]
print(f"\nSuccessful experiments: {len(successful)}/{len(all_results)}")

if successful:
    changed = [r for r in successful if r.get('answer_changed', False)]
    print(f"Interventions that changed answer: {len(changed)}/{len(successful)}")
    
    improved = [r for r in successful if r.get('intervention_improved_toward_source', False)]
    print(f"Answers moved toward source: {len(improved)}/{len(successful)}")
    
    # Per-layer breakdown
    print("\nPer-layer results:")
    for layer in INTERVENTION_LAYERS:
        layer_results = [r for r in successful if r.get('intervention_layer') == layer]
        if layer_results:
            layer_improved = [r for r in layer_results 
                            if r.get('intervention_improved_toward_source', False)]
            print(f"  Layer {layer}: {len(layer_improved)}/{len(layer_results)} improved toward source")

print()
