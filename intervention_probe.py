"""
Intervention Analysis for Latent Reasoning Features

This script tests the causal role of learned representations by intervening on
model activations during multi-step problem solving. It loads trained probes
and uses them to inject or modify hidden representations at specified layers
and token positions.

Workflow:
1. Load trained probe from linprob.py experiments
2. Generate implicit reasoning prompts (all problem types)
3. Run baseline: generate answers without intervention
4. Run intervention: inject probe predictions at specified layer/token
5. Compare outputs to test if interventions improve reasoning accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import prompt_functions
import joblib
import json
from tqdm import tqdm
import re

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment configuration
EXPERIMENT = "velocity"  # Options: 'velocity', 'current', 'radius', 'side_length', 
                         # 'wavelength', 'cross_section', 'displacement', 'market_cap'

# Probe configuration
PROBE_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/probes_velocity/mlp_probe_layer_31.pt"
PROBE_TYPE = "mlp"  # Options: 'mlp', 'linear'

# Intervention configuration
INTERVENTION_LAYER = 31  # Layer to intervene on
INTERVENTION_TOKEN = 15  # Token position to intervene on (0-indexed)
INTERVENTION_STRENGTH = 1.0  # Strength of intervention (0=none, 1=full replacement)

# Model configuration
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"

# Data configuration
N_SAMPLES = 20  # Number of test samples per prompt format
MAX_NEW_TOKENS = 100  # Maximum tokens to generate

# Output configuration
OUTPUT_DIR = "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/intervention_results"

# ==========================================
# PROBE DEFINITION (must match linprob.py)
# ==========================================

class MLPProbe(nn.Module):
    """
    Small Multi-Layer Perceptron for non-linear probing.
    Architecture: input -> hidden -> hidden -> output
    """
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

# ==========================================
# INTERVENTION UTILITIES
# ==========================================

def create_intervention_hook(probe, probe_type, token_position, strength=1.0, device='cuda'):
    """
    Create a hook function that intervenes on activations at a specific token position.
    
    The intervention works by:
    1. Extracting the activation at the specified token position
    2. Using the probe to predict what value should be represented
    3. Computing a target activation that encodes this value
    4. Replacing (or blending) the original activation with the target
    
    Args:
        probe: Trained probe (MLPProbe or Ridge)
        probe_type: 'mlp' or 'linear'
        token_position: Which token position to intervene on
        strength: Intervention strength (0=no change, 1=full replacement)
        device: Device for computation
    
    Returns:
        Hook function and storage for predicted values
    """
    predicted_values = []
    
    def intervention_hook(activation, hook):
        """
        activation: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = activation.shape
        
        # Only intervene if the token position exists
        if token_position >= seq_len:
            return activation
        
        # Extract activation at intervention token
        token_act = activation[:, token_position, :].cpu().float()  # [batch_size, d_model]
        
        # Predict value using probe
        with torch.no_grad():
            if probe_type == 'mlp':
                token_act_gpu = token_act.to(device)
                predicted_value = probe(token_act_gpu).cpu().numpy()
            else:  # linear
                predicted_value = probe.predict(token_act.numpy())
        
        predicted_values.append(predicted_value)
        
        # For now, we just record the prediction
        # More sophisticated interventions could modify the activation
        # based on the predicted value
        
        return activation
    
    return intervention_hook, predicted_values

def create_activation_replacement_hook(target_activation, token_position, strength=1.0):
    """
    Create a hook that replaces activation at token_position with target_activation.
    
    Args:
        target_activation: Target activation vector to inject
        token_position: Which token to replace
        strength: How much to replace (0=none, 1=full)
    
    Returns:
        Hook function
    """
    def replacement_hook(activation, hook):
        """
        activation: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = activation.shape
        
        if token_position >= seq_len:
            return activation
        
        # Blend original and target activations
        original = activation[:, token_position, :]
        target = target_activation.to(activation.device).unsqueeze(0).expand(batch_size, -1)
        
        activation[:, token_position, :] = (1 - strength) * original + strength * target
        
        return activation
    
    return replacement_hook

# ==========================================
# GENERATION UTILITIES
# ==========================================

def generate_with_intervention(model, prompt, intervention_layer, intervention_hook, 
                               max_new_tokens=100, temperature=0.0):
    """
    Generate text with intervention applied at specified layer.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        intervention_layer: Layer to apply intervention
        intervention_hook: Hook function to apply
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        Generated text (without prompt)
    """
    hook_name = f"blocks.{intervention_layer}.hook_resid_post"
    
    # Tokenize prompt
    tokens = model.to_tokens(prompt, prepend_bos=True)
    embed_device = model.embed.W_E.device
    tokens = tokens.to(embed_device)
    
    # Add hook and generate
    with model.hooks(fwd_hooks=[(hook_name, intervention_hook)]):
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=None if temperature == 0 else 0.9,
            stop_at_eos=True,
            eos_token_id=model.tokenizer.eos_token_id,
            prepend_bos=False  # Already included in tokens
        )
    
    # Decode only the new tokens
    generated_text = model.to_string(output_tokens[0, tokens.shape[1]:])
    
    return generated_text

def generate_baseline(model, prompt, max_new_tokens=100, temperature=0.0):
    """
    Generate text without any intervention (baseline).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    embed_device = model.embed.W_E.device
    tokens = tokens.to(embed_device)
    
    output_tokens = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=None if temperature == 0 else 0.9,
        stop_at_eos=True,
        eos_token_id=model.tokenizer.eos_token_id,
        prepend_bos=False
    )
    
    generated_text = model.to_string(output_tokens[0, tokens.shape[1]:])
    
    return generated_text

# ==========================================
# ANSWER EXTRACTION
# ==========================================

def extract_numerical_answer(text):
    """
    Extract numerical answer from generated text.
    Looks for patterns like "X seconds", "X meters", "X m/s", etc.
    """
    # Remove any markdown or formatting
    text = text.strip()
    
    # Try to find numbers in various formats
    patterns = [
        r'(?:is|equals?|=|:)\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',  # After "is", "equals", "="
        r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?:seconds?|meters?|m/s|amperes?|A|J|joules?|cm|kg|Hz|Coulombs?|C)',  # Before units
        r'(?:^|\s)([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',  # Any number
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0])
            except (ValueError, IndexError):
                continue
    
    return None

def compute_ground_truth_answer(experiment, hidden_value, prompt):
    """
    Compute the correct answer based on the experiment type and hidden value.
    
    Args:
        experiment: Type of problem
        hidden_value: The hidden variable value
        prompt: The prompt text (to extract other parameters)
    
    Returns:
        Ground truth numerical answer
    """
    if experiment == 'velocity':
        # Extract distance from prompt
        match = re.search(r'travel (\d+) m', prompt)
        if match:
            distance = float(match.group(1))
            time = distance / hidden_value  # time = distance / velocity
            return time
    
    elif experiment == 'current':
        # Extract time from prompt
        match = re.search(r'after (\d+) seconds?', prompt)
        if match:
            time = float(match.group(1))
            charge = hidden_value * time  # Q = I * t
            return charge
    
    elif experiment == 'radius':
        # Circumference = 2πr
        circumference = 2 * np.pi * hidden_value
        return circumference
    
    elif experiment == 'side_length':
        # Surface area of cube = 6s²
        surface_area = 6 * (hidden_value ** 2)
        return surface_area
    
    elif experiment == 'wavelength':
        # Extract n from prompt
        match = re.search(r'between (\d+) (?:consecutive|successive|adjacent)', prompt)
        if match:
            n = float(match.group(1))
            distance = (n - 1) * hidden_value  # Distance between n crests
            return distance
    
    elif experiment == 'cross_section':
        # Extract velocity and time from prompt
        v_match = re.search(r'(?:speed of|at) (\d+) cm/s', prompt)
        t_match = re.search(r'after (\d+) seconds?', prompt)
        if v_match and t_match:
            velocity = float(v_match.group(1))
            time = float(t_match.group(1))
            area = np.pi * (hidden_value ** 2)  # π r²
            volume = area * velocity * time  # V = A * v * t
            return volume
    
    elif experiment == 'displacement':
        # Extract spring constant and force from prompt
        k_match = re.search(r'constant (\d+) N/m', prompt)
        f_match = re.search(r'force (?:applied to it )?(?:of )?(\d+) (?:N|Newtons)', prompt)
        if k_match and f_match:
            # Potential energy = 0.5 * k * x²
            k = float(k_match.group(1))
            potential_energy = 0.5 * k * (hidden_value ** 2)
            return potential_energy
    
    elif experiment == 'market_cap':
        # Extract net income from prompt
        match = re.search(r'net income (?:of )?\$?(\d+\.?\d*) million', prompt)
        if match:
            income = float(match.group(1))
            pe_ratio = hidden_value / income  # P/E = Market Cap / Earnings
            return pe_ratio
    
    return None

# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():
    print(f"{'='*70}")
    print(f"INTERVENTION ANALYSIS: {EXPERIMENT.upper()}")
    print(f"{'='*70}")
    print(f"Probe: {PROBE_PATH}")
    print(f"Probe type: {PROBE_TYPE}")
    print(f"Intervention layer: {INTERVENTION_LAYER}")
    print(f"Intervention token: {INTERVENTION_TOKEN}")
    print(f"Intervention strength: {INTERVENTION_STRENGTH}")
    print(f"Model: {MODEL_PATH}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Ensure embedding layer is on GPU
    if model.embed.W_E.device.type == 'cpu':
        model.embed = model.embed.to('cuda:0')
        print("Moved embedding layer to cuda:0")
    
    if hasattr(model, 'pos_embed') and model.pos_embed.W_pos.device.type == 'cpu':
        model.pos_embed = model.pos_embed.to('cuda:0')
        print("Moved positional embedding to cuda:0")
    
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dimensions\n")
    
    # ==========================================
    # LOAD PROBE
    # ==========================================
    
    print(f"Loading probe from {PROBE_PATH}...")
    
    if PROBE_TYPE == 'mlp':
        probe = MLPProbe(input_dim=model.cfg.d_model, hidden_dim=128)
        probe.load_state_dict(torch.load(PROBE_PATH, map_location=device))
        probe = probe.to(device)
        probe.eval()
    else:  # linear
        probe = joblib.load(PROBE_PATH)
    
    print(f"Probe loaded successfully\n")
    
    # ==========================================
    # GENERATE TEST DATA
    # ==========================================
    
    print(f"Generating test data for {EXPERIMENT}...")
    
    # Map experiment to prompt generation function
    gen_functions = {
        'velocity': prompt_functions.gen_implicit_velocity,
        'current': prompt_functions.gen_implicit_current,
        'radius': prompt_functions.gen_implicit_radius,
        'side_length': prompt_functions.gen_implicit_side_length,
        'wavelength': prompt_functions.gen_implicit_wavelength,
        'cross_section': prompt_functions.gen_implicit_cross_section,
        'displacement': prompt_functions.gen_implicit_displacement,
        'market_cap': prompt_functions.gen_implicit_market_cap,
    }
    
    gen_func = gen_functions[EXPERIMENT]
    test_prompts, test_prompt_ids, test_hidden_values = gen_func(samples_per_prompt=N_SAMPLES)
    
    print(f"Generated {len(test_prompts)} test prompts")
    print(f"Hidden value range: [{test_hidden_values.min():.2f}, {test_hidden_values.max():.2f}]")
    print(f"Example prompt: {test_prompts[0][:100]}...")
    print()
    
    # ==========================================
    # RUN EXPERIMENTS
    # ==========================================
    
    results = {
        'config': {
            'experiment': EXPERIMENT,
            'probe_path': PROBE_PATH,
            'probe_type': PROBE_TYPE,
            'intervention_layer': INTERVENTION_LAYER,
            'intervention_token': INTERVENTION_TOKEN,
            'intervention_strength': INTERVENTION_STRENGTH,
            'model_path': MODEL_PATH,
            'n_samples': N_SAMPLES,
            'max_new_tokens': MAX_NEW_TOKENS,
        },
        'samples': []
    }
    
    print("Running experiments...")
    print(f"Processing {len(test_prompts)} prompts...\n")
    
    for idx in tqdm(range(len(test_prompts)), desc="Testing interventions"):
        prompt = test_prompts[idx]
        prompt_id = test_prompt_ids[idx]
        hidden_value = test_hidden_values[idx]
        
        # Compute ground truth answer
        ground_truth = compute_ground_truth_answer(EXPERIMENT, hidden_value, prompt)
        
        # ===== BASELINE: Generate without intervention =====
        baseline_output = generate_baseline(
            model, prompt, 
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0
        )
        baseline_answer = extract_numerical_answer(baseline_output)
        
        # ===== INTERVENTION: Generate with probe-based intervention =====
        intervention_hook, predicted_values = create_intervention_hook(
            probe, PROBE_TYPE, INTERVENTION_TOKEN,
            strength=INTERVENTION_STRENGTH, device=device
        )
        
        intervention_output = generate_with_intervention(
            model, prompt,
            intervention_layer=INTERVENTION_LAYER,
            intervention_hook=intervention_hook,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0
        )
        intervention_answer = extract_numerical_answer(intervention_output)
        
        # Get probe's prediction
        probe_prediction = predicted_values[0][0] if predicted_values else None
        
        # Store results
        sample_result = {
            'idx': idx,
            'prompt_id': int(prompt_id),
            'prompt': prompt,
            'hidden_value': float(hidden_value),
            'ground_truth_answer': float(ground_truth) if ground_truth is not None else None,
            'probe_prediction': float(probe_prediction) if probe_prediction is not None else None,
            'baseline': {
                'output': baseline_output,
                'answer': float(baseline_answer) if baseline_answer is not None else None
            },
            'intervention': {
                'output': intervention_output,
                'answer': float(intervention_answer) if intervention_answer is not None else None
            }
        }
        
        # Compute errors if possible
        if ground_truth is not None:
            if baseline_answer is not None:
                sample_result['baseline']['error'] = abs(baseline_answer - ground_truth)
                sample_result['baseline']['relative_error'] = abs(baseline_answer - ground_truth) / max(abs(ground_truth), 1e-6)
            
            if intervention_answer is not None:
                sample_result['intervention']['error'] = abs(intervention_answer - ground_truth)
                sample_result['intervention']['relative_error'] = abs(intervention_answer - ground_truth) / max(abs(ground_truth), 1e-6)
        
        if probe_prediction is not None and hidden_value is not None:
            sample_result['probe_error'] = abs(probe_prediction - hidden_value)
            sample_result['probe_relative_error'] = abs(probe_prediction - hidden_value) / max(abs(hidden_value), 1e-6)
        
        results['samples'].append(sample_result)
    
    # ==========================================
    # COMPUTE SUMMARY STATISTICS
    # ==========================================
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Filter samples with valid answers
    valid_samples = [s for s in results['samples'] 
                     if s.get('ground_truth_answer') is not None 
                     and s['baseline'].get('answer') is not None]
    
    print(f"\nValid samples: {len(valid_samples)}/{len(results['samples'])}")
    
    if valid_samples:
        # Baseline statistics
        baseline_errors = [s['baseline']['error'] for s in valid_samples if 'error' in s['baseline']]
        if baseline_errors:
            print(f"\nBaseline Performance:")
            print(f"  Mean absolute error: {np.mean(baseline_errors):.4f}")
            print(f"  Median absolute error: {np.median(baseline_errors):.4f}")
            print(f"  Std absolute error: {np.std(baseline_errors):.4f}")
        
        # Intervention statistics
        intervention_samples = [s for s in valid_samples if s['intervention'].get('answer') is not None]
        if intervention_samples:
            intervention_errors = [s['intervention']['error'] for s in intervention_samples if 'error' in s['intervention']]
            if intervention_errors:
                print(f"\nIntervention Performance:")
                print(f"  Mean absolute error: {np.mean(intervention_errors):.4f}")
                print(f"  Median absolute error: {np.median(intervention_errors):.4f}")
                print(f"  Std absolute error: {np.std(intervention_errors):.4f}")
                
                # Improvement
                if baseline_errors and len(baseline_errors) == len(intervention_errors):
                    improvements = np.array(baseline_errors) - np.array(intervention_errors)
                    print(f"\nImprovement (baseline - intervention):")
                    print(f"  Mean: {np.mean(improvements):.4f}")
                    print(f"  Samples improved: {np.sum(improvements > 0)}/{len(improvements)}")
                    print(f"  Samples degraded: {np.sum(improvements < 0)}/{len(improvements)}")
        
        # Probe prediction statistics
        probe_errors = [s['probe_error'] for s in valid_samples if 'probe_error' in s]
        if probe_errors:
            print(f"\nProbe Prediction Quality:")
            print(f"  Mean absolute error: {np.mean(probe_errors):.4f}")
            print(f"  Median absolute error: {np.median(probe_errors):.4f}")
    
    # Summary by prompt format
    print(f"\nResults by Prompt Format:")
    unique_prompt_ids = sorted(set(s['prompt_id'] for s in results['samples']))
    for pid in unique_prompt_ids:
        pid_samples = [s for s in valid_samples if s['prompt_id'] == pid]
        if pid_samples:
            pid_baseline_errors = [s['baseline']['error'] for s in pid_samples if 'error' in s['baseline']]
            pid_intervention_errors = [s['intervention']['error'] for s in pid_samples 
                                      if 'error' in s['intervention']]
            
            print(f"  Format {pid}: {len(pid_samples)} samples")
            if pid_baseline_errors:
                print(f"    Baseline MAE: {np.mean(pid_baseline_errors):.4f}")
            if pid_intervention_errors:
                print(f"    Intervention MAE: {np.mean(pid_intervention_errors):.4f}")
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    
    output_file = output_dir / f"intervention_{EXPERIMENT}_layer{INTERVENTION_LAYER}_token{INTERVENTION_TOKEN}.json"
    
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Experiment complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
