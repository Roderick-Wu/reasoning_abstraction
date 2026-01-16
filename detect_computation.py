"""
Intermediate Computation Detection via Probing

Instead of asking "Does the model know the final answer at the start?", this script asks:
"Can we detect WHEN the model computes intermediate variables?"

Hypothesis: When solving KE=0.5·m·v², the model breaks this down into steps.
It might compute v² first, then 0.5·m, then multiply. We can train probes on 
simple arithmetic and detect when they fire during physics reasoning.

Workflow:
1. Train Arithmetic Probes: Train probes on simple arithmetic like "10 times 10 = 100"
2. Generate Physics CoT: Feed model step-by-step reasoning chains
3. Detection: Run arithmetic probes across all tokens in the reasoning chain
4. Analysis: See if multiplication probe spikes when model computes v²

This bypasses non-linearity by probing for the linear computational steps
that constitute the reasoning algorithm.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
from pathlib import Path
import prompt_functions

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment Configuration
EXPERIMENT = "velocity_computation"  # Detecting v² computation in KE problems
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
PLOTS_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_computation_{EXPERIMENT}")
PLOTS_DIR.mkdir(exist_ok=True)

# Data Configuration
N_TRAIN_ARITHMETIC = 2000  # Number of arithmetic training samples
N_TEST_COT = 50   # Number of physics CoT samples to test

# Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis Configuration
LAYERS_TO_TEST = [7, 15, 23, 31, 47, 55, 63]  # Which layers to analyze
REGULARIZATION_ALPHA = 1.0  # Ridge regression regularization parameter

print(f"="*60)
print(f"INTERMEDIATE COMPUTATION DETECTION: {EXPERIMENT.upper()}")
print(f"="*60)
print(f"Model: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Plots directory: {PLOTS_DIR}")
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

# Ensure embedding layer is on a GPU device
if model.embed.W_E.device.type == 'cpu':
    model.embed = model.embed.to('cuda:0')
    print("Moved embedding layer to cuda:0")

if hasattr(model, 'pos_embed') and model.pos_embed.W_pos.device.type == 'cpu':
    model.pos_embed = model.pos_embed.to('cuda:0')
    print("Moved positional embedding to cuda:0")

print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dimensions")
print(f"Embedding device: {model.embed.W_E.device}\n")

# ==========================================
# DATASET GENERATION
# ==========================================

def gen_arithmetic_multiplication(n_samples=1000):
    """Generate simple multiplication problems with answers: 'X times Y equals Z'"""
    prompts = []
    values = []  # The result (Z)
    
    templates = [
        "{x} times {y} equals {z}",
        "{x} multiplied by {y} is {z}",
        "The product of {x} and {y} is {z}",
        "{x} × {y} = {z}",
        "Multiply {x} by {y}: {z}",
    ]
    
    for _ in range(n_samples):
        x = np.random.randint(2, 20)
        y = np.random.randint(2, 20)
        z = x * y
        template = np.random.choice(templates)
        
        prompt = template.format(x=x, y=y, z=z)
        prompts.append(prompt)
        values.append(float(z))
        
    return prompts, np.array(values)


def gen_arithmetic_squaring(n_samples=1000):
    """Generate squaring problems with answers: 'X squared is Y'"""
    prompts = []
    values = []  # The result (X²)
    
    templates = [
        "{x} squared equals {y}",
        "The square of {x} is {y}",
        "{x}² = {y}",
        "{x} to the power of 2 is {y}",
        "What is {x} squared? {y}",
    ]
    
    for _ in range(n_samples):
        x = np.random.randint(2, 20)
        y = x ** 2
        template = np.random.choice(templates)
        
        prompt = template.format(x=x, y=y)
        prompts.append(prompt)
        values.append(float(y))
        
    return prompts, np.array(values)


print("Generating datasets...")

# Generate arithmetic training data
mult_prompts, mult_values = gen_arithmetic_multiplication(N_TRAIN_ARITHMETIC // 2)
square_prompts, square_values = gen_arithmetic_squaring(N_TRAIN_ARITHMETIC // 2)

print(f"Generated {len(mult_prompts)} multiplication prompts")
print(f"  Example: '{mult_prompts[0]}' -> {mult_values[0]}")
print(f"Generated {len(square_prompts)} squaring prompts")
print(f"  Example: '{square_prompts[0]}' -> {square_values[0]}")

# Generate physics test data using the same function as linprob.py
physics_prompts, physics_prompt_ids, physics_true_velocities = prompt_functions.gen_implicit_velocity(
    samples_per_prompt=N_TEST_COT // 5  # Distribute across 5 prompt formats
)
# Calculate v² from the true velocity values
physics_v_squared = physics_true_velocities ** 2

print(f"\nGenerated {len(physics_prompts)} implicit physics prompts")
print(f"  Example: '{physics_prompts[0]}'")
print(f"  Example v value: {physics_true_velocities[0]:.1f} m/s")
print(f"  Example v² value: {physics_v_squared[0]:.1f}")
print()

# ==========================================
# LINEAR PROBE DEFINITION
# ==========================================

def train_linear_probe(train_acts, train_labels, alpha=1.0):
    """
    Train a linear probe using Ridge regression (closed-form solution).
    Much faster than gradient descent for linear models.
    
    Args:
        train_acts: numpy array of shape [n_samples, input_dim]
        train_labels: numpy array of shape [n_samples]
        alpha: regularization strength
        
    Returns:
        Trained Ridge regression model
    """
    print(f"    Training Ridge regression with alpha={alpha}...")
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(train_acts, train_labels)
    
    # Evaluate training performance
    train_predictions = probe.predict(train_acts)
    train_r2 = r2_score(train_labels, train_predictions)
    train_mae = mean_absolute_error(train_labels, train_predictions)
    print(f"    Training R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
    
    return probe


# ==========================================
# ACTIVATION EXTRACTION
# ==========================================

def extract_all_token_activations(prompts, model, layer, batch_size=16):
    """Extract activations from all tokens at specified layer"""
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_activations = []
    
    embed_device = model.embed.W_E.device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        batch_tokens_list = []
        batch_token_lengths = []
        max_len = 0
        
        for prompt in batch_prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            batch_tokens_list.append(tokens)
            batch_token_lengths.append(tokens.shape[1])
            max_len = max(max_len, tokens.shape[1])
        
        # Pad tokens
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
                names_filter=lambda name: name == hook_name
            )
        
        batch_acts = cache[hook_name]
        
        for j in range(len(batch_prompts)):
            n_tokens = batch_token_lengths[j]
            prompt_acts = batch_acts[j, :n_tokens].cpu().float()
            all_activations.append(prompt_acts)
    
    return torch.cat(all_activations, dim=0).numpy()


def extract_last_token_activations(prompts, model, layer, batch_size=16):
    """
    Extract activations from ONLY the last token of each prompt.
    This is where the answer information is most strongly represented,
    since the last token has seen the entire prompt including the answer.
    
    Args:
        prompts: List of text prompts
        model: HookedTransformer model
        layer: Layer index to extract from
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape [n_prompts, d_model]
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_activations = []
    
    embed_device = model.embed.W_E.device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        batch_tokens_list = []
        batch_token_lengths = []
        max_len = 0
        
        for prompt in batch_prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            batch_tokens_list.append(tokens)
            batch_token_lengths.append(tokens.shape[1])
            max_len = max(max_len, tokens.shape[1])
        
        # Pad tokens
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
                names_filter=lambda name: name == hook_name
            )
        
        batch_acts = cache[hook_name]
        
        # Extract ONLY the last non-padding token for each prompt
        for j in range(len(batch_prompts)):
            last_token_idx = batch_token_lengths[j] - 1
            last_token_act = batch_acts[j, last_token_idx].cpu().float()
            all_activations.append(last_token_act)
    
    return torch.stack(all_activations, dim=0).numpy()


def extract_token_sequence_activations(prompt, model, layer):
    """
    Extract activations for each token in a single prompt.
    Returns: activations [seq_len, d_model], tokens [seq_len]
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    embed_device = model.embed.W_E.device
    
    tokens = model.to_tokens(prompt, prepend_bos=True).to(embed_device)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
    
    activations = cache[hook_name][0].cpu().float().numpy()  # [seq_len, d_model]
    token_ids = tokens[0].cpu().numpy()  # [seq_len]
    
    return activations, token_ids


# ==========================================
# TRAIN PROBES ON ARITHMETIC
# ==========================================

print("Training linear probes on arithmetic operations...")

# Combine multiplication and squaring data
all_arithmetic_prompts = mult_prompts + square_prompts
all_arithmetic_values = np.concatenate([mult_values, square_values])

# Get input dimension from first layer using last token only
print("Extracting last token activations (where answer is visible)...")
first_layer_acts = extract_last_token_activations(all_arithmetic_prompts[:1], model, LAYERS_TO_TEST[0])
input_dim = first_layer_acts.shape[1]
print(f"Input dimension: {input_dim}")
print(f"Training on last token of each prompt only\\n")

# Store probes per layer
arithmetic_probes = {}

for layer in LAYERS_TO_TEST:
    print(f"\nLayer {layer:2d}:")
    
    # Extract activations from LAST TOKEN only (where answer has been seen)
    train_acts = extract_last_token_activations(all_arithmetic_prompts, model, layer)
    
    # Labels: one per prompt (no repetition needed)
    train_labels = all_arithmetic_values
    
    print(f"  Training on {len(train_acts)} samples (last token only)...")
    
    # Train linear probe (closed-form solution)
    probe = train_linear_probe(
        train_acts, train_labels,
        alpha=REGULARIZATION_ALPHA
    )
    arithmetic_probes[layer] = probe
    print(f"  Training complete")

# ==========================================
# TEST ON PHYSICS COT
# ==========================================

print("\n" + "="*60)
print("DETECTING COMPUTATION IN PHYSICS REASONING")
print("="*60)

# Store results for each physics problem
physics_results = []

for idx, physics_prompt in enumerate(physics_prompts[:10]):  # Analyze first 10 in detail
    print(f"\nPhysics Problem {idx + 1}:")
    print(f"Expected v² value: {physics_v_squared[idx]}")
    print(f"Prompt preview: {physics_prompt[:100]}...")
    
    result = {
        'prompt': physics_prompt,
        'true_v_squared': physics_v_squared[idx],
        'layers': {}
    }
    
    # For each layer, run the arithmetic probe across all tokens
    for layer in LAYERS_TO_TEST:
        probe = arithmetic_probes[layer]
        
        # Get activations for each token
        activations, token_ids = extract_token_sequence_activations(physics_prompt, model, layer)
        
        # Run probe on each token
        predictions = probe.predict(activations)
        
        # Decode tokens for visualization
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
        
        result['layers'][layer] = {
            'predictions': predictions,
            'token_ids': token_ids,
            'token_strs': token_strs,
            'activations': activations
        }
        
        # Find where prediction is closest to true v²
        errors = np.abs(predictions - physics_v_squared[idx])
        best_match_idx = np.argmin(errors)
        best_match_token = token_strs[best_match_idx]
        best_match_pred = predictions[best_match_idx]
        
        print(f"  Layer {layer}: Best match at token '{best_match_token}' (pos {best_match_idx})")
        print(f"    Predicted: {best_match_pred:.1f}, Error: {errors[best_match_idx]:.1f}")
    
    physics_results.append(result)

# ==========================================
# VISUALIZATION
# ==========================================

print("\nGenerating visualizations...")

# Plot 1: Heatmap of predictions across tokens for first few physics problems
n_problems_to_plot = min(5, len(physics_results))
fig, axes = plt.subplots(n_problems_to_plot, 1, figsize=(20, 4*n_problems_to_plot))
if n_problems_to_plot == 1:
    axes = [axes]

for prob_idx in range(n_problems_to_plot):
    ax = axes[prob_idx]
    result = physics_results[prob_idx]
    
    # Get predictions from different layers
    layer_preds = []
    for layer in LAYERS_TO_TEST:
        layer_preds.append(result['layers'][layer]['predictions'])
    
    # Create heatmap: layers x tokens
    heatmap_data = np.array(layer_preds)
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', 
                   vmin=0, vmax=result['true_v_squared']*2)
    
    ax.set_yticks(range(len(LAYERS_TO_TEST)))
    ax.set_yticklabels(LAYERS_TO_TEST)
    ax.set_ylabel('Layer', fontsize=10)
    ax.set_xlabel('Token Position', fontsize=10)
    
    # Add horizontal line for true value
    true_val = result['true_v_squared']
    ax.axhline(y=-0.5, color='blue', linewidth=3, label=f'True v²={true_val:.0f}')
    
    ax.set_title(f'Problem {prob_idx+1}: v²={true_val:.0f} | Probe Predictions Across Tokens', 
                 fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Predicted v²')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'computation_heatmaps.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'computation_heatmaps.png'}")
plt.close()

# Plot 2: Line plots showing predictions across token positions for each layer
for prob_idx in range(min(3, len(physics_results))):
    result = physics_results[prob_idx]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    for layer in LAYERS_TO_TEST:
        predictions = result['layers'][layer]['predictions']
        ax.plot(predictions, 'o-', linewidth=2, markersize=4, label=f'Layer {layer}', alpha=0.7)
    
    # True value line
    true_val = result['true_v_squared']
    ax.axhline(y=true_val, color='red', linewidth=2.5, linestyle='--', 
               label=f'True v²={true_val:.0f}', alpha=0.8)
    
    # Get token strings for x-axis
    token_strs = result['layers'][LAYERS_TO_TEST[0]]['token_strs']
    
    # Show token labels at bottom (sample every nth token to avoid crowding)
    step = max(1, len(token_strs) // 20)
    ax.set_xticks(range(0, len(token_strs), step))
    ax.set_xticklabels([token_strs[i][:10] for i in range(0, len(token_strs), step)], 
                       rotation=45, ha='right', fontsize=8)
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Predicted v²', fontsize=12)
    ax.set_title(f'Computation Detection: Problem {prob_idx+1} (v²={true_val:.0f})', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'computation_trace_problem_{prob_idx+1}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOTS_DIR / f'computation_trace_problem_{prob_idx+1}.png'}")
    plt.close()

# Plot 3: Summary - average error distance from true value
print("\nComputing summary statistics...")

avg_min_errors_by_layer = {layer: [] for layer in LAYERS_TO_TEST}

for result in physics_results:
    true_val = result['true_v_squared']
    for layer in LAYERS_TO_TEST:
        predictions = result['layers'][layer]['predictions']
        min_error = np.min(np.abs(predictions - true_val))
        avg_min_errors_by_layer[layer].append(min_error)

# Convert to averages
avg_errors = [np.mean(avg_min_errors_by_layer[layer]) for layer in LAYERS_TO_TEST]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(LAYERS_TO_TEST)), avg_errors, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(LAYERS_TO_TEST)))
ax.set_xticklabels(LAYERS_TO_TEST)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Average Minimum Error', fontsize=12)
ax.set_title('Computation Detection Performance: Average Best Match Error per Layer', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'summary_detection_performance.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'summary_detection_performance.png'}")
plt.close()

# ==========================================
# ANALYSIS SUMMARY
# ==========================================

print(f"\n{'='*60}")
print("ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Experiment: {EXPERIMENT}")
print(f"Training: {len(all_arithmetic_prompts)} arithmetic prompts")
print(f"Testing: {len(physics_prompts)} physics prompts (implicit reasoning)")
print(f"\nBest Layer Performance (by minimum error):")

for layer in LAYERS_TO_TEST:
    avg_err = np.mean(avg_min_errors_by_layer[layer])
    print(f"  Layer {layer}: Avg min error = {avg_err:.2f}")

best_layer = LAYERS_TO_TEST[np.argmin(avg_errors)]
print(f"\nBest layer: {best_layer} with avg error {min(avg_errors):.2f}")

print(f"\n{'='*60}")
print(f"All visualizations saved to: {PLOTS_DIR}")
print(f"{'='*60}")
