"""
MLP Probing Analysis with Chain-of-Thought Generation

This script trains MLP probes on explicit prompts and tests them
on implicit reasoning prompts INCLUDING the generated chain-of-thought tokens.

Workflow:
1. Decoder Training (Explicit): Generate prompts like "The car is moving at X m/s" 
   and train a small MLP to map residual stream activations to value X.
2. CoT Generation (Implicit): Generate chain-of-thought responses for implicit reasoning problems
3. The Reveal: Run the probe on BOTH the prompt tokens and the generated CoT tokens
   to see where the hidden value appears during reasoning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
from pathlib import Path
import prompt_functions
import json

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment Configuration
EXPERIMENT = "velocity"  # Options: "velocity", "current"
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
PLOTS_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_linprob_cot_{EXPERIMENT}")
PLOTS_DIR.mkdir(exist_ok=True)

# Data Configuration
N_TRAIN_EXPLICIT = 1000  # Number of explicit training samples
N_TEST_PER_FORMAT = 1   # Number of test samples per implicit prompt format

# Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis Configuration
LAYERS_TO_TEST = [7, 15, 23, 31, 47, 55, 63]  # Which layers to analyze
MLP_HIDDEN_SIZE = 128  # Hidden layer size for MLP probe
MLP_LEARNING_RATE = 0.0001  # Learning rate for MLP training
MLP_EPOCHS = 50  # Number of training epochs
MLP_BATCH_SIZE = 256  # Batch size for MLP training

# CoT Generation Configuration
MAX_NEW_TOKENS = 512  # Maximum tokens to generate (reduced to avoid OOM)
TEMPERATURE = 0.7
TOP_P = 0.9

print(f"="*60)
print(f"MLP PROBING WITH CHAIN-OF-THOUGHT: {EXPERIMENT.upper()}")
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
    move_to_device=False  # Don't move model - it's already distributed
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

# ==========================================
# GENERATE DATASETS
# ==========================================

print("Generating datasets...")

# Select appropriate prompt generation functions based on experiment
if EXPERIMENT == "velocity":
    gen_explicit = prompt_functions.gen_explicit_velocity
    gen_implicit = lambda: prompt_functions.gen_implicit_velocity(samples_per_prompt=N_TEST_PER_FORMAT)
elif EXPERIMENT == "current":
    gen_explicit = prompt_functions.gen_explicit_current
    gen_implicit = lambda: prompt_functions.gen_implicit_current(samples_per_prompt=N_TEST_PER_FORMAT)
else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")

# Generate explicit training data
train_prompts, train_values = gen_explicit(n_samples=N_TRAIN_EXPLICIT)
print(f"Generated {len(train_prompts)} explicit training prompts")
print(f"  Example: '{train_prompts[0]}' -> {train_values[0]}")
print(f"  Value range: [{train_values.min():.1f}, {train_values.max():.1f}]")

# Generate implicit test data (one per format)
test_prompts, test_prompt_ids, test_true_values = gen_implicit()

print(f"Generated {len(test_prompts)} implicit test prompts ({N_TEST_PER_FORMAT} per format)")
for i, (prompt, val) in enumerate(zip(test_prompts, test_true_values)):
    print(f"  Format {test_prompt_ids[i]}: '{prompt[:80]}...' -> {val:.1f}")
print()

# ==========================================
# MLP PROBE DEFINITION
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

def train_mlp_probe(train_acts, train_labels, input_dim, hidden_dim=128, 
                    learning_rate=0.001, epochs=50, batch_size=256, device='cuda'):
    """
    Train an MLP probe on the given activations and labels.
    
    Args:
        train_acts: numpy array of shape [n_samples, input_dim]
        train_labels: numpy array of shape [n_samples]
        input_dim: dimension of input features
        hidden_dim: size of hidden layers
        learning_rate: learning rate for optimizer
        epochs: number of training epochs
        batch_size: batch size for training
        device: device to train on
        
    Returns:
        Trained MLPProbe model
    """
    # Convert to tensors
    X = torch.FloatTensor(train_acts)
    y = torch.FloatTensor(train_labels)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    probe = MLPProbe(input_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    
    # Training loop
    probe.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = probe(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    probe.eval()
    return probe

# ==========================================
# ACTIVATION EXTRACTION
# ==========================================

def find_value_position_in_prompt(prompt, value, model):
    """
    Find the token position where the target value appears in the prompt.
    Returns the first token position where the value string appears.
    
    Args:
        prompt: Text prompt
        value: Numeric value to find
        model: HookedTransformer model
    
    Returns:
        Token index where value appears (or 0 if not found)
    """
    # Convert value to string with different formats to match
    value_strs = [f"{value:.1f}", f"{value:.2f}", f"{value}", str(int(value)) if value == int(value) else str(value)]
    
    # Try to find value string in prompt
    value_pos_in_text = -1
    matched_str = None
    for val_str in value_strs:
        if val_str in prompt:
            value_pos_in_text = prompt.index(val_str)
            matched_str = val_str
            break
    
    if value_pos_in_text == -1:
        return 0
    
    # Tokenize and find the token position
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    # Get text for each token
    for i in range(tokens.shape[1]):
        # Check if we've reached the value position
        # Reconstruct text up to this point
        reconstructed = model.to_string(tokens[0, :i+1])
        if matched_str in reconstructed and matched_str not in model.to_string(tokens[0, :i]):
            return i
    
    return 0

def extract_post_value_activations(prompts, values, model, layer, batch_size=16):
    """
    Extract activations only from tokens at or after the position where the value appears.
    This respects causal attention masking - only these tokens can have information about the value.
    
    Args:
        prompts: List of text prompts
        values: Array of target values corresponding to each prompt
        model: HookedTransformer model
        layer: Layer index to extract from
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (activations numpy array, labels numpy array)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_activations = []
    all_labels = []
    
    # Get the device of the embedding layer
    embed_device = model.embed.W_E.device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_values = values[i:i + batch_size]
        
        # Tokenize each prompt individually
        batch_tokens_list = []
        batch_token_lengths = []
        batch_value_positions = []
        max_len = 0
        
        for j, prompt in enumerate(batch_prompts):
            tokens = model.to_tokens(prompt, prepend_bos=True)
            batch_tokens_list.append(tokens)
            batch_token_lengths.append(tokens.shape[1])
            
            # Find where the value appears
            value_pos = find_value_position_in_prompt(prompt, batch_values[j], model)
            batch_value_positions.append(value_pos)
            
            max_len = max(max_len, tokens.shape[1])
        
        # Pad tokens to same length for batching
        padded_tokens = []
        for tokens in batch_tokens_list:
            if tokens.shape[1] < max_len:
                padding = torch.zeros((1, max_len - tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=1)
            padded_tokens.append(tokens)
        
        # Stack into batch and move to embedding device
        batch_tokens = torch.cat(padded_tokens, dim=0).to(embed_device)
        
        with torch.no_grad():
            # Run with cache using pre-tokenized input
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda name: name == hook_name
            )
        
        # Extract activations only from tokens at/after value position
        batch_acts = cache[hook_name]  # [batch_size, seq_len, d_model]
        
        for j in range(len(batch_prompts)):
            n_tokens = batch_token_lengths[j]
            value_pos = batch_value_positions[j]
            
            # Only take activations from value_pos onwards
            if value_pos < n_tokens:
                prompt_acts = batch_acts[j, value_pos:n_tokens].cpu().float()  # [n_valid_tokens, d_model]
                n_valid_tokens = n_tokens - value_pos
                
                all_activations.append(prompt_acts)
                # Repeat the label for all valid tokens
                all_labels.extend([batch_values[j]] * n_valid_tokens)
    
    # Concatenate all activations
    activations = torch.cat(all_activations, dim=0).numpy()
    labels = np.array(all_labels)
    
    return activations, labels

def extract_activations_with_generation(prompt, model, layer, max_new_tokens=512):
    """
    Generate chain-of-thought completion and extract activations from all tokens
    (both prompt and generated).
    
    Args:
        prompt: Text prompt
        model: HookedTransformer model
        layer: Layer index to extract from
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Tuple of (activations, token_strings, prompt_length, generated_text)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    embed_device = model.embed.W_E.device
    
    # Tokenize prompt
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(embed_device)
    prompt_length = prompt_tokens.shape[1]
    
    # Generate with caching
    all_tokens = prompt_tokens.clone()
    all_activations = []
    
    print(f"    Generating CoT (max {max_new_tokens} tokens)...")
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                all_tokens,
                names_filter=lambda name: name == hook_name
            )
        
        # Get activations for the last token
        last_token_act = cache[hook_name][0, -1].cpu().float()  # [d_model]
        all_activations.append(last_token_act)
        
        # Clear cache to save memory
        del cache
        if step % 10 == 0:  # Periodically clear CUDA cache
            torch.cuda.empty_cache()
        
        # Get logits and sample next token
        logits = model(all_tokens)[0, -1]  # [vocab_size]
        
        # Sample with temperature
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        
        # Top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum_probs > TOP_P
        mask[0] = False  # Keep at least one token
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        
        next_token_idx = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
        next_token = torch.tensor([[next_token_idx]], dtype=all_tokens.dtype, device=embed_device)
        
        # Check for EOS
        if next_token_idx == model.tokenizer.eos_token_id:
            break
        
        # Append token
        all_tokens = torch.cat([all_tokens, next_token], dim=1)
    
    # Convert tokens to strings
    token_strings = [model.to_string(all_tokens[0, i]) for i in range(all_tokens.shape[1])]
    generated_text = model.to_string(all_tokens[0, prompt_length:])
    
    # Stack activations
    activations = torch.stack(all_activations, dim=0).numpy()  # [seq_len, d_model]
    
    print(f"    Generated {len(generated_text)} characters in {len(all_activations)} tokens")
    
    return activations, token_strings, prompt_length, generated_text

# ==========================================
# TRAIN PROBES
# ==========================================

print("Training MLP probes on explicit prompts...")

# Get input dimension from first layer
first_layer_acts, first_layer_labels = extract_post_value_activations(
    train_prompts[:1], train_values[:1], model, LAYERS_TO_TEST[0]
)
input_dim = first_layer_acts.shape[1]
print(f"Input dimension: {input_dim}\n")

# Store trained probes
trained_probes = {}

for i, layer in enumerate(LAYERS_TO_TEST):
    print(f"Layer {layer:2d}:")
    
    # Extract activations only from tokens at/after value position
    train_acts, train_labels = extract_post_value_activations(
        train_prompts, train_values, model, layer
    )
    
    print(f"  Training on {len(train_acts)} tokens (only post-value tokens)...")
    
    # Train MLP probe
    probe = train_mlp_probe(
        train_acts, train_labels, 
        input_dim=input_dim,
        hidden_dim=MLP_HIDDEN_SIZE,
        learning_rate=MLP_LEARNING_RATE,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        device=device
    )
    trained_probes[layer] = probe
    
    print(f"  Training complete\n")

# ==========================================
# GENERATE COT AND ANALYZE
# ==========================================

print("="*60)
print("GENERATING CHAIN-OF-THOUGHT AND ANALYZING")
print("="*60)

# Store results for each test sample
cot_results = []

for sample_idx in range(len(test_prompts)):
    prompt = test_prompts[sample_idx]
    prompt_id = test_prompt_ids[sample_idx]
    true_value = test_true_values[sample_idx]
    
    print(f"\nSample {sample_idx + 1}/{len(test_prompts)} (Format {prompt_id}, True value: {true_value:.1f})")
    print(f"Prompt: {prompt[:100]}...")
    
    sample_result = {
        'prompt': prompt,
        'prompt_id': int(prompt_id),
        'true_value': float(true_value),
        'layers': {}
    }
    
    # OPTIMIZATION: Generate CoT once, then extract activations from all layers on the complete sequence
    print(f"  Generating CoT...")
    _, token_strings, prompt_length, generated_text = extract_activations_with_generation(
        prompt, model, LAYERS_TO_TEST[0], max_new_tokens=MAX_NEW_TOKENS
    )
    
    # Now we have the complete token sequence - extract activations from all layers at once
    # Reconstruct the full token sequence
    full_tokens = model.to_tokens(prompt, prepend_bos=True)
    embed_device = model.embed.W_E.device
    
    # Regenerate the same tokens (deterministic with same seed or we could cache)
    # Actually, we need to just get the token IDs from token_strings
    # Better approach: extract activations for all layers in one pass
    print(f"  Generated {len(generated_text)} characters in {len(token_strings)} tokens")
    print(f"  Extracting activations from all layers...")
    
    # Tokenize the full sequence (prompt + generated)
    full_text = prompt + generated_text
    full_tokens = model.to_tokens(full_text, prepend_bos=True).to(embed_device)
    
    # Extract activations from all layers in a single forward pass
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in LAYERS_TO_TEST]
    with torch.no_grad():
        _, cache = model.run_with_cache(
            full_tokens,
            names_filter=lambda name: name in hook_names
        )
    
    # For each layer, run probe on extracted activations
    for layer in LAYERS_TO_TEST:
        print(f"  Layer {layer}:")
        probe = trained_probes[layer]
        
        # Get activations for this layer
        hook_name = f"blocks.{layer}.hook_resid_post"
        activations = cache[hook_name][0].cpu().float().numpy()  # [seq_len, d_model]
        
        # Run probe on all tokens
        with torch.no_grad():
            acts_tensor = torch.FloatTensor(activations).to(device)
            predictions = probe(acts_tensor).cpu().numpy()
        
        # Compute statistics
        prompt_predictions = predictions[:prompt_length]
        generated_predictions = predictions[prompt_length:]
        
        prompt_mean = np.mean(prompt_predictions)
        prompt_std = np.std(prompt_predictions)
        generated_mean = np.mean(generated_predictions) if len(generated_predictions) > 0 else 0.0
        generated_std = np.std(generated_predictions) if len(generated_predictions) > 0 else 0.0
        
        # Find tokens closest to true value
        errors = np.abs(predictions - true_value)
        closest_indices = np.argsort(errors)[:5]
        
        print(f"    Prompt tokens: mean={prompt_mean:.2f} ± {prompt_std:.2f}")
        print(f"    Generated tokens: mean={generated_mean:.2f} ± {generated_std:.2f}")
        print(f"    Closest predictions to {true_value:.1f}:")
        for idx in closest_indices:
            token_type = "PROMPT" if idx < prompt_length else "GENERATED"
            token_str = model.to_string(full_tokens[0, idx])
            print(f"      Token {idx} ({token_type}): '{token_str[:20]}' -> {predictions[idx]:.2f}")
        
        sample_result['layers'][layer] = {
            'predictions': predictions.tolist(),
            'token_strings': [model.to_string(full_tokens[0, i]) for i in range(full_tokens.shape[1])],
            'prompt_length': prompt_length,
            'generated_text': generated_text,
            'prompt_mean': float(prompt_mean),
            'prompt_std': float(prompt_std),
            'generated_mean': float(generated_mean),
            'generated_std': float(generated_std),
        }
    
    cot_results.append(sample_result)

# Save results to JSON
results_file = PLOTS_DIR / 'cot_analysis_results.json'
with open(results_file, 'w') as f:
    json.dump(cot_results, f, indent=2)
print(f"\nSaved detailed results to {results_file}")

# ==========================================
# VISUALIZATION
# ==========================================

print("\nGenerating visualizations...")

# For each sample, create a plot showing predictions across tokens
for sample_idx, sample_result in enumerate(cot_results):
    prompt_id = sample_result['prompt_id']
    true_value = sample_result['true_value']
    
    fig, axes = plt.subplots(len(LAYERS_TO_TEST), 1, figsize=(16, 3*len(LAYERS_TO_TEST)))
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]
    
    for layer_idx, layer in enumerate(LAYERS_TO_TEST):
        ax = axes[layer_idx]
        layer_data = sample_result['layers'][layer]
        
        predictions = layer_data['predictions']
        prompt_length = layer_data['prompt_length']
        
        # Plot predictions
        token_positions = np.arange(len(predictions))
        ax.plot(token_positions, predictions, 'b-', alpha=0.6, linewidth=1)
        
        # Highlight prompt vs generated
        ax.axvspan(0, prompt_length, alpha=0.1, color='green', label='Prompt')
        ax.axvspan(prompt_length, len(predictions), alpha=0.1, color='orange', label='Generated')
        
        # True value line
        ax.axhline(y=true_value, color='r', linestyle='--', linewidth=2, label=f'True Value ({true_value:.1f})')
        
        ax.set_xlabel('Token Position', fontsize=11)
        ax.set_ylabel('Probe Prediction', fontsize=11)
        ax.set_title(f'Layer {layer}: Predictions Across Tokens', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sample {sample_idx + 1} (Format {prompt_id}): Probe Predictions During CoT Generation', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'cot_predictions_sample_{sample_idx}_format_{prompt_id}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOTS_DIR / f'cot_predictions_sample_{sample_idx}_format_{prompt_id}.png'}")
    plt.close()

# Summary plot: mean predictions for prompt vs generated tokens across layers
fig, axes = plt.subplots(1, len(test_prompts), figsize=(6*len(test_prompts), 6))
if len(test_prompts) == 1:
    axes = [axes]

for sample_idx, sample_result in enumerate(cot_results):
    ax = axes[sample_idx]
    prompt_id = sample_result['prompt_id']
    true_value = sample_result['true_value']
    
    prompt_means = []
    generated_means = []
    
    for layer in LAYERS_TO_TEST:
        layer_data = sample_result['layers'][layer]
        prompt_means.append(layer_data['prompt_mean'])
        generated_means.append(layer_data['generated_mean'])
    
    x = np.arange(len(LAYERS_TO_TEST))
    width = 0.35
    
    ax.bar(x - width/2, prompt_means, width, label='Prompt Tokens', alpha=0.7)
    ax.bar(x + width/2, generated_means, width, label='Generated Tokens', alpha=0.7)
    ax.axhline(y=true_value, color='r', linestyle='--', linewidth=2, label=f'True Value ({true_value:.1f})')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Prediction', fontsize=12)
    ax.set_title(f'Format {prompt_id} (True: {true_value:.1f})', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS_TO_TEST)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Mean Probe Predictions: Prompt vs Generated Tokens', fontsize=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'mean_predictions_summary.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'mean_predictions_summary.png'}")
plt.close()

print(f"\n{'='*60}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"All visualizations saved to: {PLOTS_DIR}")
print(f"Detailed results saved to: {results_file}")
print(f"{'='*60}")
