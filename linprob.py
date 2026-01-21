"""
MLP Probing Analysis for Latent Reasoning Features

This script trains small MLP probes on explicit prompts and tests them
on implicit reasoning prompts to detect shared abstract representations.
Uses non-linear MLPs instead of linear probes to capture non-linear hidden variables.

Workflow:
1. Decoder Training (Explicit): Generate prompts like "The car is moving at X m/s" 
   and train a small MLP to map residual stream activations to value X.
2. Hidden Test (Implicit): Generate reasoning problems where the value must be inferred
   (e.g., "Mass is 2kg, Energy is 100J..." requires calculating v=10).
3. The Reveal: Run the explicit probe on implicit prompts. If the probe predicts ≈10.0,
   we have evidence of a shared, abstract representation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
from pathlib import Path
import prompt_functions
import joblib

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment Configuration
EXPERIMENT = "velocity"  # Options: "velocity", "current"
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
PLOTS_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_linprob_{EXPERIMENT}")
PLOTS_DIR.mkdir(exist_ok=True)
PROBES_DIR = Path(f"/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/probes_{EXPERIMENT}")
PROBES_DIR.mkdir(exist_ok=True)

# Data Configuration
N_TRAIN_EXPLICIT = 1000  # Number of explicit training samples
N_TEST_PER_FORMAT = 5   # Number of test examples per prompt format

# Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis Configuration
LAYERS_TO_TEST = [7, 15, 23, 31, 47, 55, 63]  # Which layers to analyze
MLP_HIDDEN_SIZE = 128  # Hidden layer size for MLP probe
MLP_LEARNING_RATE = 0.0001  # Learning rate for MLP training
MLP_EPOCHS = 50  # Number of training epochs
MLP_BATCH_SIZE = 256  # Batch size for MLP training

print(f"="*60)
print(f"MLP PROBING ANALYSIS: {EXPERIMENT.upper()}")
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

# Generate implicit test data
test_prompts, test_prompt_ids, test_true_values = gen_implicit()

print(f"Generated {len(test_prompts)} implicit test prompts")
print(f"  Example: '{test_prompts[0]}'")
print(f"  Hidden value: {test_true_values[0]:.1f}")
print(f"  True value range: [{test_true_values.min():.1f}, {test_true_values.max():.1f}]")
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

def train_ridge_probe(train_acts, train_labels, alpha=1.0):
    """
    Train a Ridge regression probe on the given activations and labels.
    
    Args:
        train_acts: numpy array of shape [n_samples, input_dim]
        train_labels: numpy array of shape [n_samples]
        alpha: L2 regularization strength
        
    Returns:
        Trained Ridge model
    """
    probe = Ridge(alpha=alpha)
    probe.fit(train_acts, train_labels)
    return probe

# ==========================================
# ACTIVATION EXTRACTION
# ==========================================

def extract_all_layers_activations(prompts, model, layers, batch_size=16):
    """
    Extract activations from all tokens in prompts at ALL specified layers in a single forward pass.
    Works with multi-GPU distributed models.
    
    Args:
        prompts: List of text prompts
        model: HookedTransformer model
        layers: List of layer indices to extract from
        batch_size: Batch size for processing
    
    Returns:
        Dictionary mapping layer -> numpy array of shape [total_tokens, d_model]
    """
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    all_layer_activations = {layer: [] for layer in layers}
    
    # Get the device of the embedding layer specifically
    embed_device = model.embed.W_E.device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize each prompt individually
        batch_tokens_list = []
        batch_token_lengths = []
        max_len = 0
        
        for prompt in batch_prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            batch_tokens_list.append(tokens)
            batch_token_lengths.append(tokens.shape[1])
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
            # Run with cache for ALL layers at once
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda name: name in hook_names
            )
        
        # Extract activations from all layers
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            batch_acts = cache[hook_name]  # [batch_size, seq_len, d_model]
            
            # Flatten batch and sequence dimensions, excluding padding
            for j in range(len(batch_prompts)):
                n_tokens = batch_token_lengths[j]
                prompt_acts = batch_acts[j, :n_tokens].cpu().float()  # [n_tokens, d_model]
                all_layer_activations[layer].append(prompt_acts)
    
    # Concatenate all activations for each layer
    return {layer: torch.cat(all_layer_activations[layer], dim=0).numpy() for layer in layers}

def extract_all_token_activations(prompts, model, layer, batch_size=16):
    """
    Extract activations from all tokens in prompts at specified layer.
    Works with multi-GPU distributed models.
    
    Args:
        prompts: List of text prompts
        model: HookedTransformer model
        layer: Layer index to extract from
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape [total_tokens, d_model] where total_tokens is sum of all prompt lengths
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_activations = []
    
    # Get the device of the embedding layer specifically
    embed_device = model.embed.W_E.device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize each prompt individually
        batch_tokens_list = []
        batch_token_lengths = []
        max_len = 0
        
        for prompt in batch_prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            batch_tokens_list.append(tokens)
            batch_token_lengths.append(tokens.shape[1])
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
        
        # Extract activations from all tokens (excluding padding)
        batch_acts = cache[hook_name]  # [batch_size, seq_len, d_model]
        
        # Flatten batch and sequence dimensions, excluding padding
        for j in range(len(batch_prompts)):
            n_tokens = batch_token_lengths[j]
            prompt_acts = batch_acts[j, :n_tokens].cpu().float()  # [n_tokens, d_model], convert bfloat16 to float32
            all_activations.append(prompt_acts)
    
    # Concatenate all activations
    return torch.cat(all_activations, dim=0).numpy()

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
        print(f"Warning: Could not find value {value} in prompt: {prompt[:50]}...")
        return 0
    
    # Tokenize and find the token position
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    # Get text for each token
    for i in range(tokens.shape[1]):
        token_str = model.to_string(tokens[0, i])
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

# ==========================================
# TRAIN AND EVALUATE PROBES
# ==========================================

print("Training and evaluating MLP probes...")
print(f"Training on tokens AFTER value appears from {len(train_prompts)} explicit prompts")
print(f"Testing on all tokens from {len(test_prompts)} implicit prompts")
print()

# Group test prompts by prompt_id
unique_prompt_ids = sorted(set(test_prompt_ids))
print(f"Found {len(unique_prompt_ids)} unique prompt formats")
for pid in unique_prompt_ids:
    n_samples = sum(1 for p in test_prompt_ids if p == pid)
    print(f"  Prompt format {pid}: {n_samples} samples")
print()

# Store results per prompt_id
results = {
    'mlp_probes': {},  # Store trained MLP probes per layer
    'linear_probes': {},  # Store trained linear probes per layer
    'by_prompt_id': {}  # Store results grouped by prompt_id (for MLP)
}

linear_results = {
    'by_prompt_id': {}  # Store results grouped by prompt_id (for linear probes)
}

# Initialize storage for each prompt_id (for both probe types)
for pid in unique_prompt_ids:
    results['by_prompt_id'][pid] = {
        'correlations': [],  # One per layer
        'r2_scores': [],
        'mae_scores': [],
        'predictions': {},  # predictions per layer
        'true_values': [],
        'all_predictions': {},  # Store all predictions for scatter plots
        'all_true_values': {}  # Store all true values for scatter plots
    }
    linear_results['by_prompt_id'][pid] = {
        'correlations': [],
        'r2_scores': [],
        'mae_scores': [],
        'predictions': {},
        'true_values': [],
        'all_predictions': {},
        'all_true_values': {}
    }

# Train probes on explicit prompts (only tokens at/after value position)
print("Training MLP and Linear probes on explicit prompts...")

# Get input dimension from first layer
first_layer_acts, first_layer_labels = extract_post_value_activations(
    train_prompts[:1], train_values[:1], model, LAYERS_TO_TEST[0]
)
input_dim = first_layer_acts.shape[1]
print(f"Input dimension: {input_dim}")

for i, layer in enumerate(LAYERS_TO_TEST):
    print(f"\nLayer {layer:2d}:")
    
    # Extract activations only from tokens at/after value position
    train_acts, train_labels = extract_post_value_activations(
        train_prompts, train_values, model, layer
    )
    
    print(f"  Training on {len(train_acts)} tokens (only post-value tokens)...")
    
    # Train MLP probe
    print(f"  Training MLP probe...")
    mlp_probe = train_mlp_probe(
        train_acts, train_labels, 
        input_dim=input_dim,
        hidden_dim=MLP_HIDDEN_SIZE,
        learning_rate=MLP_LEARNING_RATE,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        device=device
    )
    results['mlp_probes'][layer] = mlp_probe
    
    # Train Linear probe
    print(f"  Training Linear (Ridge) probe...")
    linear_probe = train_ridge_probe(train_acts, train_labels, alpha=1.0)
    results['linear_probes'][layer] = linear_probe
    
    # Save probes
    torch.save(mlp_probe.state_dict(), PROBES_DIR / f'mlp_probe_layer_{layer}.pt')
    joblib.dump(linear_probe, PROBES_DIR / f'linear_probe_layer_{layer}.pkl')
    
    print(f"  Training complete")

print(f"\nProbes saved to: {PROBES_DIR}\n")

# Evaluate probes on implicit prompts (per-token analysis), grouped by prompt_id
print("\nEvaluating MLP probes on implicit prompts (per-token position)...")

# First, determine max sequence length for each prompt format
max_seq_lengths = {}
for pid in unique_prompt_ids:
    pid_indices = [idx for idx, p in enumerate(test_prompt_ids) if p == pid]
    pid_prompts = [test_prompts[idx] for idx in pid_indices]
    max_len = max(model.to_tokens(prompt, prepend_bos=True).shape[1] for prompt in pid_prompts)
    max_seq_lengths[pid] = max_len

# Initialize storage for per-token results (for both probe types)
for pid in unique_prompt_ids:
    results['by_prompt_id'][pid]['per_token_correlations'] = np.zeros((len(LAYERS_TO_TEST), max_seq_lengths[pid]))
    results['by_prompt_id'][pid]['per_token_r2_scores'] = np.zeros((len(LAYERS_TO_TEST), max_seq_lengths[pid]))
    results['by_prompt_id'][pid]['per_token_mae_scores'] = np.zeros((len(LAYERS_TO_TEST), max_seq_lengths[pid]))
    results['by_prompt_id'][pid]['per_token_counts'] = np.zeros(max_seq_lengths[pid], dtype=int)
    
    linear_results['by_prompt_id'][pid]['per_token_correlations'] = np.zeros((len(LAYERS_TO_TEST), max_seq_lengths[pid]))
    linear_results['by_prompt_id'][pid]['per_token_r2_scores'] = np.zeros((len(LAYERS_TO_TEST), max_seq_lengths[pid]))
    linear_results['by_prompt_id'][pid]['per_token_mae_scores'] = np.zeros((len(LAYERS_TO_TEST), max_seq_lengths[pid]))
    linear_results['by_prompt_id'][pid]['per_token_counts'] = np.zeros(max_seq_lengths[pid], dtype=int)

# Restructure: Extract activations once per prompt for all layers, then apply all probes
for pid in unique_prompt_ids:
    print(f"\nProcessing prompt format {pid}...")
    
    # Get indices for this prompt_id
    pid_indices = [idx for idx, p in enumerate(test_prompt_ids) if p == pid]
    pid_prompts = [test_prompts[idx] for idx in pid_indices]
    pid_true_values = test_true_values[pid_indices]
    
    # Initialize storage for predictions organized by layer and token position
    mlp_token_predictions = {layer: [[] for _ in range(max_seq_lengths[pid])] for layer in LAYERS_TO_TEST}
    linear_token_predictions = {layer: [[] for _ in range(max_seq_lengths[pid])] for layer in LAYERS_TO_TEST}
    token_true_values = [[] for _ in range(max_seq_lengths[pid])]
    
    # Also collect all predictions for scatter plots
    mlp_all_predictions = {layer: [] for layer in LAYERS_TO_TEST}
    mlp_all_true_values = {layer: [] for layer in LAYERS_TO_TEST}
    linear_all_predictions = {layer: [] for layer in LAYERS_TO_TEST}
    linear_all_true_values = {layer: [] for layer in LAYERS_TO_TEST}
    
    # Process each prompt once, extracting all layers at once
    for prompt_idx, prompt in enumerate(pid_prompts):
        # Extract activations from ALL layers in a single forward pass
        all_layer_acts = extract_all_layers_activations([prompt], model, LAYERS_TO_TEST)
        
        # Apply all probes for all layers
        for layer in LAYERS_TO_TEST:
            test_acts = all_layer_acts[layer]
            mlp_probe = results['mlp_probes'][layer]
            linear_probe = results['linear_probes'][layer]
            
            # Predict using MLP
            with torch.no_grad():
                test_acts_tensor = torch.FloatTensor(test_acts).to(device)
                mlp_predictions = mlp_probe(test_acts_tensor).cpu().numpy().flatten()
            
            # Predict using Linear probe
            linear_predictions = linear_probe.predict(test_acts)
            
            # Store predictions by token position
            n_tokens = len(mlp_predictions)
            for tok_pos in range(n_tokens):
                mlp_token_predictions[layer][tok_pos].append(mlp_predictions[tok_pos])
                linear_token_predictions[layer][tok_pos].append(linear_predictions[tok_pos])
                if layer == LAYERS_TO_TEST[0]:  # Only store true values once
                    token_true_values[tok_pos].append(pid_true_values[prompt_idx])
            
            # Store all predictions for scatter plots
            mlp_all_predictions[layer].extend(mlp_predictions)
            mlp_all_true_values[layer].extend([pid_true_values[prompt_idx]] * n_tokens)
            linear_all_predictions[layer].extend(linear_predictions)
            linear_all_true_values[layer].extend([pid_true_values[prompt_idx]] * n_tokens)
    
    # Compute per-token metrics for all layers
    for layer_idx, layer in enumerate(LAYERS_TO_TEST):
        print(f"  Layer {layer:2d}")
        
        token_predictions = mlp_token_predictions[layer]
        token_predictions_linear = linear_token_predictions[layer]
        
        # Compute per-token metrics for both probe types
        for tok_pos in range(max_seq_lengths[pid]):
            if len(token_predictions[tok_pos]) >= 2:  # Need at least 2 samples for correlation
                # MLP metrics
                preds = np.array(token_predictions[tok_pos])
                trues = np.array(token_true_values[tok_pos])
                
                # Correlation
                if np.std(preds) > 1e-6 and np.std(trues) > 1e-6:
                    corr = np.corrcoef(trues, preds)[0, 1]
                else:
                    corr = 0.0
                
                # R² and MAE
                r2 = r2_score(trues, preds)
                mae = mean_absolute_error(trues, preds)
                
                results['by_prompt_id'][pid]['per_token_correlations'][layer_idx, tok_pos] = corr
                results['by_prompt_id'][pid]['per_token_r2_scores'][layer_idx, tok_pos] = r2
                results['by_prompt_id'][pid]['per_token_mae_scores'][layer_idx, tok_pos] = mae
                
                # Linear probe metrics (using same true values)
                preds_linear = np.array(token_predictions_linear[tok_pos])
                
                if np.std(preds_linear) > 1e-6 and np.std(trues) > 1e-6:
                    corr_linear = np.corrcoef(trues, preds_linear)[0, 1]
                else:
                    corr_linear = 0.0
                
                r2_linear = r2_score(trues, preds_linear)
                mae_linear = mean_absolute_error(trues, preds_linear)
                
                linear_results['by_prompt_id'][pid]['per_token_correlations'][layer_idx, tok_pos] = corr_linear
                linear_results['by_prompt_id'][pid]['per_token_r2_scores'][layer_idx, tok_pos] = r2_linear
                linear_results['by_prompt_id'][pid]['per_token_mae_scores'][layer_idx, tok_pos] = mae_linear
                
                if layer_idx == 0:  # Only count once
                    results['by_prompt_id'][pid]['per_token_counts'][tok_pos] = len(preds)
                    linear_results['by_prompt_id'][pid]['per_token_counts'][tok_pos] = len(preds_linear)
        
        # Report overall stats for both probe types
        valid_corrs_mlp = results['by_prompt_id'][pid]['per_token_correlations'][layer_idx, 
                                                                                results['by_prompt_id'][pid]['per_token_counts'] > 0]
        valid_corrs_linear = linear_results['by_prompt_id'][pid]['per_token_correlations'][layer_idx,
                                                                                linear_results['by_prompt_id'][pid]['per_token_counts'] > 0]
        if len(valid_corrs_mlp) > 0:
            mean_corr_mlp = np.mean(valid_corrs_mlp)
            max_corr_mlp = np.max(valid_corrs_mlp)
            mean_corr_linear = np.mean(valid_corrs_linear)
            max_corr_linear = np.max(valid_corrs_linear)
            print(f"  Format {pid}: MLP Mean={mean_corr_mlp:.3f}, Max={max_corr_mlp:.3f} | Linear Mean={mean_corr_linear:.3f}, Max={max_corr_linear:.3f}")
        else:
            print(f"  Format {pid}: No valid tokens")
    
    # Store all predictions for this prompt format
    for layer in LAYERS_TO_TEST:
        results['by_prompt_id'][pid]['all_predictions'][layer] = np.array(mlp_all_predictions[layer])
        results['by_prompt_id'][pid]['all_true_values'][layer] = np.array(mlp_all_true_values[layer])
        linear_results['by_prompt_id'][pid]['all_predictions'][layer] = np.array(linear_all_predictions[layer])
        linear_results['by_prompt_id'][pid]['all_true_values'][layer] = np.array(linear_all_true_values[layer])

# Find best configuration (layer, token position, prompt format) for both probe types
best_score_mlp = -np.inf
best_config_mlp = None
best_score_linear = -np.inf
best_config_linear = None

for pid in unique_prompt_ids:
    per_token_corrs_mlp = results['by_prompt_id'][pid]['per_token_correlations']
    per_token_counts_mlp = results['by_prompt_id'][pid]['per_token_counts']
    per_token_corrs_linear = linear_results['by_prompt_id'][pid]['per_token_correlations']
    per_token_counts_linear = linear_results['by_prompt_id'][pid]['per_token_counts']
    
    # Only consider positions with data
    for layer_idx, layer in enumerate(LAYERS_TO_TEST):
        for tok_pos in range(max_seq_lengths[pid]):
            if per_token_counts_mlp[tok_pos] > 0:
                corr_mlp = per_token_corrs_mlp[layer_idx, tok_pos]
                if corr_mlp > best_score_mlp:
                    best_score_mlp = corr_mlp
                    best_config_mlp = (layer, tok_pos, pid)
            
            if per_token_counts_linear[tok_pos] > 0:
                corr_linear = per_token_corrs_linear[layer_idx, tok_pos]
                if corr_linear > best_score_linear:
                    best_score_linear = corr_linear
                    best_config_linear = (layer, tok_pos, pid)

print(f"\n{'='*60}")
print(f"BEST CONFIGURATION (MLP):")
print(f"Layer: {best_config_mlp[0]}, Token Position: {best_config_mlp[1]}, Prompt Format: {best_config_mlp[2]}")
print(f"Correlation: {best_score_mlp:.3f}")
print(f"\nBEST CONFIGURATION (LINEAR):")
print(f"Layer: {best_config_linear[0]}, Token Position: {best_config_linear[1]}, Prompt Format: {best_config_linear[2]}")
print(f"Correlation: {best_score_linear:.3f}")
print(f"{'='*60}\n")

# ==========================================
# VISUALIZATION: PER-EXAMPLE HEATMAPS
# ==========================================

print("Generating per-example heatmaps...")

# Process each test example individually for both MLP and Linear probes
for test_idx in range(len(test_prompts)):
    prompt = test_prompts[test_idx]
    true_value = test_true_values[test_idx]
    prompt_id = test_prompt_ids[test_idx]
    
    print(f"Processing test {test_idx + 1}/{len(test_prompts)} (Format {prompt_id})...")
    
    # Tokenize and get tokens for visualization
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    token_strings = [model.to_string(tokens[i]) for i in range(len(tokens))]
    
    # Extract activations for all tokens at all test layers
    activations_dict = {}
    for layer in LAYERS_TO_TEST:
        hook_name = f"blocks.{layer}.hook_resid_post"
        embed_device = model.embed.W_E.device
        input_ids = model.to_tokens(prompt, prepend_bos=True).to(embed_device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        
        layer_acts = cache[hook_name][0]  # [seq_len, d_model]
        activations_dict[layer] = layer_acts.cpu().numpy()
    
    n_tokens = len(token_strings)
    n_layers = len(LAYERS_TO_TEST)
    
    # Store predictions for each layer and token position
    mlp_predictions_matrix = np.zeros((n_layers, n_tokens))
    linear_predictions_matrix = np.zeros((n_layers, n_tokens))
    
    for i, layer in enumerate(LAYERS_TO_TEST):
        layer_acts = activations_dict[layer]  # [seq_len, d_model]
        
        # MLP probe predictions
        mlp_probe = results['mlp_probes'][layer]
        mlp_probe.eval()
        with torch.no_grad():
            mlp_acts_tensor = torch.FloatTensor(layer_acts).to(device)
            mlp_preds = mlp_probe(mlp_acts_tensor).cpu().numpy()
        mlp_predictions_matrix[i, :] = mlp_preds
        
        # Linear probe predictions
        linear_probe = results['linear_probes'][layer]
        linear_preds = linear_probe.predict(layer_acts)
        linear_predictions_matrix[i, :] = linear_preds
    
    # Create heatmap for MLP probe
    fig, ax = plt.subplots(1, 1, figsize=(max(12, n_tokens * 0.5), 8))
    
    # Truncate long tokens for display
    tokens_display = [t[:8] + '...' if len(t) > 8 else t for t in token_strings]
    
    # Heatmap: Predictions with dynamic range based on true value
    vmin = 0
    vmax = max(true_value * 2, mlp_predictions_matrix.max() * 1.1)
    
    im = ax.imshow(mlp_predictions_matrix, aspect='auto', cmap='RdYlGn', 
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'MLP Probe: Predicted {EXPERIMENT.title()} per Token\n(True: {true_value:.1f})', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{l}' for l in LAYERS_TO_TEST])
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tokens_display, rotation=45, ha='right', fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{EXPERIMENT.title()}', fontsize=11)
    
    # Add text annotations for predictions at selected positions
    token_step = max(1, n_tokens // 10)
    layer_step = max(1, n_layers // 8)
    for i in range(0, n_layers, layer_step):
        for j in range(0, n_tokens, token_step):
            text = ax.text(j, i, f'{mlp_predictions_matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    # Add prompt text as subtitle (truncated if too long)
    prompt_display = prompt if len(prompt) < 100 else prompt[:97] + "..."
    plt.suptitle(f'Test Example {test_idx + 1} (Format {prompt_id})\n"{prompt_display}"', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save MLP heatmap
    filename = f'mlp_heatmap_format{prompt_id}_example{test_idx}.png'
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create heatmap for Linear probe
    fig, ax = plt.subplots(1, 1, figsize=(max(12, n_tokens * 0.5), 8))
    
    vmin = 0
    vmax = max(true_value * 2, linear_predictions_matrix.max() * 1.1)
    
    im = ax.imshow(linear_predictions_matrix, aspect='auto', cmap='RdYlGn', 
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'Linear Probe: Predicted {EXPERIMENT.title()} per Token\n(True: {true_value:.1f})', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{l}' for l in LAYERS_TO_TEST])
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tokens_display, rotation=45, ha='right', fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{EXPERIMENT.title()}', fontsize=11)
    
    # Add text annotations
    for i in range(0, n_layers, layer_step):
        for j in range(0, n_tokens, token_step):
            text = ax.text(j, i, f'{linear_predictions_matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.suptitle(f'Test Example {test_idx + 1} (Format {prompt_id})\n"{prompt_display}"', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save Linear heatmap
    filename = f'linear_heatmap_format{prompt_id}_example{test_idx}.png'
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate error at last token for each layer
    mlp_last_token_errors = np.abs(mlp_predictions_matrix[:, -1] - true_value)
    linear_last_token_errors = np.abs(linear_predictions_matrix[:, -1] - true_value)
    
    mlp_best_layer_idx = np.argmin(mlp_last_token_errors)
    linear_best_layer_idx = np.argmin(linear_last_token_errors)
    
    print(f"  Tokens: {n_tokens}")
    print(f"  MLP best layer: {LAYERS_TO_TEST[mlp_best_layer_idx]}, Error: {mlp_last_token_errors[mlp_best_layer_idx]:.1f}")
    print(f"  Linear best layer: {LAYERS_TO_TEST[linear_best_layer_idx]}, Error: {linear_last_token_errors[linear_best_layer_idx]:.1f}")

print("\nAll heatmaps generated!")
print()

# ==========================================
# ANALYSIS SUMMARY
# ==========================================

print(f"\n{'='*60}")
print("ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Experiment: {EXPERIMENT}")
print(f"Training data: {N_TRAIN_EXPLICIT} explicit prompts (post-value tokens)")
print(f"Test data: {len(test_prompts)} implicit prompts ({N_TEST_PER_FORMAT} per format)")
print()
print(f"Generated {len(test_prompts)} heatmap visualizations for each probe type (MLP and Linear)")
print(f"Each heatmap shows predictions across {len(LAYERS_TO_TEST)} layers and all token positions")
print(f"Prompt formats: {unique_prompt_ids}")
print()
print(f"Training Value Statistics:")
print(f"  Mean: {train_values.mean():.2f}")
print(f"  Std: {train_values.std():.2f}")
print(f"  Range: [{train_values.min():.2f}, {train_values.max():.2f}]")
print()
print(f"True Hidden Values (Test):")
print(f"  Mean: {test_true_values.mean():.2f}")
print(f"  Std: {test_true_values.std():.2f}")
print(f"  Range: [{test_true_values.min():.2f}, {test_true_values.max():.2f}]")
print()
print(f"{'='*60}")
print(f"All visualizations saved to: {PLOTS_DIR}")
print(f"{'='*60}")
