import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import os
from pathlib import Path

# Create plots directory
plots_dir = Path("/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots")
plots_dir.mkdir(exist_ok=True)
print(f"Saving plots to: {plots_dir}")

# 1. Load the Model
# Using Qwen2.5-32B (64 layers, 5120 d_model) from local directory
# device="cuda" if you have a GPU, otherwise "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Check number of available GPUs
n_gpus = torch.cuda.device_count()
print(f"Available GPUs: {n_gpus}")

# Load model using TransformerLens with multi-GPU support
# TransformerLens will load from local HuggingFace cache if available
# Options: "Qwen/Qwen2.5-32B" (non-reasoning) or "Qwen/QwQ-32B-Preview" (reasoning)
model = HookedTransformer.from_pretrained(
    "Qwen/QwQ-32B-Preview",
    device=device,
    n_devices=n_gpus,  # Distribute across multiple GPUs
    dtype="bfloat16",  # Use bfloat16 to reduce memory
    fold_ln=False,  # Skip layer norm folding to avoid temporary memory spike
    fold_value_biases=False,  # Skip value bias folding to avoid device mismatch
    center_writing_weights=False,  # Skip centering to avoid device issues
    center_unembed=False,  # Skip unembed centering
    cache_dir="/home/wuroderi/projects/def-zhijing/wuroderi/models"  # Use local model cache
)

# Get tokenizer from the loaded model
tokenizer = model.tokenizer




import prompt_functions






def extract_activations(prompts, interest_token_indices, model, layers=None):
    """
    Extract activations at the interest token positions across specified layers.
    
    Args:
        prompts: List of prompts
        interest_token_indices: List of token indices for the tokens we are interested in
        model: HookedTransformer model
        layers: List of layer indices to extract from (default: all layers)
    
    Returns:
        Dictionary mapping layer_idx -> tensor of shape [n_samples, d_model]
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    
    # Store activations for each layer
    layer_activations = {layer: [] for layer in layers}
    
    print(f"\nExtracting interest token activations from {len(prompts)} prompts...")
    
    # Determine the device of the embedding layer
    embed_device = next(model.embed.parameters()).device
    
    batch_size = 16  # Process in batches to avoid memory issues
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_indices = interest_token_indices[i:i + batch_size]
        
        # Tokenize batch and move to the correct device
        tokens = model.to_tokens(batch_prompts)
        tokens = tokens.to(embed_device)
        
        # Run model and cache all activations
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        
        # Extract activations at interest token positions for each layer
        for layer in layers:
            # Get residual stream activations after this layer
            # Shape: [batch, seq_len, d_model]
            layer_acts = cache[f"blocks.{layer}.hook_resid_post"]
            
            # Extract activation at interest token position for each prompt
            for j, idx in enumerate(batch_indices):
                interest_act = layer_acts[j, idx, :]  # Shape: [d_model]
                layer_activations[layer].append(interest_act.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(prompts))}/{len(prompts)} prompts")
    
    # Convert lists to tensors
    for layer in layers:
        layer_activations[layer] = torch.stack(layer_activations[layer])
        print(f"  Layer {layer}: {layer_activations[layer].shape}")
    
    return layer_activations


def compute_average_representation(layer_activations):
    """
    Compute the average activation vector for each layer.
    This represents the "interest concept" in the model's latent space.
    
    Args:
        layer_activations: Dictionary mapping layer_idx -> tensor of shape [n_samples, d_model]
    
    Returns:
        Dictionary mapping layer_idx -> average activation vector [d_model]
    """
    avg_representations = {}
    
    print("\nComputing average interest representations...")
    for layer, acts in layer_activations.items():
        avg_representations[layer] = acts.mean(dim=0)
        print(f"  Layer {layer}: average representation shape {avg_representations[layer].shape}")
    
    return avg_representations

def extract_all_token_activations(prompts, model, layers=None):
    """
    Extract activations for ALL tokens in the test prompts.
    
    Args:
        prompts: List of prompts
        model: HookedTransformer model
        layers: List of layer indices to extract from (default: all layers)
    
    Returns:
        List of dictionaries, one per prompt, each containing:
            - 'tokens': token IDs
            - 'token_strs': decoded token strings
            - 'activations': dict mapping layer_idx -> tensor [seq_len, d_model]
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    
    print(f"\nExtracting activations from all tokens in {len(prompts)} test prompts...")
    
    all_prompt_data = []
    
    batch_size = 8  # Smaller batches for test set
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        tokens = model.to_tokens(batch_prompts)
        
        # Run model and cache all activations
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        
        # Process each prompt in batch
        for j, prompt in enumerate(batch_prompts):
            prompt_tokens = tokens[j]
            token_strs = [model.tokenizer.decode([t]) for t in prompt_tokens]
            
            prompt_data = {
                'prompt': prompt,
                'tokens': prompt_tokens.cpu(),
                'token_strs': token_strs,
                'activations': {}
            }
            
            # Extract activations for each layer
            for layer in layers:
                layer_acts = cache[f"blocks.{layer}.hook_resid_post"][j]  # [seq_len, d_model]
                prompt_data['activations'][layer] = layer_acts.cpu()
            
            all_prompt_data.append(prompt_data)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Processed {min(i + batch_size, len(prompts))}/{len(prompts)} prompts")
    
    return all_prompt_data

def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def compute_alignment_scores(test_data, interest_representations, layers=None):
    """
    Compute alignment metrics between test activations and interest representations.
    
    Returns:
        Dictionary with alignment scores
    """
    if layers is None:
        layers = list(interest_representations.keys())
    
    results = {
        'cosine_similarity': [],  # Shape: [n_prompts, n_layers, seq_len]
        'layer_info': layers
    }
    
    print("\nComputing alignment scores...")
    
    for prompt_idx, prompt_data in enumerate(test_data):
        prompt_cosine = {}
        
        for layer in layers:
            test_acts = prompt_data['activations'][layer]  # [seq_len, d_model]
            interest_rep = interest_representations[layer]  # [d_model]
            
            # Cosine Similarity
            cos_sim = cosine_similarity(test_acts, interest_rep.unsqueeze(0))  # [seq_len]
            prompt_cosine[layer] = cos_sim
        
        results['cosine_similarity'].append(prompt_cosine)
        
        if (prompt_idx + 1) % 20 == 0:
            print(f"  Processed {prompt_idx + 1}/{len(test_data)} prompts")
    
    return results


def get_interest_token_ids(tokenizer, interest_terms):
    """
    Get token IDs for interest-related terms.
    """
    print("\nGetting interest token IDs...")
    
    token_ids = []
    for term in interest_terms:
        # Tokenize the term (may be multiple tokens)
        tokens = tokenizer.encode(term, add_special_tokens=False)
        token_ids.extend(tokens)
        print(f"  '{term}' -> token IDs: {tokens}")
    
    # Remove duplicates
    token_ids = list(set(token_ids))
    print(f"  Total unique interest token IDs: {len(token_ids)}")
    
    return token_ids


def compute_entrec_scores(test_data, model, interest_token_ids, layers=None):
    """
    Compute ENTREC-style scores: probability of predicting interest tokens
    from activations at different layers using the unembedding matrix.
    
    Similar to logit lens - apply W_U to activations to get logits, then
    sum probabilities for interest-related tokens.
    
    Args:
        test_data: List of prompt data dictionaries
        model: HookedTransformer model
        interest_token_ids: List of token IDs for interest terms
        layers: List of layer indices
    
    Returns:
        Dictionary with ENTREC scores
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    
    results = {
        'entrec_scores': [],  # Shape: [n_prompts, n_layers, seq_len]
        'layer_info': layers
    }
    
    print("\nComputing ENTREC scores (logit-lens style)...")
    print(f"  Tracking {len(interest_token_ids)} interest token IDs")
    
    # Get unembedding matrix and move to CPU for compatibility
    W_U = model.W_U.cpu()  # Shape: [d_model, vocab_size]
    
    for prompt_idx, prompt_data in enumerate(test_data):
        prompt_entrec = {}
        
        for layer in layers:
            test_acts = prompt_data['activations'][layer]  # [seq_len, d_model] - on CPU
            
            # Apply unembedding: activations @ W_U -> logits
            logits = torch.matmul(test_acts, W_U)  # [seq_len, vocab_size]
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # [seq_len, vocab_size]
            
            # Sum probabilities for interest tokens
            interest_probs = probs[:, interest_token_ids].sum(dim=-1)  # [seq_len]
            
            prompt_entrec[layer] = interest_probs
        
        results['entrec_scores'].append(prompt_entrec)
        
        if (prompt_idx + 1) % 20 == 0:
            print(f"  Processed {prompt_idx + 1}/{len(test_data)} prompts")
    
    return results


def get_term_embeddings(model, tokenizer, terms):
    """
    Get the average embedding of specified terms from the embedding layer.
    """
    
    print("\nExtracting embeddings...")
    
    embeddings = []
    for term in terms:
        # Tokenize the term (may be multiple tokens)
        tokens = tokenizer.encode(term, add_special_tokens=False)
        token_embeds = model.embed.W_E[tokens]  # [n_tokens, d_model]
        # Average if multiple tokens
        avg_embed = token_embeds.mean(dim=0)
        embeddings.append(avg_embed)
        print(f"  '{term}' -> {len(tokens)} token(s)")
    
    # Average across all terms
    avg_embedding = torch.stack(embeddings).mean(dim=0).cpu()
    print(f"  Average embedding shape: {avg_embedding.shape}")
    
    return avg_embedding


def compare_with_embeddings(test_data, avg_embedding, layers=None):
    """
    Compare test activations with the average interest embedding across all layers.
    """
    print("\nComparing test activations with interest term embeddings...")
    
    if layers is None:
        # Get all available layers from first prompt
        layers = list(test_data[0]['activations'].keys())
    
    results = {
        'cosine_similarity': [],  # [n_prompts, n_layers, seq_len]
        'layer_info': layers
    }
    
    for prompt_idx, prompt_data in enumerate(test_data):
        prompt_embed_sim = {}
        
        for layer in layers:
            test_acts = prompt_data['activations'][layer]  # [seq_len, d_model]
            cos_sim = cosine_similarity(test_acts, avg_embedding.unsqueeze(0))
            prompt_embed_sim[layer] = cos_sim
        
        results['cosine_similarity'].append(prompt_embed_sim)
    
    return results


def find_top_aligned_tokens(test_data, alignment_results, top_k=5):
    """
    For each test prompt, find the tokens most aligned with interest representation.
    """
    print("\n" + "="*60)
    print("TOP ALIGNED TOKENS IN TEST PROMPTS")
    print("="*60)
    
    for prompt_idx in range(min(5, len(test_data))):  # Show first 5 prompts
        prompt_data = test_data[prompt_idx]
        print(f"\nPrompt {prompt_idx + 1}: {prompt_data['prompt']}")
        print(f"Tokens: {prompt_data['token_strs']}")
        
        # For each layer, find top-k aligned tokens
        selected_layers = [8, 12, 15]  # Show middle and late layers
        for layer in selected_layers:
            if layer >= len(alignment_results['layer_info']):
                continue
                
            cos_scores = alignment_results['cosine_similarity'][prompt_idx][layer]
            top_indices = torch.topk(cos_scores, min(top_k, len(cos_scores))).indices
            
            print(f"\n  Layer {layer} - Top {top_k} aligned tokens:")
            for rank, idx in enumerate(top_indices):
                token_str = prompt_data['token_strs'][idx]
                score = cos_scores[idx].item()
                print(f"    {rank+1}. Position {idx}: '{token_str}' (cosine: {score:.4f})")


def plot_alignment_heatmaps(test_data, alignment_results, entrec_results=None, embedding_alignment=None, prompt_ids=None, save_dir="align_plots"):
    """
    Create heatmaps showing cosine similarity and ENTREC scores across layers and tokens.
    Groups data by prompt_id and averages scores within each group.
    
    Args:
        test_data: List of prompt data dictionaries
        alignment_results: Alignment scores from compute_alignment_scores
        entrec_results: ENTREC scores from compute_entrec_scores
        embedding_alignment: Embedding alignment scores from compare_with_embeddings
        prompt_ids: List of prompt IDs (one per test prompt) to group by
        save_dir: Directory to save plots
    """
    import seaborn as sns
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING ALIGNMENT HEATMAPS (GROUPED BY PROMPT ID)")
    print("="*60)
    
    if prompt_ids is None:
        print("ERROR: prompt_ids must be provided")
        return
    
    # Group indices by prompt_id
    unique_ids = sorted(set(prompt_ids))
    id_to_indices = {pid: [] for pid in unique_ids}
    for idx, pid in enumerate(prompt_ids):
        id_to_indices[pid].append(idx)
    
    print(f"Found {len(unique_ids)} unique prompt IDs")
    for pid in unique_ids:
        print(f"  Prompt ID {pid}: {len(id_to_indices[pid])} samples")
    
    layers = alignment_results['layer_info']
    
    # Process each unique prompt_id
    for pid_idx, pid in enumerate(unique_ids):
        indices = id_to_indices[pid]
        
        print(f"\nProcessing Prompt ID {pid} ({len(indices)} samples)...")
        
        # Get a representative prompt for the title (use first sample)
        representative_prompt = test_data[indices[0]]['prompt']
        
        # Get token strings from first sample (all samples with same ID should have similar structure)
        token_strs = test_data[indices[0]]['token_strs']
        n_tokens = len(token_strs)
        
        # Average cosine similarity across all samples with this prompt_id
        cos_matrices = []
        for idx in indices:
            cos_matrix = np.zeros((len(layers), n_tokens))
            for layer_idx, layer in enumerate(layers):
                cos_scores = alignment_results['cosine_similarity'][idx][layer]
                # Handle varying token lengths by padding/truncating
                actual_len = min(len(cos_scores), n_tokens)
                # Convert bfloat16 to float32 before numpy conversion
                cos_matrix[layer_idx, :actual_len] = cos_scores[:actual_len].detach().float().cpu().numpy()
            cos_matrices.append(cos_matrix)
        
        # Average across all samples
        avg_cos_matrix = np.mean(cos_matrices, axis=0)
        
        # Create cosine similarity heatmap
        fig, ax = plt.subplots(figsize=(max(12, n_tokens * 0.4), 8))
        
        sns.heatmap(
            avg_cos_matrix,
            xticklabels=token_strs,
            yticklabels=[f"Layer {l}" for l in layers],
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity (Averaged)'},
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={'size': 8}
        )
        
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Model Layer', fontsize=12)
        ax.set_title(f'Concept Alignment (Cosine) - Prompt ID {pid} (n={len(indices)})\n{representative_prompt[:80]}...', 
                     fontsize=14, pad=20)
        
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save cosine plot
        plot_path = save_path / f"cosine_alignment_prompt_id_{pid}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved cosine heatmap: {plot_path}")
        
        # Create ENTREC heatmap if results provided
        if entrec_results is not None:
            entrec_matrices = []
            for idx in indices:
                entrec_matrix = np.zeros((len(layers), n_tokens))
                for layer_idx, layer in enumerate(layers):
                    entrec_scores = entrec_results['entrec_scores'][idx][layer]
                    actual_len = min(len(entrec_scores), n_tokens)
                    entrec_matrix[layer_idx, :actual_len] = entrec_scores[:actual_len].detach().float().cpu().numpy()
                entrec_matrices.append(entrec_matrix)
            
            # Average and scale
            avg_entrec_matrix = np.mean(entrec_matrices, axis=0) * 1000
            
            fig, ax = plt.subplots(figsize=(max(12, n_tokens * 0.4), 8))
            
            sns.heatmap(
                avg_entrec_matrix,
                xticklabels=token_strs,
                yticklabels=[f"Layer {l}" for l in layers],
                cmap="YlOrRd",
                vmin=0,
                vmax=avg_entrec_matrix.max(),
                cbar_kws={'label': 'Sum of Interest Token Probabilities (×10⁻³, Averaged)'},
                ax=ax,
                annot=True,
                fmt=".2f",
                annot_kws={'size': 8}
            )
            
            ax.set_xlabel('Token Position', fontsize=12)
            ax.set_ylabel('Model Layer', fontsize=12)
            ax.set_title(f'ENTREC: Interest Token Prediction Probability (×10⁻³) - Prompt ID {pid} (n={len(indices)})\n{representative_prompt[:80]}...', 
                         fontsize=14, pad=20)
            
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save ENTREC plot
            plot_path = save_path / f"entrec_alignment_prompt_id_{pid}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved ENTREC heatmap: {plot_path}")
        
        # Create Embedding alignment heatmap if results provided
        if embedding_alignment is not None:
            embed_matrices = []
            for idx in indices:
                embed_matrix = np.zeros((len(layers), n_tokens))
                for layer_idx, layer in enumerate(layers):
                    embed_scores = embedding_alignment['cosine_similarity'][idx][layer]
                    actual_len = min(len(embed_scores), n_tokens)
                    embed_matrix[layer_idx, :actual_len] = embed_scores[:actual_len].detach().float().cpu().numpy()
                embed_matrices.append(embed_matrix)
            
            # Average
            avg_embed_matrix = np.mean(embed_matrices, axis=0)
            
            fig, ax = plt.subplots(figsize=(max(12, n_tokens * 0.4), 8))
            
            sns.heatmap(
                avg_embed_matrix,
                xticklabels=token_strs,
                yticklabels=[f"Layer {l}" for l in layers],
                cmap="RdYlGn",
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={'label': 'Cosine Similarity with Select Terms (Averaged)'},
                ax=ax,
                annot=True,
                fmt=".2f",
                annot_kws={'size': 8}
            )
            
            ax.set_xlabel('Token Position', fontsize=12)
            ax.set_ylabel('', fontsize=12)
            ax.set_title(f'Embedding Alignment: Interest Term Similarity - Prompt ID {pid} (n={len(indices)})\n{representative_prompt[:80]}...', 
                         fontsize=14, pad=20)
            
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save embedding plot
            plot_path = save_path / f"embedding_alignment_prompt_id_{pid}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved embedding heatmap: {plot_path}")
    
    print(f"\nAll heatmaps saved to: {save_path}")
    print(f"Total plots created: {len(unique_ids) * (1 + int(entrec_results is not None) + int(embedding_alignment is not None))}")




# Define all experiments to run
experiments = [
    {
        "name": "velocity",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/velocity.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_velocity,
        "terms": ["velocity", "speed", "pace", "rate"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_velocity"
    },
    {
        "name": "current",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/current.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_current,
        "terms": ["current", "ampere", "amp", "flow"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_current"
    },
    {
        "name": "radius",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/radius.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_radius,
        "terms": ["radius", "radii", "distance"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_radius"
    },
    {
        "name": "side_length",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/side_length.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_side_length,
        "terms": ["side", "length", "edge"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_side_length"
    },
    {
        "name": "wavelength",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/wavelength.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_wavelength,
        "terms": ["wavelength", "lambda", "distance"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_wavelength"
    },
    {
        "name": "cross_section",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/cross_section.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_cross_section,
        "terms": ["area", "cross-section", "radius"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_cross_section"
    },
    {
        "name": "displacement",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/displacement.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_displacement,
        "terms": ["displacement", "extension", "stretch"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_displacement"
    },
    {
        "name": "market_cap",
        "explicit_path": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/representations/market_cap.txt",
        "generate_explicit": prompt_functions.gen_explicit_from_file,
        "generate_implicit": prompt_functions.gen_implicit_market_cap,
        "terms": ["capitalization", "valuation", "market"],
        "save_dir": "/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/plots_market_cap"
    }
]

# Run experiments for all prompts
for exp_idx, exp in enumerate(experiments):
    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_idx + 1}/{len(experiments)}: {exp['name'].upper()}")
    print("="*80)
    
    # Generate train and test prompts
    if exp["explicit_path"] is not None:
        train_prompts, interest_token_indices = exp["generate_explicit"](exp["explicit_path"], tokenizer)
    else:
        train_prompts, interest_token_indices = exp["generate_explicit"](n_samples=50)
    
    test_prompts, test_prompt_ids, _ = exp["generate_implicit"](samples_per_prompt=20)
    
    print()
    print(f"Explicit Prompt (Train): '{train_prompts[0]}'")
    print(f"Implicit Prompt (Test):  '{test_prompts[0]}'")
    print(f"Train samples: {len(train_prompts)}, Test samples: {len(test_prompts)}")
    
    # Extract activations from training prompts
    print("\n" + "="*60)
    print("EXTRACTING INTEREST TOKEN ACTIVATIONS FROM TRAIN SET")
    print("="*60)
    
    train_layer_activations = extract_activations(
        train_prompts, 
        interest_token_indices, 
        model, 
        layers=list(range(model.cfg.n_layers))
    )
    
    # Compute average interest representation for each layer
    interest_representations = compute_average_representation(train_layer_activations)
    
    print("\n" + "="*60)
    print("INTEREST REPRESENTATIONS COMPUTED")
    print("="*60)
    print(f"Model has {model.cfg.n_layers} layers")
    print(f"Each representation has dimension {model.cfg.d_model}")
    
    # Extract test prompt activations
    print("\n" + "="*60)
    print("EXTRACTING TEST PROMPT ACTIVATIONS")
    print("="*60)
    
    test_data = extract_all_token_activations(test_prompts, model)
    
    # Cosine alignment with representations
    print("\n" + "="*60)
    print("COMPUTING ALIGNMENT WITH SELECT REPRESENTATIONS")
    print("="*60)
    alignment_results = compute_alignment_scores(test_data, interest_representations)
    
    # ENTREC: Most promising method - kept active
    print("\n" + "="*60)
    print("COMPUTING ENTREC SCORES (LOGIT-LENS)")
    print("="*60)
    
    interest_token_ids = get_interest_token_ids(tokenizer, exp["terms"])
    entrec_results = compute_entrec_scores(test_data, model, interest_token_ids)
    
    # COMMENTED OUT: Embedding alignment
    # print("\n" + "="*60)
    # print("COMPARING WITH INTEREST TERM EMBEDDINGS")
    # print("="*60)
    # avg_interest_embedding = get_term_embeddings(model, tokenizer, exp["terms"])
    # embedding_alignment = compare_with_embeddings(test_data, avg_interest_embedding)
    
    # Show top aligned tokens
    find_top_aligned_tokens(test_data, alignment_results)
    
    # Visualize cosine alignment and ENTREC results
    print("\n" + "="*60)
    print("VISUALIZING ALIGNMENTS")
    print("="*60)
    
    plot_alignment_heatmaps(
        test_data, 
        alignment_results,
        entrec_results,
        None,  # embedding_alignment - keep commented out
        prompt_ids=test_prompt_ids,
        save_dir=exp["save_dir"]
    )
    
    print(f"\nCompleted experiment: {exp['name']}")
    print(f"Results saved to: {exp['save_dir']}")

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETED")
print("="*80) 





