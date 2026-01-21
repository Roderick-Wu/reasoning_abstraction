# %%
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import random
import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import argparse

from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyvene as pv
from pyvene import CausalModel
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene import create_mlp_classifier
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)


# %%
# Parse command line arguments
parser = argparse.ArgumentParser(description='DAS training for causal representation learning')
parser.add_argument('--layer', type=int, default=5, help='Layer to intervene on (default: 5)')
args = parser.parse_args()

pv.set_seed(0)
random.seed(0)
torch.manual_seed(0)


# "A {m} kg {obj} has {ke} Joules of Kinetic Energy. How long does it take to travel {d} m?"
# Given mass (m), kinetic energy (KE), and distance (d), compute velocity (v) and then time (t).
variables = ["m", "T", "d", "v", "t"]
values = {}
values["m"] = [i for i in range(2, 10)]
values["T"] = [i for i in range(5, 226, 5)]
values["d"] = [i for i in range(10, 101, 10)]
values["v"] = [(2 * T / m) ** 0.5 for T in values["T"] for m in values["m"]]
values["t"] = [d / v for d in values["d"] for v in values["v"]]

for value in values:
    print(value)
    print(values[value])

parents = {
    "m": [],
    "T": [],
    "d": [],
    "v": ["T", "m"],
    "t": ["d", "v"]
}

functions = {
    "m": lambda : random.choice(values["m"]),
    "T": lambda : random.choice(values["T"]),
    "d": lambda : random.choice(values["d"]),
    "v": lambda T, m: (2 * T / m) ** 0.5,
    "t": lambda d, v: d / v
}

pos = {
    "m": (0, 2),
    "T": (2, 2),
    "d": (4, 2),
    "v": (1, 1),
    "t": (2, 0)
}

causal_model = CausalModel(
    variables=variables,
    parents=parents,
    functions=functions,
    values=values,
    pos=pos
)


# %%
# Load the model with automatic device mapping across available GPUs
model_path = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-72B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16,  # Use float16 instead of float32 to save memory
    device_map="auto"  # Automatically distribute across available GPUs
)
#model_path = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Llama-3.1-8B"
#model = AutoModelForCausalLM.from_pretrained(
    #model_path, 
    #torch_dtype=torch.float32,  
    #device_map="auto"  # Automatically distribute across available GPUs
#)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = model.device  # Get the primary device

# %%
causal_model.print_structure()
print("Timesteps:", causal_model.timesteps)


# %%
n_examples = 200
#import pdb; pdb.set_trace()
dataset_fact = causal_model.generate_factual_dataset(
    size=n_examples, 
    sampler=causal_model.sample_input_tree_balanced
)

# %%
# Test model on factual dataset
#prompt_format = "Question: A {m} kg {obj} has {ke} Joules of Kinetic Energy. How long does it take to travel {d} m? Provide your answer as a single number. Do not show your work or reasoning process. Answer: "
#prompt_format = "Question: A {m} kg {obj} has {ke} Joules of Kinetic Energy. How long does it take to travel {d}? Provide your answer as a single number. Answer: "
prompt_format = "A {m} kg {obj} has {ke} Joules of Kinetic Energy. How long does it take to travel {d}? Provide your answer as a just a single number, without explanation."

import re


v_patterns = [
    r"v\s*=\s*(\d+(?:\.\d+)?)\s*m\s*/\s*s",  # v = 2.8 m/s
    r"(\d+(?:\.\d+)?)\s*=\s*v",               # 6.83130051 = v
    r"v\s*=\s*(\d+(?:\.\d+)?)",               # v = 7.75
    r"(\d+(?:\.\d+)?)\s*m/s",                 # 5.43 m/s
]

t_patterns = [
    r"t\s*=\s*(\d+(?:\.\d+)?)\s*s(?:econds)?",  # t = 7.3 s or t = 7.3 seconds
    r"(\d+(?:\.\d+)?)\s*=\s*t",                  # 8.780107476 = t
    r"t\s*=\s*(\d+(?:\.\d+)?)",                  # t = 12.4
    r"The answer is\s*(\d+(?:\.\d+)?)",          # The answer is 19.38
]

capture_patterns = {
    "v": [re.compile(pat) for pat in v_patterns],
    "t": [re.compile(pat) for pat in t_patterns],
}


import json

# Store all results for later analysis
results = []

for i in range(len(dataset_fact)):
    example = dataset_fact[i]
    objects = ["mass", "car", "train", "ball", "runner", "plane", "rocket", "vehicle"]
    obj = random.choice(objects)
    
    m = int(example["input_ids"][2])
    T = float(example["input_ids"][0])
    d = float(example["input_ids"][1])
    t_true = float(example["labels"][0])

    #input_text = prompt_format.format(m=round(m), obj=obj, ke=round(T), d=round(d))
    question_text = prompt_format.format(m=round(m), obj=obj, ke=round(T), d=round(d))
    
    # Format for instruction-tuned models using chat template
    messages = [
        {"role": "user", "content": question_text}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0])
    
    # Calculate ground truth values
    v_true = (2 * T / m) ** 0.5
    t_true = d / v_true
    
    # Store result
    result = {
        "example_id": i,
        "input": {
            "mass_kg": m,
            "kinetic_energy_J": T,
            "distance_m": d,
            "object_type": obj,
            "prompt": input_text
        },
        "ground_truth": {
            "velocity_m_s": v_true,
            "time_s": t_true
        },
        "model_output": output_text
    }
    
    results.append(result)
    
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset_fact)} examples")

# Save results to JSON
output_file = "model_outputs_with_cot_qwen72b-instruct.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {output_file}")
print("You can now analyze the outputs offline with custom regex patterns.")


# First assessing performance
exit()


print(f"Factual accuracy: {num_correct}/{len(dataset_fact)} = {num_correct/len(dataset_fact):.4f}")

# %%
# Simplified intervention_id for the three main intervention types
def intervention_id_sample(intervention):
    """
    Maps interventions to IDs based on which computational path is intervened:
    - ID 0: Intervene on ones digit path (X0+Y0%10)
    - ID 1: Intervene on tens digit path ((X1+Y1)*10)
    - ID 2: Intervene on carry bit
    """
    if "X0+Y0%10" in intervention:
        return 0
    elif "(X1+Y1)*10" in intervention:
        return 1
    elif "carry" in intervention:
        return 2
    else:
        return -1  # shouldn't happen with controlled sampler

# Controlled intervention sampler - only generates the 3 types we want to train on
def controlled_intervention_sampler():
    """
    Generate only the interventions we want to train on:
    - Type 0: Intervene on ones digit (X0+Y0%10)
    - Type 1: Intervene on tens digit ((X1+Y1)*10)
    - Type 2: Intervene on carry bit
    """
    intervention_type = random.choice([0, 1, 2])
    
    if intervention_type == 0:
        # Intervene on ones digit only
        var = "X0+Y0%10"
        return {var: random.choice(causal_model.values[var])}
    
    elif intervention_type == 1:
        # Intervene on tens digit only
        var = "(X1+Y1)*10"
        return {var: random.choice(causal_model.values[var])}
    
    else:  # type 2
        # Intervene on carry bit
        var = "carry"
        return {var: random.choice(causal_model.values[var])}


# %%
n_samples = 1000
batch_size = 4

dataset_counterfact = causal_model.generate_counterfactual_dataset(
    size=n_samples,
    intervention_id=intervention_id_sample,
    batch_size=batch_size,    
    device=device,
    intervention_sampler=controlled_intervention_sampler,
    sampler=causal_model.sample_input_tree_balanced
)

# %%
# Validate a sample of counterfactual examples from the dataset
print("Validating counterfactual labels in dataset:\n")
print("Checking intervention ID distribution in first 100 examples:")
id_counts = {}
for ex in dataset_counterfact[:100]:
    id_val = ex["intervention_id"].item() if hasattr(ex["intervention_id"], 'item') else ex["intervention_id"][0]
    id_counts[id_val] = id_counts.get(id_val, 0) + 1
print(f"  ID 0 (ones digit): {id_counts.get(0, 0)} examples")
print(f"  ID 1 (tens digit): {id_counts.get(1, 0)} examples")
print(f"  ID 2 (carry): {id_counts.get(2, 0)} examples\n")

intervention_names = {0: "X0+Y0%10", 1: "(X1+Y1)*10", 2: "carry"}

for i in range(min(20, len(dataset_counterfact))):
    example = dataset_counterfact[i]
    
    # Extract the numeric values - input_ids is a tensor with [X1X0, Y1Y0]
    base_inputs = example["input_ids"]  # This is a tensor with 2 elements
    source_inputs = example["source_input_ids"]  # Tensor with sources for each intervention
    intervention_id_val = example["intervention_id"].item() if hasattr(example["intervention_id"], 'item') else example["intervention_id"][0]
    
    # The labels in the dataset should be the counterfactual outputs
    dataset_label = example["labels"].item() if hasattr(example["labels"], 'item') else example["labels"]
    base_label = example["base_labels"].item() if hasattr(example["base_labels"], 'item') else example["base_labels"]
    
    print(f"Example {i}:")
    print(f"  Intervention: {intervention_names[intervention_id_val]} (ID {intervention_id_val})")
    print(f"  Base inputs [X1X0, Y1Y0]: {base_inputs.tolist()}")
    
    # Show source input (always single source now)
    src0 = source_inputs[0].tolist()
    print(f"  Source (for {intervention_names[intervention_id_val]}): {src0}")
    
    # Compute what the values should be
    base_x, base_y = base_inputs.tolist()
    base_ones = (base_x % 10 + base_y % 10) % 10
    base_carry = (base_x % 10 + base_y % 10) // 10
    base_tens = (base_x // 10 + base_y // 10) * 10
    base_computed = base_ones + base_carry * 10 + base_tens
    
    print(f"  Base computation: {base_ones} + {base_carry}*10 + {base_tens} = {base_computed}")
    print(f"  Base label: {base_label}")
    print(f"  Counterfactual label: {dataset_label}")
    print(f"  Difference: {dataset_label - base_label:f}")
    print()

print("\n" + "="*70)
print("Expected behavior:")
print("  - ID 0: Ones digit changes (X0+Y0%10 intervened)")
print("  - ID 1: Tens digit changes ((X1+Y1)*10 intervened)")
print("  - ID 2: Carry bit changes (carry intervened)")
print("="*70)

# %%
# Testing tokenizer

#tokens = tokenizer("14+19=", return_tensors="pt", add_special_tokens=False)
#print(tokens)
#print()
#print(tokens["input_ids"])
#print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist()))

# %%
# DAS Configuration for 3-way causal alignment
# We have 3 intermediate variables in our causal model:
# - X0+Y0%10 (ones digit)
# - carry (carry bit)
# - (X1+Y1)*10 (tens digit)
#

layer = args.layer
print("#################################")
print(f"Using layer: {layer}")
print("#################################")

# Get embedding dimension
embedding_dim = model.config.hidden_size  # Should be 2048 for Llama-3.2-1B


config = IntervenableConfig(
    model_type=type(model),
    representations=[
        # Intervention 1: Learn which rotated dimensions correspond to X0+Y0%10
        RepresentationConfig(
            layer,  # layer
            "block_output",  # component
            "pos",  # intervention unit
            1,  # max number of units
            subspace_partition=None,  # Let training discover the subspace
            intervention_link_key=0,  # 
        ),
        # Intervention 2: Learn which rotated dimensions correspond to (X1+Y1)*10
        RepresentationConfig(
            layer,  # layer
            "block_output",  # component
            "pos",  # intervention unit
            1,  # max number of units
            subspace_partition=None,  # Let training discover the subspace
            intervention_link_key=1,  # 
        ),
        # Intervention 3: Learn which rotated dimensions correspond to carry
        RepresentationConfig(
            layer,  # layer
            "block_output",  # component
            "pos",  # intervention unit
            1,  # max number of units
            subspace_partition=None,  # Let training discover the subspace
            intervention_link_key=2,  # 
        ),
    ],
    intervention_types=BoundlessRotatedSpaceIntervention,
)

print(config)
print(f"\nTraining will learn:")
print(f"  - A single {embedding_dim}×{embedding_dim} rotation matrix")
print(f"  - Which dimensions in rotated space align with X0+Y0%10")
print(f"  - Which dimensions in rotated space align with (X1+Y1)*10")
print(f"  - Which dimensions in rotated space align with carry")
#print(f"\nThis is the core of Distributed Alignment Search (DAS)!") 

# %%
intervenable = IntervenableModel(config, model, use_fast=True)
intervenable.set_device(device)
intervenable.disable_model_gradients()

# %%
epochs = 30
embedding_dim = 2048
gradient_accumulation_steps = 1
total_step = 0
target_total_step = len(dataset_counterfact) * epochs

t_total = int(len(dataset_counterfact) * epochs)

# Set up temperature schedule for BoundlessRotatedSpaceIntervention
temperature_start = 50.0
temperature_end = 0.1
temperature_schedule = (
    torch.linspace(temperature_start, temperature_end, target_total_step)
    .to(torch.bfloat16)
    .to(device)
)

# Optimizer setup - CRITICAL: Must include both rotation matrix AND boundary parameters
optimizer_params = []
for k, v in intervenable.interventions.items():
    optimizer_params += [{"params": v.rotate_layer.parameters()}]
    # Add boundary parameters with higher learning rate for faster convergence
    optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]
    
optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)

# Set initial temperature
for k, v in intervenable.interventions.items():
    v.set_temperature(temperature_schedule[total_step])


def compute_metrics(eval_preds, eval_labels):
    # Both are tensors - compare them directly
    correct = (eval_preds == eval_labels).sum().item()
    total = eval_labels.size(0)
    accuracy = float(correct) / float(total)
    return {"accuracy": accuracy}


def compute_loss(outputs, labels):
    CE = torch.nn.CrossEntropyLoss()
    loss = CE(outputs, labels)
    
    # Add boundary regularization to encourage sparsity
    for k, v in intervenable.interventions.items():
        boundary_loss = 1.0 * v.intervention_boundaries.sum()
        loss += boundary_loss
    
    return loss


def batched_random_sampler(data):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

print(f"Total training steps: {target_total_step}")
print(f"Temperature schedule: {temperature_start} -> {temperature_end}")


# %%
intervenable.model.train()  # train enables drop-off but no grads
print("intervention trainable parameters: ", intervenable.count_parameters())
train_iterator = trange(0, int(epochs), desc="Epoch")

for epoch in train_iterator:
    epoch_iterator = tqdm(
        DataLoader(
            dataset_counterfact,
            batch_size=batch_size,
            sampler=batched_random_sampler(dataset_counterfact),
        ),
        desc=f"Epoch: {epoch}",
        position=0,
        leave=True,
    )
    for batch in epoch_iterator:
        b_size = batch["input_ids"].shape[0]
        
        # Move tensors to device
        for k, v in batch.items():
            if v is not None and isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")
        
        # Convert numerical values to text and tokenize
        # Format: "15+19=" creates consistent token pattern for intervention
        base_texts = [f"{int(batch['input_ids'][i, 0])}+{int(batch['input_ids'][i, 1])}=" for i in range(b_size)]
        source_texts = [f"{int(batch['source_input_ids'][i, 0, 0])}+{int(batch['source_input_ids'][i, 0, 1])}=" for i in range(b_size)]
        
        # Tokenize with consistent padding
        base_tokens = tokenizer(base_texts, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.to(device)
        source_tokens = tokenizer(source_texts, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.to(device)
        
        # Get intervention ID
        intervention_id_val = batch["intervention_id"][0].item()
        
        # Intervene at token position 3 (the "=" token) where the model computes the output
        # Each intervention type learns which dimensions in the rotated space correspond to its causal variable
        
        if intervention_id_val == 0:
            # Intervene on X0+Y0%10 - swap representation at "=" token
            _, counterfactual_outputs = intervenable(
                {"input_ids": base_tokens},
                [{"input_ids": source_tokens}, None, None],
                {"sources->base": ([[[3]] * b_size, None, None], [[[3]] * b_size, None, None])},
            )
        elif intervention_id_val == 1:
            # Intervene on (X1+Y1)*10 - swap representation at "=" token
            _, counterfactual_outputs = intervenable(
                {"input_ids": base_tokens},
                [None, {"input_ids": source_tokens}, None],
                {"sources->base": ([None, [[3]] * b_size, None], [None, [[3]] * b_size, None])},
            )
        elif intervention_id_val == 2:
            # Intervene on carry - swap representation at "=" token
            _, counterfactual_outputs = intervenable(
                {"input_ids": base_tokens},
                [None, None, {"input_ids": source_tokens}],
                {"sources->base": ([None, None, [[3]] * b_size], [None, None, [[3]] * b_size])},
            )

        # Extract logits at the last position (after "=") to predict the output
        last_token_logits = counterfactual_outputs[0][:, -1, :]  # [batch_size, vocab_size]
        
        # Convert labels to token IDs
        label_token_ids = []
        for i in range(b_size):
            label_val = int(batch["labels"][i].item())
            token_id = tokenizer(str(label_val), add_special_tokens=False)["input_ids"][0]
            label_token_ids.append(token_id)
        
        label_token_ids = torch.tensor(label_token_ids, device=device)
        
        # Compute loss (includes boundary regularization)
        loss = compute_loss(last_token_logits, label_token_ids)
        
        epoch_iterator.set_postfix({"loss": loss.item()})

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if total_step % gradient_accumulation_steps == 0:
            if not (gradient_accumulation_steps > 1 and total_step == 0):
                optimizer.step()
                intervenable.set_zero_grad()
                # Update temperature schedule
                if total_step < len(temperature_schedule):
                    for k, v in intervenable.interventions.items():
                        v.set_temperature(temperature_schedule[total_step])
        total_step += 1 

    print(f"Epoch {epoch} completed.")
    print(f"Most recent loss: {loss.item()}")


# %%
# Inspect learned intervention boundaries
print("\n" + "="*70)
print("Learned Intervention Boundaries (as proportion of embedding dimensions)")
print("="*70)

for k, v in intervenable.interventions.items():
    boundary_param = v.intervention_boundaries.item()
    # Apply sigmoid to get actual proportion (BoundlessRotatedSpaceIntervention uses sigmoid)
    boundary_proportion = torch.sigmoid(torch.tensor(boundary_param)).item()
    num_dims = int(boundary_proportion * embedding_dim)
    
    print(f"\nIntervention {k}:")
    print(f"  Raw boundary parameter: {boundary_param:.4f}")
    print(f"  Sigmoid(boundary): {boundary_proportion:.4f}")
    print(f"  Effective dimensions: {num_dims}/{embedding_dim}")
    print(f"  Temperature: {v.get_temperature().item():.4f}")

print("\n" + "="*70)
print("Interpretation:")
print("  - Sigmoid(boundary) is the actual proportion of dimensions used")
print("  - Lower proportion = sparser intervention (fewer dimensions)")
print("  - Each intervention learns different boundaries for different causal variables")
print("  - This shows which causal variables need more/less dimensions")
print("="*70)

# %%
# Save learned rotation matrices and intervention boundaries
print("\n" + "="*70)
print("Saving Learned Rotation Matrices and Boundaries")
print("="*70)

import os
os.makedirs("outputs", exist_ok=True)

save_dict = {
    'layer': layer,
    'embedding_dim': embedding_dim,
    'epochs': epochs,
    'interventions': {},
    'metadata': {
        'variable_names': {0: 'X0+Y0%10', 1: '(X1+Y1)*10', 2: 'carry'},
        'model_name': '../models/Llama-3.2-1B',
    }
}

for k, v in intervenable.interventions.items():
    # Extract intervention index from key (e.g., 'layer_15_...#0' -> 0)
    intervention_idx = int(k.split('#')[-1])
    intervention_name = save_dict['metadata']['variable_names'][intervention_idx]
    boundary_param = v.intervention_boundaries.item()
    # Apply sigmoid to get actual proportion
    boundary_proportion = torch.sigmoid(torch.tensor(boundary_param)).item()
    num_dims = int(boundary_proportion * embedding_dim)
    
    # Extract rotation matrix
    rotation_matrix = v.rotate_layer.weight.data.cpu()
    
    save_dict['interventions'][intervention_idx] = {
        'name': intervention_name,
        'raw_boundary_parameter': boundary_param,
        'boundary_proportion': boundary_proportion,
        'effective_dimensions': num_dims,
        'temperature': v.get_temperature().item(),
        'rotation_matrix': rotation_matrix,
        'intervention_link_key': intervention_idx
    }
    
    print(f"\nIntervention {intervention_idx} ({intervention_name}):")
    print(f"  Rotation matrix shape: {rotation_matrix.shape}")
    print(f"  Raw boundary param: {boundary_param:.4f}")
    print(f"  Sigmoid(boundary): {boundary_proportion:.4f}")
    print(f"  Effective dimensions: {num_dims}/{embedding_dim}")

# Save to file
save_path = f"/home/wuroderi/scratch/das_output_llama/das_rotations_layer{layer}_epochs{epochs}.pt"
torch.save(save_dict, save_path)
print(f"\nSaved to: {save_path}")
print("="*70)


# %% [markdown]
# ## Test Learned Interventions
# 
# Let's verify that the trained model correctly produces counterfactual outputs that match the expected causal structure.

# %%
n_samples = 200
print("Generate test dataset...")
test_dataset_counterfact = causal_model.generate_counterfactual_dataset(
    size=n_samples,
    intervention_id=intervention_id_sample,
    batch_size=batch_size,    
    device=device,
    intervention_sampler=controlled_intervention_sampler,
    sampler=causal_model.sample_input_tree_balanced
)
print("Done")

# %%
# Test the trained intervenable model on some examples
intervenable.model.eval()

print("Testing learned interventions:\n")
num_correct = 0
test_results = []
intervention_accuracies = {0: {'correct': 0, 'total': 0}, 
                          1: {'correct': 0, 'total': 0}, 
                          2: {'correct': 0, 'total': 0}}

for idx in range(len(test_dataset_counterfact)):
        
    example = test_dataset_counterfact[idx]
    
    # Convert numerical values to text and tokenize
    # Format: "15+19=" creates consistent token pattern
    base_text = f"{int(example['input_ids'][0])}+{int(example['input_ids'][1])}="
    source_text = f"{int(example['source_input_ids'][0, 0])}+{int(example['source_input_ids'][0, 1])}="
    
    # Tokenize
    base_tokens = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    source_tokens = tokenizer(source_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    intervention_id = example["intervention_id"].item() if hasattr(example["intervention_id"], 'item') else example["intervention_id"]
    
    # Run intervention at token position 3 (the "=" token)
    with torch.no_grad():
        if intervention_id == 0:
            _, cf_outputs = intervenable(
                {"input_ids": base_tokens},
                [{"input_ids": source_tokens}, None, None],
                {"sources->base": ([[[3]], None, None], [[[3]], None, None])},
            )
        elif intervention_id == 1:
            _, cf_outputs = intervenable(
                {"input_ids": base_tokens},
                [None, {"input_ids": source_tokens}, None],
                {"sources->base": ([None, [[3]], None], [None, [[3]], None])},
            )
        elif intervention_id == 2:
            _, cf_outputs = intervenable(
                {"input_ids": base_tokens},
                [None, None, {"input_ids": source_tokens}],
                {"sources->base": ([None, None, [[3]]], [None, None, [[3]]])},
            )
    
    # Extract prediction from last token logits
    last_token_logits = cf_outputs[0][:, -1, :]
    predicted_token_id = last_token_logits.argmax(1).item()
    
    # Convert predicted token back to number
    predicted_text = tokenizer.decode([predicted_token_id])
    try:
        predicted = int(predicted_text)
    except:
        predicted = -1  # Invalid prediction
    
    # Get expected values
    expected = example["labels"].item() if hasattr(example["labels"], 'item') else example["labels"]
    base_label = example["base_labels"].item() if hasattr(example["base_labels"], 'item') else example["base_labels"]
    
    intervention_names = {0: "X0+Y0%10", 1: "(X1+Y1)*10", 2: "carry"}
    
    is_correct = (predicted == expected)
    num_correct += 1 if is_correct else 0
    
    # Track per-intervention accuracy
    intervention_accuracies[intervention_id]['total'] += 1
    intervention_accuracies[intervention_id]['correct'] += 1 if is_correct else 0
    
    # Store detailed results
    test_results.append({
        'example_id': idx,
        'base_input': base_text,
        'source_input': source_text,
        'intervention_type': intervention_names[intervention_id],
        'intervention_id': int(intervention_id),
        'base_output': int(base_label),
        'expected_counterfactual': int(expected),
        'predicted': int(predicted),
        'correct': bool(is_correct),
        'error': int(predicted - expected) if predicted != -1 else None
    })
    
    #print(f"Example {idx}:")
    #print(f"  Base Input: {base_text}")
    #print(f"  Base output: {base_label}")
    #print(f"  Intervention: {intervention_names[intervention_id]}")
    #print(f"  Source input: {source_text}")
    #print(f"  Expected counterfactual: {expected}")
    #print(f"  Model predicted: {predicted}")

    #print(f"  ✓ Correct" if predicted == expected else f"  ✗ Wrong (diff: {predicted - expected})")
    #print()

accuracy = num_correct / len(test_dataset_counterfact)
print(f"Test accuracy over {len(test_dataset_counterfact)} examples: {accuracy*100:.2f}%")

# Print per-intervention accuracy
print("\nPer-intervention accuracy:")
for interv_id, stats in intervention_accuracies.items():
    if stats['total'] > 0:
        interv_acc = stats['correct'] / stats['total']
        print(f"  {intervention_names[interv_id]}: {interv_acc*100:.2f}% ({stats['correct']}/{stats['total']})")

# Save test results to JSON
import json
test_summary = {
    'layer': layer,
    'epochs': epochs,
    'total_test_examples': len(test_dataset_counterfact),
    'overall_accuracy': float(accuracy),
    'num_correct': int(num_correct),
    'per_intervention_accuracy': {
        intervention_names[k]: {
            'accuracy': float(v['correct'] / v['total']) if v['total'] > 0 else 0.0,
            'correct': int(v['correct']),
            'total': int(v['total'])
        } for k, v in intervention_accuracies.items()
    },
    'detailed_results': test_results
}

json_save_path = f"/home/wuroderi/scratch/das_output_llama/das_test_results_layer{layer}_epochs{epochs}.json"
with open(json_save_path, 'w') as f:
    json.dump(test_summary, f, indent=2)
print(f"\nTest results saved to: {json_save_path}")







