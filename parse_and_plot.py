#!/usr/bin/env python3
"""
Parse model outputs from test_no_cot.txt and create a scatter plot
comparing model predictions against true answers.
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Read the output file
with open('test_no_cot.txt', 'r') as f:
    content = f.read()

# Extract all examples with pattern matching
# Looking for lines like:
# "Question: ... Answer: X.XX<|endoftext|>"
# followed by a true answer line (single number)

# Pattern to find model responses
model_pattern = r'Question:.*?Answer:\s*([0-9.]+)<\|endoftext\|>'
true_pattern = r'^([0-9.]+)$'

# Find all model predictions
model_predictions = []
true_answers = []

# Split by "Example" to process each one
examples = content.split('Example ')

for example in examples[1:]:  # Skip the first split (before first example)
    lines = example.split('\n')
    
    # Find the model's answer (in the Question/Answer line)
    model_answer = None
    for line in lines:
        match = re.search(model_pattern, line)
        if match:
            try:
                model_answer = float(match.group(1))
            except ValueError:
                continue
            break
    
    # Find the true answer (single number on its own line after the model's response)
    true_answer = None
    found_model_response = False
    for line in lines:
        if '<|endoftext|>' in line:
            found_model_response = True
            continue
        if found_model_response:
            match = re.match(true_pattern, line.strip())
            if match:
                try:
                    true_answer = float(match.group(1))
                    break
                except ValueError:
                    continue
    
    if model_answer is not None and true_answer is not None:
        model_predictions.append(model_answer)
        true_answers.append(true_answer)

print(f"Found {len(model_predictions)} valid examples")
print(f"Model predictions range: {min(model_predictions):.2f} to {max(model_predictions):.2f}")
print(f"True answers range: {min(true_answers):.2f} to {max(true_answers):.2f}")

# Convert to numpy arrays
model_predictions = np.array(model_predictions)
true_answers = np.array(true_answers)

# Calculate some statistics
mae = np.mean(np.abs(model_predictions - true_answers))
rmse = np.sqrt(np.mean((model_predictions - true_answers)**2))
correlation = np.corrcoef(model_predictions, true_answers)[0, 1]

print(f"\nStatistics:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Correlation: {correlation:.4f}")

# Create scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(true_answers, model_predictions, alpha=0.5, s=50)

# Add perfect prediction line (y=x)
max_val = max(max(true_answers), max(model_predictions))
min_val = min(min(true_answers), min(model_predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('True Answer (seconds)', fontsize=12)
plt.ylabel('Model Prediction (seconds)', fontsize=12)
plt.title(f'Model Predictions vs True Answers\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, Correlation: {correlation:.4f}', 
          fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Make axes equal scale
plt.axis('equal')
plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])

# Save the plot
plt.tight_layout()
plt.savefig('model_predictions_vs_true.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'model_predictions_vs_true.png'")

# Also create a log-scale version for better visualization if values span large range
if max_val / min_val > 10:  # If values span more than one order of magnitude
    plt.figure(figsize=(10, 10))
    plt.scatter(true_answers, model_predictions, alpha=0.5, s=50)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Answer (seconds)', fontsize=12)
    plt.ylabel('Model Prediction (seconds)', fontsize=12)
    plt.title(f'Model Predictions vs True Answers (Log Scale)\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, Correlation: {correlation:.4f}', 
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('model_predictions_vs_true_log.png', dpi=300, bbox_inches='tight')
    print(f"Log-scale plot saved as 'model_predictions_vs_true_log.png'")

plt.show()
