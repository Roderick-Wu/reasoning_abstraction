#!/usr/bin/env python3
"""
Analyze model outputs from JSON file and extract velocity/time using regex.
"""
import json
import re
from typing import Dict, Optional, Tuple

# Define regex patterns for extracting values
v_patterns = [
    r"v\s*=\s*(\d+(?:\.\d+)?)\s*m\s*/\s*s",  # v = 2.8 m/s or v = 2.8 m / s
    r"(\d+(?:\.\d+)?)\s*=\s*v",               # 6.83130051 = v
    r"v\s*=\s*(\d+(?:\.\d+)?)",               # v = 7.75
    r"(\d+(?:\.\d+)?)\s*m\s*/\s*s",           # 5.43 m/s or 5.43 m / s
    r"velocity\s*[=:]\s*(\d+(?:\.\d+)?)",     # velocity = 5.43 or velocity: 5.43
    r"v\s*≈\s*(\d+(?:\.\d+)?)",               # v ≈ 2.58
]

t_patterns = [
    r"(\d+(?:\.\d+)?)\s*seconds?",                # 19.38 seconds or 10 second
    r"t\s*=\s*(\d+(?:\.\d+)?)\s*s(?:econds)?",  # t = 7.3 s or t = 7.3 seconds
    r"(\d+(?:\.\d+)?)\s*=\s*t",                  # 8.780107476 = t
    r"t\s*=\s*(\d+(?:\.\d+)?)",                  # t = 12.4
    r"time\s*[=:]\s*(\d+(?:\.\d+)?)",           # time = 19 or time: 19
    r"t\s*≈\s*(\d+(?:\.\d+)?)",                  # t ≈ 19.4
    r"The answer is\s*(\d+(?:\.\d+)?)\.\s*\[",  # The answer is 19. [Question - stops before next question
    r"The answer is\s*(\d+(?:\.\d+)?)\.?\s*$",  # The answer is 19 at end
]

# Compile patterns
v_patterns_compiled = [re.compile(pat) for pat in v_patterns]
t_patterns_compiled = [re.compile(pat) for pat in t_patterns]


def extract_value(text: str, patterns: list) -> Optional[float]:
    """Try each pattern and return the last match found."""
    last_value = None
    
    for pattern in patterns:
        # Find all matches for this pattern
        matches = pattern.finditer(text)
        for match in matches:
            try:
                last_value = float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return last_value


def analyze_results(json_file: str) -> Dict:
    """Analyze model outputs and compute statistics."""
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    stats = {
        'total_examples': len(results),
        'velocity_captured': 0,
        'time_captured': 0,
        'velocity_correct_5pct': 0,
        'time_correct_5pct': 0,
        'velocity_errors': [],
        'time_errors': [],
        'examples': []
    }
    
    for result in results:
        example_id = result['example_id']
        output = result['model_output']
        v_true = result['ground_truth']['velocity_m_s']
        t_true = result['ground_truth']['time_s']
        
        # Truncate output at the next question to avoid capturing values from other problems
        # Look for patterns that indicate a new question/answer pair starting
        if len(output) > 50:
            # First, try to find where the answer to the original question ends
            # Look for "The answer is X." followed by more content
            first_answer_match = re.search(r'The answer is \d+(?:\.\d+)?\s*\.\s*\[', output)
            if first_answer_match:
                # Truncate right after the first complete answer
                truncate_at = first_answer_match.end() - 1  # Before the [
                output = output[:truncate_at]
            else:
                # Fall back to looking for new question markers
                next_question_patterns = [
                    output.find('[Question]', 50),
                    output.find('[Answer]', 50),
                    output.find(' Question:', 50),
                    output.find('\nQuestion:', 50),
                    output.find('. Question:', 50)
                ]
                valid_positions = [pos for pos in next_question_patterns if pos != -1]
                if valid_positions:
                    truncate_at = min(valid_positions)
                    output = output[:truncate_at]
        
        # Extract values
        v_captured = extract_value(output, v_patterns_compiled)
        t_captured = extract_value(output, t_patterns_compiled)
        
        # Sanity check: if captured value is way too large, it's likely a garbage output
        # (e.g., model listing multiple choice options)
        if v_captured is not None and v_true > 0:
            if v_captured > v_true * 100 or v_captured < v_true / 100:
                v_captured = None  # Reject unreasonable values
        
        if t_captured is not None and t_true > 0:
            if t_captured > t_true * 100 or t_captured < t_true / 100:
                t_captured = None  # Reject unreasonable values
        
        example_analysis = {
            'example_id': example_id,
            'input': result['input']['prompt'],
            'v_true': v_true,
            't_true': t_true,
            'v_captured': v_captured,
            't_captured': t_captured,
            'v_error_pct': None,
            't_error_pct': None
        }
        
        # Compute errors
        if v_captured is not None:
            stats['velocity_captured'] += 1
            v_error_pct = abs(v_captured - v_true) / v_true * 100
            example_analysis['v_error_pct'] = v_error_pct
            stats['velocity_errors'].append(v_error_pct)
            
            if v_error_pct < 5.0:
                stats['velocity_correct_5pct'] += 1
        
        if t_captured is not None:
            stats['time_captured'] += 1
            t_error_pct = abs(t_captured - t_true) / t_true * 100
            example_analysis['t_error_pct'] = t_error_pct
            stats['time_errors'].append(t_error_pct)
            
            if t_error_pct < 5.0:
                stats['time_correct_5pct'] += 1
        
        stats['examples'].append(example_analysis)
    
    # Compute average errors
    if stats['velocity_errors']:
        stats['avg_velocity_error_pct'] = sum(stats['velocity_errors']) / len(stats['velocity_errors'])
    else:
        stats['avg_velocity_error_pct'] = None
    
    if stats['time_errors']:
        stats['avg_time_error_pct'] = sum(stats['time_errors']) / len(stats['time_errors'])
    else:
        stats['avg_time_error_pct'] = None
    
    return stats


def print_summary(stats: Dict):
    """Print analysis summary."""
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal examples: {stats['total_examples']}")
    print(f"\nVelocity:")
    print(f"  Captured: {stats['velocity_captured']}/{stats['total_examples']} "
          f"({stats['velocity_captured']/stats['total_examples']*100:.1f}%)")
    print(f"  Within 5% of true: {stats['velocity_correct_5pct']}/{stats['velocity_captured']}")
    if stats['avg_velocity_error_pct'] is not None:
        print(f"  Average error: {stats['avg_velocity_error_pct']:.2f}%")
    
    print(f"\nTime:")
    print(f"  Captured: {stats['time_captured']}/{stats['total_examples']} "
          f"({stats['time_captured']/stats['total_examples']*100:.1f}%)")
    print(f"  Within 5% of true: {stats['time_correct_5pct']}/{stats['time_captured']}")
    if stats['avg_time_error_pct'] is not None:
        print(f"  Average error: {stats['avg_time_error_pct']:.2f}%")
    print("=" * 80)


def print_failed_examples(stats: Dict, max_examples: int = 10):
    """Print examples where extraction failed."""
    print("\nFailed extractions (up to first 10):")
    print("-" * 80)
    
    count = 0
    for ex in stats['examples']:
        if (ex['v_captured'] is None or ex['t_captured'] is None) and count < max_examples:
            print(f"\nExample {ex['example_id']}:")
            print(f"Input: {ex['input'][:100]}...")
            if ex['v_captured'] is None:
                print(f"  ❌ Velocity NOT captured (true: {ex['v_true']:.4f})")
            if ex['t_captured'] is None:
                print(f"  ❌ Time NOT captured (true: {ex['t_true']:.4f})")
            count += 1


def print_worst_errors(stats: Dict, max_examples: int = 10):
    """Print examples with the worst errors."""
    print("\nWorst time errors (top 10):")
    print("-" * 80)
    
    # Get examples with time errors, sorted by error percentage
    examples_with_errors = [
        ex for ex in stats['examples'] 
        if ex['t_captured'] is not None and ex['t_error_pct'] is not None
    ]
    examples_with_errors.sort(key=lambda x: x['t_error_pct'], reverse=True)
    
    for ex in examples_with_errors[:max_examples]:
        print(f"\nExample {ex['example_id']}:")
        print(f"  True time: {ex['t_true']:.4f} s")
        print(f"  Captured: {ex['t_captured']:.4f}")
        print(f"  Error: {ex['t_error_pct']:.2f}%")
        print(f"  Input: {ex['input'][:80]}...")


def save_failed_examples(stats: Dict, json_file: str, original_results: list):
    """Save failed extraction examples to a separate file for debugging."""
    failed_examples = []
    
    for i, ex in enumerate(stats['examples']):
        if ex['v_captured'] is None or ex['t_captured'] is None:
            failed_example = {
                'example_id': ex['example_id'],
                'input': ex['input'],
                'model_output': original_results[i]['model_output'],
                'v_true': ex['v_true'],
                't_true': ex['t_true'],
                'v_captured': ex['v_captured'],
                't_captured': ex['t_captured'],
                'v_failed': ex['v_captured'] is None,
                't_failed': ex['t_captured'] is None
            }
            failed_examples.append(failed_example)
    
    output_file = json_file.replace('.json', '_failed_captures.json')
    with open(output_file, 'w') as f:
        json.dump(failed_examples, f, indent=2)
    
    return output_file, len(failed_examples)


if __name__ == "__main__":
    import sys
    
    json_file = sys.argv[1]
    
    print(f"Analyzing {json_file}...")
    
    # Load original results for failed examples
    with open(json_file, 'r') as f:
        original_results = json.load(f)
    
    stats = analyze_results(json_file)
    
    print_summary(stats)
    print_failed_examples(stats)
    print_worst_errors(stats)
    
    # Save detailed analysis
    output_file = json_file.replace('.json', '_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Detailed analysis saved to {output_file}")
    
    # Save failed captures for debugging
    failed_file, failed_count = save_failed_examples(stats, json_file, original_results)
    print(f"✓ Failed captures ({failed_count}) saved to {failed_file}")
