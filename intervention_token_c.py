"""
Unified CoT Token Intervention Runner.

This merges the previous Experiment C masking/truncation interventions with the
original value-patching intervention in one script so all conditions can be
compared side-by-side.

Condition blocks
----------------
1) censoring/masking:
   - blank, whitespace, underscores, redacted, variable
2) value patching:
   - patch_from_trace (reasonable swap)
   - patch_zero
   - patch_pos_inf
   - patch_neg_inf
   - patch_large_random (uniform integer in [1000, 10000])
3) truncation:
   - jump_to_final
   - jump_to_answer

Metrics
-------
- accuracy_vs_real: correctness against original expected answer
- accuracy_vs_patched: correctness against counterfactual expected answer under
  patched hidden value (when finite)
- moved_toward_patched: whether prediction is closer to patched target than to
  the original target (when both finite)

Usage
-----
python intervention_token_c.py --experiment velocity
python intervention_token_c.py --experiment velocity --n_traces 90
python intervention_token_c.py --experiment velocity --blocks value_patching
"""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# PER-EXPERIMENT CONFIGURATION
# ==========================================

EXPERIMENT_CONFIGS: Dict[str, Dict] = {
    "velocity": {
        "hidden_var_key": "v",
        "answer_key": "expected_time",
        "jump_to_symbol": "t = ",
        "var_placeholder": "$VELOCITY",
        "assign_prefix_re": r"\bv\s*=\s*$",
        "forward_var_patterns": [r"(?<![A-Za-z0-9_/])(?:v|V)\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[vV]\b(?!\s*[\^*/])"],
        "answer_power": -1.0,
    },
    "current": {
        "hidden_var_key": "i",
        "answer_key": "expected_charge",
        "jump_to_symbol": "Q = ",
        "var_placeholder": "$CURRENT",
        "assign_prefix_re": r"\bi\s*=\s*$",
        "forward_var_patterns": [r"(?<![A-Za-z0-9_/])(?:i|I)\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[iI]\b(?!\s*[\^*/])"],
        "answer_power": 1.0,
    },
    "radius": {
        "hidden_var_key": "r",
        "answer_key": "expected_circumference",
        "jump_to_symbol": "C = ",
        "var_placeholder": "$RADIUS",
        "assign_prefix_re": r"\br\s*=\s*$",
        "forward_var_patterns": [r"(?<![A-Za-z0-9_/])(?:r|R)\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[rR]\b(?!\s*[\^*/])"],
        "answer_power": 1.0,
    },
    "side_length": {
        "hidden_var_key": "s",
        "answer_key": "expected_surface_area",
        "jump_to_symbol": "SA = ",
        "var_placeholder": "$SIDE",
        "assign_prefix_re": r"\bs\s*=\s*$",
        "forward_var_patterns": [
            r"(?<![A-Za-z0-9_/])(?:s|S)\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)",
            r"\bside\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        ],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[sS]\b(?!\s*[\^*/])"],
        "answer_power": 2.0,
    },
    "wavelength": {
        "hidden_var_key": "wavelength",
        "answer_key": "expected_distance",
        "jump_to_symbol": "d = ",
        "var_placeholder": "$WAVELENGTH",
        "assign_prefix_re": r"\bwavelength\s*=\s*$|[Ll]\s*ambda\s*=\s*$",
        "forward_var_patterns": [r"\bwavelength\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", r"\blambda\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*wavelength\b", r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*lambda\b"],
        "answer_power": 1.0,
    },
    "cross_section": {
        "hidden_var_key": "area",
        "answer_key": "expected_volume",
        "jump_to_symbol": "V = ",
        "var_placeholder": "$AREA",
        "assign_prefix_re": r"\barea\s*=\s*$|\bA\s*=\s*$",
        "forward_var_patterns": [r"\barea\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", r"(?<![A-Za-z0-9_/])A\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [
            r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*area\b(?!\s*[\^*/])",
            r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*A\b(?!\s*[\^*/])",
        ],
        "answer_power": 1.0,
    },
    "displacement": {
        "hidden_var_key": "x",
        "answer_key": "expected_pe",
        "jump_to_symbol": "PE = ",
        "var_placeholder": "$DISPLACEMENT",
        "assign_prefix_re": r"\bx\s*=\s*$",
        "forward_var_patterns": [r"(?<![A-Za-z0-9_/])x\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*x\b(?!\s*[\^*/])"],
        "answer_power": 2.0,
    },
    "market_cap": {
        "hidden_var_key": "market_cap",
        "answer_key": "expected_pe",
        "jump_to_symbol": "P/E = ",
        "var_placeholder": "$MARKET_CAP",
        "assign_prefix_re": r"\bmarket[\s_]?cap\s*=\s*$",
        "forward_var_patterns": [r"\bmarket[\s_]?cap\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"],
        "reverse_var_patterns": [r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*market[\s_]?cap\b"],
        "answer_power": 1.0,
    },
}


CONDITION_GROUPS: Dict[str, List[str]] = {
    "censoring_masking": ["blank", "whitespace", "underscores", "redacted", "variable"],
    "value_patching": [
        "patch_from_trace",
        "patch_zero",
        "patch_pos_inf",
        "patch_neg_inf",
        "patch_large_random",
    ],
    "truncation": ["jump_to_final", "jump_to_answer"],
    "baseline": ["no_cot"],
}


# ==========================================
# ARGUMENT PARSING
# ==========================================

parser = argparse.ArgumentParser(
    description="Unified intervention runner: censoring + value patching + truncation"
)
parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS))
parser.add_argument("--n_traces", type=int, default=None)
parser.add_argument("--model_path", default="/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B")
parser.add_argument("--traces_root", default=os.path.expanduser("~/links/scratch/reasoning_traces/Qwen2.5-32B"))
parser.add_argument("--blocks", nargs="+", default=["censoring_masking", "value_patching", "truncation", "baseline"],
                    choices=list(CONDITION_GROUPS))
parser.add_argument("--conditions", nargs="+", default=None,
                    help="Optional explicit condition subset; overrides --blocks")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

random.seed(args.seed)

cfg = EXPERIMENT_CONFIGS[args.experiment]
HIDDEN_VAR_KEY = cfg["hidden_var_key"]
ANSWER_KEY = cfg["answer_key"]
JUMP_TO_SYMBOL = cfg["jump_to_symbol"]
VAR_PLACEHOLDER = cfg["var_placeholder"]
ASSIGN_PREFIX_RE = cfg["assign_prefix_re"]
FORWARD_VAR_PATTERNS = cfg["forward_var_patterns"]
REVERSE_VAR_PATTERNS = cfg["reverse_var_patterns"]
ANSWER_POWER = float(cfg["answer_power"])

EXPERIMENT = args.experiment
MODEL_PATH = args.model_path
TRACES_METADATA_FILE = Path(args.traces_root) / EXPERIMENT / "traces_metadata.json"

MAX_TOKENS_AFTER_INTERVENTION = 256
TEMPERATURE = 0.0
TOP_P = 1.0
RELATIVE_TOLERANCE = 0.05

OUTPUT_DIR = Path(
    "/home/wuroderi/links/projects/def-rgrosse/wuroderi/reasoning_abstraction"
    "/intervention_token_results"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def get_selected_conditions() -> List[str]:
    if args.conditions:
        return args.conditions
    selected: List[str] = []
    for block in args.blocks:
        selected.extend(CONDITION_GROUPS[block])
    # preserve order, remove duplicates
    return list(dict.fromkeys(selected))


ALL_CONDITIONS = get_selected_conditions()

CONDITION_TO_GROUP = {}
for group_name, conditions in CONDITION_GROUPS.items():
    for cond in conditions:
        CONDITION_TO_GROUP[cond] = group_name

CONDITION_DESCRIPTIONS = {
    "blank": f"Remove {HIDDEN_VAR_KEY} value entirely",
    "whitespace": f"Replace {HIDDEN_VAR_KEY} with spaces",
    "underscores": f"Replace {HIDDEN_VAR_KEY} with underscores",
    "redacted": "Replace hidden value with [REDACTED]",
    "variable": f"Replace hidden value with {VAR_PLACEHOLDER}",
    "patch_from_trace": "Patch hidden value from another trace",
    "patch_zero": "Patch hidden value to 0",
    "patch_pos_inf": "Patch hidden value to infinity token",
    "patch_neg_inf": "Patch hidden value to negative infinity token",
    "patch_large_random": "Patch hidden value to random int in [1000,10000]",
    "jump_to_final": f"Truncate before hidden value, append '... {JUMP_TO_SYMBOL}'",
    "jump_to_answer": "Truncate before hidden value, append '... The answer is '",
    "no_cot": "No chain-of-thought prompt baseline",
}


print("=" * 80)
print("UNIFIED INTERVENTION RUNNER")
print("=" * 80)
print(f"Experiment          : {EXPERIMENT}")
print(f"Hidden var / answer : {HIDDEN_VAR_KEY} / {ANSWER_KEY}")
print(f"Model               : {MODEL_PATH}")
print(f"Traces              : {TRACES_METADATA_FILE}")
print(f"Selected blocks     : {args.blocks}")
print(f"Selected conditions : {ALL_CONDITIONS}")
print(f"Output dir          : {OUTPUT_DIR}")
print()


# ==========================================
# DATA + MODEL
# ==========================================

with open(TRACES_METADATA_FILE) as f:
    traces = json.load(f)

if args.n_traces is not None:
    traces = traces[:args.n_traces]

usable_traces = [
    t for t in traces
    if t.get(HIDDEN_VAR_KEY) is not None and t.get(ANSWER_KEY) is not None and t.get("generated_text")
]

print(f"Usable traces: {len(usable_traces)} / {len(traces)}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"Model loaded: {model.config.num_hidden_layers} layers")
print()


# ==========================================
# UTILS
# ==========================================

def safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def find_all_occurrences(text: str, value: float, tolerance: float = 1.0) -> List[Tuple[int, int, str]]:
    patterns = [
        rf"\b{int(value)}\b(?!\.\d)",
        rf"\b{int(value)}\.0+\b",
        rf"\b{value:.1f}\b",
        rf"\b{value:.2f}\b",
        rf"\b{value:.4f}\b",
    ]
    seen: List[Tuple[int, int, str]] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            if not any(s <= m.start() < e for s, e, _ in seen):
                seen.append((m.start(), m.end(), m.group()))

    if not seen:
        for m in re.finditer(r"\b(\d+\.?\d*)\b", text):
            try:
                if abs(float(m.group(1)) - value) <= tolerance:
                    seen.append((m.start(), m.end(), m.group()))
            except ValueError:
                continue

    seen.sort(key=lambda x: x[0])
    return seen


def find_last_occurrence(text: str, value: float, tolerance: float = 1.0) -> Optional[Tuple[int, int, str]]:
    occ = find_all_occurrences(text, value, tolerance)
    return occ[-1] if occ else None


def canonical_trace_text(trace: dict) -> Optional[str]:
    """Return a single canonical text for the trace without duplicating the prompt.

    In these trace files, `generated_text` already includes the prompt prefix.
    Older code concatenated `prompt + generated_text`, which duplicated the
    question and polluted modified prompts.
    """
    prompt = trace.get("prompt") or ""
    generated_text = trace.get("generated_text")
    if not generated_text:
        return None
    if prompt and generated_text.startswith(prompt):
        return generated_text
    return prompt + generated_text


def extract_written_hidden_var(text: str, start_idx: int = 0) -> Optional[Tuple[int, int, float, str]]:
    """Find the first written hidden-variable value span in the first-problem text."""
    matches: List[Tuple[int, int, float, str]] = []

    for pattern in FORWARD_VAR_PATTERNS:
        for match in re.finditer(pattern, text):
            number = match.group(1)
            value = safe_float(number)
            if value is None:
                continue
            start = match.start(1)
            end = match.end(1)
            if start < start_idx:
                continue
            matches.append((start, end, value, number))

    for pattern in REVERSE_VAR_PATTERNS:
        for match in re.finditer(pattern, text):
            number = match.group(1)
            value = safe_float(number)
            if value is None:
                continue
            start = match.start(1)
            end = match.end(1)
            if start < start_idx:
                continue
            matches.append((start, end, value, number))

    if matches:
        matches.sort(key=lambda item: item[0])
        return matches[0]

    return None


def get_text_up_to_hidden_var(trace: dict) -> Optional[Tuple[str, float]]:
    metadata_hv = safe_float(trace.get(HIDDEN_VAR_KEY))

    full_text = canonical_trace_text(trace)
    if full_text is None:
        return None

    # Restrict search to the first answered question so the follow-on problem
    # cannot contaminate the intervention prefix.
    first_problem_text = truncate_at_next_question(full_text)
    jump_idx = first_problem_text.find(JUMP_TO_SYMBOL)
    search_text = first_problem_text[:jump_idx] if jump_idx != -1 else first_problem_text

    # Prefer matches that occur after CoT starts to avoid latching onto earlier
    # prompt numerals or late malformed placeholder text.
    cot_start = 0
    cot_match = re.search(r"Answer\s*\(step-by-step\)\s*:", search_text, re.IGNORECASE)
    if cot_match:
        cot_start = cot_match.end()

    if metadata_hv is not None:
        hv_occ = [m for m in find_all_occurrences(search_text, metadata_hv) if m[0] >= cot_start]
        if hv_occ:
            _, end, _ = hv_occ[0]
            return search_text[:end], metadata_hv

    written = extract_written_hidden_var(search_text, start_idx=cot_start)
    if written is not None:
        _, end, written_hv, _ = written
        return search_text[:end], written_hv

    if metadata_hv is None:
        return None

    all_occ = find_all_occurrences(search_text, metadata_hv)
    if not all_occ:
        return None

    _, end, _ = all_occ[0]
    return search_text[:end], metadata_hv


def truncate_at_next_question(text: str) -> str:
    idx = text.find("Question", 1)
    return text[:idx] if idx != -1 else text


def extract_final_answer(text: str, truncate: bool = True) -> Optional[float]:
    if truncate:
        text = truncate_at_next_question(text)

    inf_patterns = [
        r"(?:answer|result|final|therefore).*?(-?inf(?:inity)?)",
        rf"{re.escape(JUMP_TO_SYMBOL.strip())}\s*(-?inf(?:inity)?)",
        r"\b(-?inf(?:inity)?)\b",
    ]
    for pattern in inf_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            s = matches[-1].group(1).lower()
            return float("-inf") if s.startswith("-") else float("inf")

    num_pat = r'-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'

    # Priority 1: explicit JUMP_TO_SYMBOL marker
    jump_pat = rf"{re.escape(JUMP_TO_SYMBOL.strip())}\s*({num_pat})"
    matches = list(re.finditer(jump_pat, text, re.IGNORECASE))
    if matches:
        try:
            return float(matches[-1].group(1))
        except (ValueError, IndexError):
            pass

    # Priority 2: "answer is [approximately] X" — exact phrase avoids grabbing
    # intermediate numbers (the old .*? pattern was matching the first number
    # after keywords like "therefore", not the final answer value).
    ans_pat = rf'(?:the\s+)?answer\s+is\s*(?:approximately\s+)?({num_pat})'
    matches = list(re.finditer(ans_pat, text, re.IGNORECASE))
    if matches:
        try:
            return float(matches[-1].group(1))
        except (ValueError, IndexError):
            pass

    # Priority 3: near end of text — "= X <unit>" or "is X <unit>"
    # Catches outputs like "t = 69 / 94.2 = 0.732 seconds. Therefore, ..."
    tail = text[-400:]
    unit_pat = rf'(?:=|≈|is)\s*(?:approximately\s+)?({num_pat})\s*[a-zA-Z/]+(?:\b|$)'
    matches = list(re.finditer(unit_pat, tail, re.IGNORECASE))
    if matches:
        try:
            return float(matches[-1].group(1))
        except (ValueError, IndexError):
            pass

    # Priority 4: last number on the final line (broadest fallback)
    last_line = text.rstrip().rsplit('\n', 1)[-1]
    matches = list(re.finditer(num_pat, last_line))
    if matches:
        try:
            return float(matches[-1].group())
        except (ValueError, IndexError):
            pass

    return None


def continue_generation(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS_AFTER_INTERVENTION,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            top_p=TOP_P if TEMPERATURE > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def is_correct(predicted: Optional[float], expected: Optional[float]) -> bool:
    if predicted is None or expected is None:
        return False
    if math.isinf(expected) and math.isinf(predicted):
        return math.copysign(1.0, expected) == math.copysign(1.0, predicted)
    if math.isinf(expected):
        return False
    if expected == 0:
        return abs(predicted) <= 1e-9
    return abs(predicted - expected) / abs(expected) <= RELATIVE_TOLERANCE


def expected_under_patch(expected_real: float, old_hv: float, new_hv: float) -> Optional[float]:
    # Scaling law: expected' = expected * (new_hv / old_hv)^power
    if old_hv == 0:
        return None
    ratio = new_hv / old_hv
    try:
        return expected_real * (ratio ** ANSWER_POWER)
    except OverflowError:
        return float("inf")
    except ZeroDivisionError:
        return float("inf")
    except ValueError:
        return None


def value_to_text(new_value: float, matched_text: str) -> str:
    if math.isinf(new_value):
        return "infinity" if new_value > 0 else "-infinity"
    if float(new_value).is_integer() and "." not in matched_text:
        return str(int(new_value))
    if "." in matched_text:
        dp = len(matched_text.split(".")[-1])
        return f"{new_value:.{dp}f}"
    return f"{new_value:.4f}".rstrip("0").rstrip(".")


def build_no_cot_text(trace: dict) -> Optional[str]:
    prompt = trace.get("prompt")
    if prompt is None:
        return None
    return re.sub(r"Answer \(step-by-step\):\s*", "Answer: ", prompt, flags=re.IGNORECASE)


def compute_patch_target(condition: str, idx: int, donor_hv: float) -> Optional[float]:
    if condition == "patch_from_trace":
        return donor_hv
    if condition == "patch_zero":
        return 0.0
    if condition == "patch_pos_inf":
        return float("inf")
    if condition == "patch_neg_inf":
        return float("-inf")
    if condition == "patch_large_random":
        return float(random.randint(1000, 10000))
    return None


def build_condition_text(
    text_up_to_hv: str,
    hv_value: float,
    condition: str,
    patch_value: Optional[float],
) -> Optional[str]:
    loc = find_last_occurrence(text_up_to_hv, hv_value)
    if loc is None:
        return None

    start, _, matched = loc
    text_before = text_up_to_hv[:start]
    char_len = len(matched)

    def with_explicit_mask_assignment(substitute: str) -> str:
        """Force masked forms to look like '<var> = <substitute>' to avoid ambiguity."""
        stripped = text_before.rstrip()
        if re.search(ASSIGN_PREFIX_RE, stripped, flags=re.IGNORECASE):
            return stripped + " " + substitute
        return stripped + f" {HIDDEN_VAR_KEY} = " + substitute

    def with_explicit_value_assignment(value_text: str) -> str:
        """Force value patches to look like '<var> = <value>' and keep a trailing space."""
        stripped = text_before.rstrip()
        if re.search(ASSIGN_PREFIX_RE, stripped, flags=re.IGNORECASE):
            return stripped + " " + value_text + " "
        return stripped + f" {HIDDEN_VAR_KEY} = " + value_text + " "

    if condition == "blank":
        return with_explicit_mask_assignment("")
    if condition == "whitespace":
        return with_explicit_mask_assignment(" " * char_len)
    if condition == "underscores":
        return with_explicit_mask_assignment("_" * char_len)
    if condition == "redacted":
        return with_explicit_mask_assignment("[REDACTED]")
    if condition == "variable":
        return with_explicit_mask_assignment(VAR_PLACEHOLDER)
    if condition == "jump_to_final":
        stripped = re.sub(ASSIGN_PREFIX_RE, "", text_before.rstrip(), flags=re.IGNORECASE)
        return stripped.rstrip() + "\n... " + JUMP_TO_SYMBOL
    if condition == "jump_to_answer":
        stripped = re.sub(ASSIGN_PREFIX_RE, "", text_before.rstrip(), flags=re.IGNORECASE)
        return stripped.rstrip() + "\n... The answer is "

    if condition.startswith("patch_"):
        if patch_value is None:
            return None
        return with_explicit_value_assignment(value_to_text(patch_value, matched))

    return None


def moved_toward_patched(pred: Optional[float], real: float, patched: Optional[float]) -> Optional[bool]:
    if pred is None or patched is None:
        return None
    if not math.isfinite(real) or not math.isfinite(patched) or not math.isfinite(pred):
        return None
    dist_real = abs(pred - real)
    dist_patch = abs(pred - patched)
    return dist_patch < dist_real


def patch_effect_size(pred: Optional[float], real: float, patched: Optional[float]) -> Optional[float]:
    if pred is None or patched is None:
        return None
    if not math.isfinite(real) or not math.isfinite(patched) or not math.isfinite(pred):
        return None
    denom = abs(real - patched)
    if denom < 1e-12:
        return None
    # 0 means exactly at real, 1 means exactly at patched
    return max(0.0, min(1.0, 1.0 - (abs(pred - patched) / denom)))


# ==========================================
# BASELINE: UNMODIFIED COT
# ==========================================

print("Computing unmodified CoT baseline...")
cot_results = []
for trace in usable_traces:
    prompt = trace.get("prompt")
    expected = safe_float(trace.get(ANSWER_KEY))
    entry = {"id": trace["id"], "expected_real": expected}
    if prompt is None or expected is None:
        entry["success"] = False
        entry["correct_real"] = False
        cot_results.append(entry)
        continue
    try:
        full_text = continue_generation(prompt)
        new_text = full_text[len(prompt):]
        pred = extract_final_answer(new_text)
        entry["success"] = True
        entry["predicted"] = pred
        entry["correct_real"] = is_correct(pred, expected)
        entry["new_text"] = new_text
    except Exception as e:
        entry["success"] = False
        entry["error"] = str(e)
        entry["correct_real"] = False
    cot_results.append(entry)

cot_success = [r for r in cot_results if r.get("success")]
cot_corr = [r for r in cot_success if r.get("correct_real")]
cot_acc = len(cot_corr) / len(cot_success) if cot_success else float("nan")
print(f"  unmod_cot baseline success: {len(cot_success)}/{len(cot_results)}")
print(f"  unmod_cot baseline accuracy: {cot_acc:.1%} ({len(cot_corr)}/{len(cot_success)})")
print()


# ==========================================
# PREP DONOR VALUES FOR patch_from_trace
# ==========================================

trace_infos: List[Optional[Tuple[str, float]]] = [get_text_up_to_hidden_var(t) for t in usable_traces]
written_hv_list = [info[1] if info is not None else None for info in trace_infos]
donor_values = written_hv_list[1:] + written_hv_list[:1]


# ==========================================
# MAIN LOOP
# ==========================================

condition_results: Dict[str, List[dict]] = {c: [] for c in ALL_CONDITIONS}

for idx, trace in enumerate(usable_traces):
    trace_id = trace["id"]
    expected_real = safe_float(trace.get(ANSWER_KEY))
    info = trace_infos[idx]
    donor_hv = donor_values[idx]

    if info is None or donor_hv is None or expected_real is None:
        for condition in ALL_CONDITIONS:
            condition_results[condition].append({
                "id": trace_id,
                "success": False,
                "error": "missing_written_hv_or_expected",
            })
        continue

    text_up_to_hv, hv_value = info
    print(f"[{idx+1}/{len(usable_traces)}] id={trace_id} {HIDDEN_VAR_KEY}={hv_value} expected={expected_real}")

    for condition in ALL_CONDITIONS:
        patch_value = compute_patch_target(condition, idx, float(donor_hv))
        expected_patch = None
        if patch_value is not None:
            expected_patch = expected_under_patch(expected_real, hv_value, patch_value)

        if condition == "no_cot":
            modified_text = build_no_cot_text(trace)
        else:
            modified_text = build_condition_text(text_up_to_hv, hv_value, condition, patch_value)

        entry = {
            "id": trace_id,
            HIDDEN_VAR_KEY: hv_value,
            "expected_real": expected_real,
            "condition": condition,
            "group": CONDITION_TO_GROUP.get(condition, "unknown"),
            "patch_value": patch_value,
            "expected_patch": expected_patch,
        }

        if modified_text is None:
            entry["success"] = False
            entry["error"] = "build_failed"
            condition_results[condition].append(entry)
            print(f"    [{condition}] SKIP build_failed")
            continue

        entry["modified_text"] = modified_text

        try:
            final_text = continue_generation(modified_text)
            new_text = final_text[len(modified_text):]
            pred = extract_final_answer(new_text)

            entry["success"] = True
            entry["predicted"] = pred
            entry["new_text"] = new_text
            entry["correct_real"] = is_correct(pred, expected_real)
            entry["correct_patch"] = is_correct(pred, expected_patch)
            entry["moved_toward_patch"] = moved_toward_patched(pred, expected_real, expected_patch)
            entry["patch_effect"] = patch_effect_size(pred, expected_real, expected_patch)

            status = "R" if entry["correct_real"] else "-"
            status += "P" if entry["correct_patch"] else "-"
            mtp = entry["moved_toward_patch"]
            mtp_str = "T" if mtp else ("F" if mtp is False else "NA")
            print(f"    [{condition}] {status} move={mtp_str} pred={pred} real={expected_real} patch={expected_patch}")

        except Exception as e:
            entry["success"] = False
            entry["error"] = str(e)
            print(f"    [{condition}] ERROR {e}")

        condition_results[condition].append(entry)

    if (idx + 1) % 25 == 0:
        checkpoint_file = OUTPUT_DIR / f"interventions_unified_{EXPERIMENT}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(condition_results, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_file}")


# ==========================================
# SAVE JSON
# ==========================================

output_file = OUTPUT_DIR / f"interventions_unified_{EXPERIMENT}_results.json"
with open(output_file, "w") as f:
    json.dump(condition_results, f, indent=2)
print()
print(f"Results saved to {output_file}")


# ==========================================
# SUMMARY PRINT + CSV
# ==========================================

def summarize_condition(rows: List[dict]) -> Dict[str, float]:
    succ = [r for r in rows if r.get("success")]
    real_ok = [r for r in succ if r.get("correct_real")]
    patch_ok = [r for r in succ if r.get("correct_patch")]
    moved = [r for r in succ if r.get("moved_toward_patch") is not None]
    moved_true = [r for r in moved if r.get("moved_toward_patch")]
    effect_vals = [r["patch_effect"] for r in succ if r.get("patch_effect") is not None]

    return {
        "success_n": len(succ),
        "real_acc": (len(real_ok) / len(succ)) if succ else float("nan"),
        "patch_acc": (len(patch_ok) / len(succ)) if succ else float("nan"),
        "moved_rate": (len(moved_true) / len(moved)) if moved else float("nan"),
        "effect_mean": (sum(effect_vals) / len(effect_vals)) if effect_vals else float("nan"),
    }


print("\n" + "=" * 80)
print(f"UNIFIED SUMMARY - {EXPERIMENT.upper()}")
print("=" * 80)
print(f"unmod_cot baseline accuracy: {cot_acc:.1%} ({len(cot_corr)}/{len(cot_success)})")
print()

summary_rows: List[Tuple[str, str, Dict[str, float]]] = []
for block_name in ["censoring_masking", "value_patching", "truncation", "baseline"]:
    block_conditions = [c for c in ALL_CONDITIONS if CONDITION_TO_GROUP.get(c) == block_name]
    if not block_conditions:
        continue

    print(f"[{block_name}]")
    for cond in block_conditions:
        metrics = summarize_condition(condition_results[cond])
        summary_rows.append((block_name, cond, metrics))
        print(
            f"  {cond:<18} "
            f"success={metrics['success_n']:>3d} "
            f"real_acc={metrics['real_acc']:.1%} "
            f"patch_acc={metrics['patch_acc']:.1%} "
            f"move_toward_patch={metrics['moved_rate']:.1%} "
            f"effect={metrics['effect_mean']:.3f}"
        )
    print()

csv_file = OUTPUT_DIR / f"interventions_unified_{EXPERIMENT}_summary.csv"
with open(csv_file, "w") as f:
    f.write(
        "block,condition,success_n,real_accuracy,patch_accuracy,moved_toward_patch_rate,mean_patch_effect,description\n"
    )
    f.write(
        f"baseline,unmod_cot,{len(cot_success)},{cot_acc:.6f},,,," +
        "Unmodified CoT baseline\n"
    )
    for block_name, cond, m in summary_rows:
        f.write(
            f"{block_name},{cond},{m['success_n']},{m['real_acc']:.6f},{m['patch_acc']:.6f},"
            f"{m['moved_rate']:.6f},{m['effect_mean']:.6f},{CONDITION_DESCRIPTIONS[cond]}\n"
        )

print(f"Summary CSV saved to {csv_file}")
