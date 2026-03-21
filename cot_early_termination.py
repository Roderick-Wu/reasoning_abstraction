"""
Early CoT Truncation Experiment

For a chosen experiment, this script keeps a base CoT wording fixed and
substitutes numeric values from other traces to create variations.

Then it reveals the CoT one space-separated word at a time:
  input_k = [prompt] + [first k CoT words] + " The final answer is "
and measures answer accuracy at each truncation depth k.

This version supports:
1) all 8 hidden-variable experiments,
2) all-format or same-format candidate sampling,
3) multiple base traces in one run (e.g., first trace per format_id), and
4) per-base outputs plus an aggregate summary CSV.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPERIMENT_CONFIGS: Dict[str, Dict[str, str]] = {
    "velocity": {
        "hidden_var_key": "v",
        "hidden_symbol": "v",
        "answer_key": "expected_time",
        "final_symbol": "t = ",
    },
    "current": {
        "hidden_var_key": "i",
        "hidden_symbol": "i",
        "answer_key": "expected_charge",
        "final_symbol": "Q = ",
    },
    "radius": {
        "hidden_var_key": "r",
        "hidden_symbol": "r",
        "answer_key": "expected_circumference",
        "final_symbol": "C = ",
    },
    "side_length": {
        "hidden_var_key": "s",
        "hidden_symbol": "s",
        "answer_key": "expected_surface_area",
        "final_symbol": "SA = ",
    },
    "wavelength": {
        "hidden_var_key": "wavelength",
        "hidden_symbol": "wavelength",
        "answer_key": "expected_distance",
        "final_symbol": "d = ",
    },
    "cross_section": {
        "hidden_var_key": "area",
        "hidden_symbol": "area",
        "answer_key": "expected_volume",
        "final_symbol": "V = ",
    },
    "displacement": {
        "hidden_var_key": "x",
        "hidden_symbol": "x",
        "answer_key": "expected_pe",
        "final_symbol": "PE = ",
    },
    "market_cap": {
        "hidden_var_key": "market_cap",
        "hidden_symbol": "market_cap",
        "answer_key": "expected_pe",
        "final_symbol": "P/E = ",
    },
}

NUM_RE = r"([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)"
SEP = "Answer (step-by-step): "
APPEND_SUFFIX = " ... The final answer is "


def get_clean_generation(trace: dict) -> str:
    """Full generated text up to any hallucinated follow-on question."""
    gen = trace["generated_text"]
    stop = gen.find("Question", 10)
    return gen[:stop].strip() if stop != -1 else gen.strip()


def strip_final_answer_sentence(cot: str) -> str:
    """Remove trailing answer-reveal sentence to keep only reasoning steps."""
    m = re.search(r"\bThe answer is\b", cot, re.IGNORECASE)
    return cot[:m.start()].strip() if m else cot.strip()


def find_all_occurrences(text: str, value: float, tolerance: float = 1.0) -> List[Tuple[int, int, str]]:
    """Return all non-overlapping occurrences of value in text as (start, end, literal)."""
    literal_1f = re.escape(f"{value:.1f}")
    literal_2f = re.escape(f"{value:.2f}")
    literal_4f = re.escape(f"{value:.4f}")
    literal_3e = re.escape(f"{value:.3e}")

    patterns = [
        rf"\b{int(value)}\b(?!\.\d)",
        rf"\b{int(value)}\.0+\b",
        rf"\b{literal_1f}\b",
        rf"\b{literal_2f}\b",
        rf"\b{literal_4f}\b",
        rf"\b{literal_3e}\b",
    ]
    seen: List[Tuple[int, int, str]] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            if not any(s <= m.start() < e for s, e, _ in seen):
                seen.append((m.start(), m.end(), m.group()))

    if not seen:
        for m in re.finditer(r"\b(\d+\.?\d*(?:[eE][+-]?\d+)?)\b", text):
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


def count_decimals_in_literal(literal: str) -> int:
    if "e" in literal.lower():
        mantissa = literal.lower().split("e", 1)[0]
        if "." in mantissa:
            return len(mantissa.split(".", 1)[1])
        return 0
    if "." in literal:
        return len(literal.split(".", 1)[1])
    return 0


def format_like_literal(value: float, template_literal: str) -> str:
    """Format value to mirror template style (integer/fixed/scientific)."""
    decimals = count_decimals_in_literal(template_literal)
    if "e" in template_literal.lower():
        return f"{value:.{decimals}e}"
    if "." in template_literal:
        return f"{value:.{decimals}f}"
    return str(int(round(value)))


def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_assignment_literal(text: str, symbol: str) -> Optional[Tuple[int, int, str]]:
    """Find last numeric literal in patterns like '<symbol> = <number>'."""
    if not symbol:
        return None

    escaped_symbol = re.escape(symbol.strip())
    pattern = rf"(?<![A-Za-z0-9_]){escaped_symbol}(?![A-Za-z0-9_])\s*=\s*{NUM_RE}"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return None
    m = matches[-1]
    return (m.start(1), m.end(1), m.group(1))


def find_hidden_literal(
    cot_steps: str,
    cot_full: str,
    hidden_value: float,
    hidden_var_key: str,
    hidden_symbol: str,
) -> Optional[Tuple[int, int, str]]:
    """Locate hidden-value literal, preferring explicit variable assignment forms."""
    symbol_candidates = []
    for candidate in [hidden_symbol, hidden_var_key]:
        if candidate and candidate not in symbol_candidates:
            symbol_candidates.append(candidate)

    # Prefer explicit assignments in the reasoning-only region.
    for sym in symbol_candidates:
        occ = find_assignment_literal(cot_steps, sym)
        if occ is not None:
            return occ

    # Fallback to full CoT if the answer sentence contains the only assignment.
    for sym in symbol_candidates:
        occ = find_assignment_literal(cot_full, sym)
        if occ is not None:
            return occ

    # Last-resort numeric matching.
    occ = find_last_occurrence(cot_steps, hidden_value)
    if occ is not None:
        return occ
    return find_last_occurrence(cot_full, hidden_value)


def extract_final_answer(text: str, final_symbol: str) -> Optional[float]:
    """Extract numeric answer from continuation with robust fallback patterns."""
    # Ignore hallucinated follow-on tasks that often begin with another Question.
    q_idx = re.search(r"\bQuestion\s*:", text, re.IGNORECASE)
    if q_idx:
        text = text[:q_idx.start()]

    sym = re.escape(final_symbol.strip()) if final_symbol.strip() else ""
    patterns = [
        rf"{sym}\s*{NUM_RE}" if sym else None,
        rf"(?:final\s+answer|answer)\s+is\s+{NUM_RE}",
        rf"^\s*{NUM_RE}(?:\s|\.|$)",
        rf"(?:result|therefore)\D+{NUM_RE}",
        rf"=\s*{NUM_RE}\s*$",
        NUM_RE,
    ]
    for pattern in patterns:
        if pattern is None:
            continue
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if not matches:
            continue
        try:
            # Prefer the earliest extraction in truncated text.
            return float(matches[0].group(1))
        except (ValueError, IndexError):
            continue
    return None


def is_correct(predicted: Optional[float], expected: float, rel_tol: float, abs_tol: float = 0.0) -> bool:
    if predicted is None:
        return False
    return abs(predicted - expected) <= max(abs_tol, rel_tol * abs(expected))


def batched_generate(
    model,
    tokenizer,
    input_texts: List[str],
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    """Run batched greedy decoding and return only newly generated continuations."""
    import torch

    all_outputs: List[str] = []
    try:
        input_device = next(iter(model.parameters())).device
    except StopIteration:
        input_device = torch.device("cpu")

    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(input_device)
        n_input = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq in out:
            new_text = tokenizer.decode(seq[n_input:], skip_special_tokens=True)
            all_outputs.append(new_text)

    return all_outputs


def resolve_base_trace_indices(args, all_traces: List[dict]) -> List[int]:
    """Resolve which base trace indices to run."""
    n = len(all_traces)

    if args.base_trace_indices:
        resolved = sorted(set(args.base_trace_indices))
        for idx in resolved:
            if idx < 0 or idx >= n:
                raise ValueError(f"base_trace_indices contains out-of-range index: {idx}")
        return resolved

    if args.first_trace_per_format:
        # Choose first trace index encountered for each format_id.
        seen_formats = set()
        resolved = []
        for idx, trace in enumerate(all_traces):
            fmt = trace.get("format_id")
            if fmt not in seen_formats:
                seen_formats.add(fmt)
                resolved.append(idx)
        return resolved

    if args.base_trace_idx < 0 or args.base_trace_idx >= n:
        raise ValueError(f"base_trace_idx out of range: {args.base_trace_idx}")
    return [args.base_trace_idx]


def trace_has_hidden_and_answer_literals(
    trace: dict,
    hidden_var_key: str,
    hidden_symbol: str,
    answer_key: str,
) -> bool:
    """Check if a trace's CoT text contains parseable hidden and answer literals."""
    hidden = safe_float(trace.get(hidden_var_key))
    answer = safe_float(trace.get(answer_key))
    if hidden is None or answer is None:
        return False

    full_text = get_clean_generation(trace)
    sep_pos = full_text.find(SEP)
    if sep_pos == -1:
        return False
    cot = full_text[sep_pos + len(SEP):]
    cot_steps = strip_final_answer_sentence(cot)

    hidden_loc = find_hidden_literal(
        cot_steps=cot_steps,
        cot_full=cot,
        hidden_value=hidden,
        hidden_var_key=hidden_var_key,
        hidden_symbol=hidden_symbol,
    )
    answer_loc = find_last_occurrence(cot_steps, answer)
    if answer_loc is None:
        answer_loc = find_last_occurrence(cot, answer)
    return hidden_loc is not None and answer_loc is not None


def find_fallback_base_index(
    all_traces: List[dict],
    requested_base_idx: int,
    hidden_var_key: str,
    hidden_symbol: str,
    answer_key: str,
) -> Optional[int]:
    """Find a usable base index, preferring same format_id as requested base."""
    base_fmt = all_traces[requested_base_idx].get("format_id")

    same_fmt = [
        i for i, t in enumerate(all_traces)
        if t.get("format_id") == base_fmt and i != requested_base_idx
    ]
    any_fmt = [i for i in range(len(all_traces)) if i != requested_base_idx]

    for i in [requested_base_idx] + same_fmt + any_fmt:
        if trace_has_hidden_and_answer_literals(all_traces[i], hidden_var_key, hidden_symbol, answer_key):
            return i
    return None


def run_for_base_trace(
    all_traces: List[dict],
    base_idx: int,
    hidden_var_key: str,
    hidden_symbol: str,
    answer_key: str,
    final_symbol: str,
    args,
    model,
    tokenizer,
    output_dir: Path,
) -> Dict:
    """Run full truncation experiment for one base trace index and save outputs."""
    base_trace = all_traces[base_idx]
    base_answer = safe_float(base_trace.get(answer_key))
    base_hidden = safe_float(base_trace.get(hidden_var_key))
    if base_answer is None or base_hidden is None:
        raise RuntimeError(
            f"Base trace {base_idx} missing numeric hidden/answer fields: {hidden_var_key}, {answer_key}"
        )

    full_text_0 = get_clean_generation(base_trace)
    sep_pos = full_text_0.find(SEP)
    if sep_pos == -1:
        raise RuntimeError(f"Base trace {base_idx} text does not contain '{SEP}'")
    sep_idx = sep_pos + len(SEP)
    prompt_0 = full_text_0[:sep_idx]
    cot_0 = full_text_0[sep_idx:]
    cot_steps_0 = strip_final_answer_sentence(cot_0)
    cot_words_0 = cot_steps_0.split()

    hidden_loc = find_hidden_literal(
        cot_steps=cot_steps_0,
        cot_full=cot_0,
        hidden_value=base_hidden,
        hidden_var_key=hidden_var_key,
        hidden_symbol=hidden_symbol,
    )
    answer_loc = find_last_occurrence(cot_steps_0, base_answer)
    if answer_loc is None:
        answer_loc = find_last_occurrence(cot_0, base_answer)
    if hidden_loc is None or answer_loc is None:
        raise RuntimeError(f"Base trace {base_idx}: could not find hidden/answer literals in CoT")

    hidden_literal_0 = hidden_loc[2]
    answer_literal_0 = answer_loc[2]

    print("-" * 70)
    print(f"BASE TRACE idx={base_idx} id={base_trace['id']} format_id={base_trace.get('format_id')}")
    print(f"  {hidden_var_key}={base_hidden}  {answer_key}={base_answer}")
    print(f"  hidden literal={hidden_literal_0!r}  answer literal={answer_literal_0!r}")

    exclude_numeric_keys = {
        "id", "format_id", "prompt_length", "tokens", "token_strings",
        answer_key, hidden_var_key,
    }

    feature_keys = [
        k for k, v in base_trace.items()
        if isinstance(v, (int, float)) and k not in exclude_numeric_keys
    ]

    feature_literals: Dict[str, str] = {}
    for key in feature_keys:
        v0 = safe_float(base_trace.get(key))
        if v0 is None:
            continue
        occ = find_all_occurrences(prompt_0, v0)
        if occ:
            feature_literals[key] = occ[0][2]

    def build_variation(trace_j: dict) -> Tuple[Optional[dict], str]:
        expected = safe_float(trace_j.get(answer_key))
        hidden = safe_float(trace_j.get(hidden_var_key))
        if expected is None:
            return None, "missing_answer"
        if hidden is None:
            return None, "missing_hidden"

        # Substitute over full text, but protect hidden/answer literals with placeholders
        # so feature replacements cannot accidentally overwrite them.
        protected_hidden = "__HIDDEN_LITERAL__"
        protected_answer = "__ANSWER_LITERAL__"
        protected_full_text = full_text_0
        protected_full_text = protected_full_text.replace(hidden_literal_0, protected_hidden)
        protected_full_text = protected_full_text.replace(answer_literal_0, protected_answer)

        # Replace question-side numeric features using base prompt literals.
        for key in feature_keys:
            if key not in feature_literals:
                continue
            target_val = safe_float(trace_j.get(key))
            if target_val is None:
                return None, f"missing_feature_{key}"
            base_literal = feature_literals[key]
            target_literal = format_like_literal(target_val, base_literal)
            protected_full_text = protected_full_text.replace(base_literal, target_literal)

        # Replace hidden and final-answer values in CoT.
        hidden_literal = format_like_literal(hidden, hidden_literal_0)
        answer_literal = format_like_literal(expected, answer_literal_0)
        text = protected_full_text.replace(protected_hidden, hidden_literal)
        text = text.replace(protected_answer, answer_literal)

        loc = text.find(SEP)
        if loc == -1:
            return None, "sep_missing_after_substitution"
        split = loc + len(SEP)
        prompt_part = text[:split]
        cot_part = text[split:]

        if hidden_literal not in cot_part:
            return None, "hidden_not_inserted"
        if answer_literal not in cot_part:
            return None, "answer_not_inserted"

        cot_steps = strip_final_answer_sentence(cot_part)
        cot_words = cot_steps.split()
        if not cot_words:
            return None, "empty_cot"

        return {
            "trace_id": trace_j["id"],
            "format_id": trace_j.get("format_id"),
            "prompt": prompt_part,
            "cot_words": cot_words,
            "expected": expected,
            "hidden": hidden,
        }, "ok"

    if args.match_format_only:
        candidate_traces = [
            t for t in all_traces
            if t.get("format_id") == base_trace.get("format_id") and t["id"] != base_trace["id"]
        ]
    else:
        candidate_traces = [t for t in all_traces if t["id"] != base_trace["id"]]

    candidate_format_counts = Counter(t.get("format_id") for t in candidate_traces)
    failure_reasons = Counter()
    variations: List[dict] = []

    for trace_j in candidate_traces:
        var, reason = build_variation(trace_j)
        if var is not None:
            variations.append(var)
        else:
            failure_reasons[reason] += 1
        if len(variations) >= args.n_variations:
            break

    if not variations:
        raise RuntimeError(f"Base trace {base_idx}: no valid variations were built")

    print("Variation build diagnostics")
    print(f"  candidates considered : {len(candidate_traces)}")
    print(f"  successful variations : {len(variations)}")
    print(f"  candidate formats     : {dict(candidate_format_counts)}")
    print(f"  failure reasons       : {dict(failure_reasons)}")
    print(
        "  CoT word count range  : "
        f"{min(len(v['cot_words']) for v in variations)}..{max(len(v['cot_words']) for v in variations)}"
    )

    if args.dry_run:
        return {
            "base_trace_idx": base_idx,
            "base_trace_id": base_trace["id"],
            "format_id": base_trace.get("format_id"),
            "n_variations": len(variations),
            "max_k": max(len(v["cot_words"]) for v in variations),
            "candidate_count": len(candidate_traces),
            "failure_reasons": dict(failure_reasons),
            "summary_rows": [],
        }

    max_k = max(len(v["cot_words"]) for v in variations)
    print(f"Running truncation experiment over k = 0 .. {max_k}")
    print(f"Total inference calls: {(max_k + 1) * len(variations)}")

    all_results: Dict[int, List[dict]] = {}

    for k in range(max_k + 1):
        input_texts: List[str] = []
        for var in variations:
            partial = " ".join(var["cot_words"][:k])
            if partial:
                input_text = var["prompt"] + partial + APPEND_SUFFIX
            else:
                input_text = var["prompt"] + APPEND_SUFFIX.lstrip()
            input_texts.append(input_text)

        continuations = batched_generate(
            model=model,
            tokenizer=tokenizer,
            input_texts=input_texts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

        k_results: List[dict] = []
        for var, input_text, continuation in zip(variations, input_texts, continuations):
            predicted = extract_final_answer(continuation, final_symbol)
            correct = is_correct(
                predicted,
                var["expected"],
                args.relative_tolerance,
                args.absolute_tolerance,
            )
            k_results.append({
                "base_trace_idx": base_idx,
                "k": k,
                "trace_id": var["trace_id"],
                "format_id": var["format_id"],
                "hidden": var["hidden"],
                "predicted": predicted,
                "expected": var["expected"],
                "correct": correct,
                "input_prompt": input_text,
                "response_text": continuation,
                # Kept for backward compatibility with previous result schema.
                "continuation": continuation,
            })

        all_results[k] = k_results

        n_correct = sum(r["correct"] for r in k_results)
        acc = n_correct / len(k_results) if k_results else float("nan")
        last_word = cot_words_0[k - 1] if (0 < k <= len(cot_words_0)) else "(none)"
        print(f"  k={k:3d}  added={last_word:<15}  acc={acc:.1%} ({n_correct}/{len(k_results)})")

    out_stub = f"cot_truncation_{args.experiment}_trace{base_idx}"
    if args.match_format_only:
        out_stub += "_samefmt"
    else:
        out_stub += "_allfmt"

    json_path = output_dir / f"{out_stub}_results.json"
    json_payload = {
        "metadata": {
            "experiment": args.experiment,
            "base_trace_idx": base_idx,
            "base_trace_id": base_trace["id"],
            "base_format_id": base_trace.get("format_id"),
            "n_variations": len(variations),
            "match_format_only": args.match_format_only,
            "max_new_tokens": args.max_new_tokens,
            "relative_tolerance": args.relative_tolerance,
            "absolute_tolerance": args.absolute_tolerance,
        },
        "results_by_k": {str(k): v for k, v in all_results.items()},
    }
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2, default=str)
    print(f"Results JSON : {json_path}")

    summary_rows: List[Dict] = []
    csv_path = output_dir / f"{out_stub}_summary.csv"
    with open(csv_path, "w") as f:
        f.write("k,word_just_added,accuracy,n_correct,n_total\n")
        for k in range(max_k + 1):
            res = all_results[k]
            n_correct = sum(r["correct"] for r in res)
            acc = n_correct / len(res) if res else float("nan")
            last_word = cot_words_0[k - 1] if (0 < k <= len(cot_words_0)) else "(none)"
            f.write(f"{k},{last_word},{acc:.4f},{n_correct},{len(res)}\n")
            summary_rows.append(
                {
                    "base_trace_idx": base_idx,
                    "k": k,
                    "word_just_added": last_word,
                    "accuracy": acc,
                    "n_correct": n_correct,
                    "n_total": len(res),
                }
            )
    print(f"Summary CSV  : {csv_path}")

    ks = list(range(max_k + 1))
    accs = [
        sum(r["correct"] for r in all_results[k]) / len(all_results[k])
        if all_results[k] else float("nan")
        for k in ks
    ]

    # Lazy import so dry-run mode does not require matplotlib/numpy.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(14, max_k // 2), 6))
    ax.plot(ks, accs, "o-", linewidth=2, markersize=5, color="steelblue")

    skip_labels = {"=", "/", "the", "is", "answer"}
    for k in ks:
        if 0 < k <= len(cot_words_0):
            word = cot_words_0[k - 1]
            if word.lower() not in skip_labels:
                ax.axvline(k, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.annotate(
                    word,
                    xy=(k, accs[k]),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    rotation=45,
                    color="dimgray",
                )

    ax.set_xlabel("CoT words revealed before truncation (k)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        f"Accuracy vs. Early CoT Truncation\\n"
        f"Experiment: {args.experiment} | Base trace {base_idx} "
        f"| {len(variations)} variations | +/-{args.relative_tolerance:.0%} tolerance",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, max_k + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    png_path = output_dir / f"{out_stub}_accuracy.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot PNG     : {png_path}")

    print("=" * 70)
    print(f"EARLY COT TRUNCATION - {args.experiment.upper()} - BASE {base_idx} SUMMARY")
    print("=" * 70)
    print(f"n_variations = {len(variations)}")
    print(f"k=0 (no CoT) : acc={accs[0]:.1%}")
    for k in ks[1:]:
        last_word = cot_words_0[k - 1] if k <= len(cot_words_0) else f"word_{k}"
        print(f"k={k:<3}  ({last_word:<15}) : acc={accs[k]:.1%}")
    print()

    return {
        "base_trace_idx": base_idx,
        "base_trace_id": base_trace["id"],
        "format_id": base_trace.get("format_id"),
        "n_variations": len(variations),
        "max_k": max_k,
        "candidate_count": len(candidate_traces),
        "failure_reasons": dict(failure_reasons),
        "summary_rows": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Early CoT truncation: accuracy vs. number of CoT words revealed"
    )
    parser.add_argument(
        "--model_path",
        default="/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B",
    )
    parser.add_argument(
        "--traces_root",
        default=os.path.expanduser("~/links/scratch/reasoning_traces/Qwen2.5-32B"),
    )
    parser.add_argument("--experiment", default="velocity", choices=list(EXPERIMENT_CONFIGS))
    parser.add_argument(
        "--base_trace_idx",
        type=int,
        default=0,
        help="Single base trace index used when multi-base options are not enabled",
    )
    parser.add_argument(
        "--base_trace_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of base trace indices to run (overrides --base_trace_idx)",
    )
    parser.add_argument(
        "--first_trace_per_format",
        action="store_true",
        help="Run one base trace per format_id (typically indices 0, 40, 80, 120, 160)",
    )
    parser.add_argument(
        "--n_variations",
        type=int,
        default=50,
        help="Number of substituted variations per base trace",
    )
    parser.add_argument(
        "--match_format_only",
        action="store_true",
        help="Restrict candidate variations to same format_id as each base trace",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--relative_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--absolute_tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for answer correctness (combined with relative tolerance)",
    )
    parser.add_argument(
        "--no_auto_fallback_base",
        action="store_true",
        help="Disable automatic fallback when selected base trace is unparsable",
    )
    parser.add_argument("--dry_run", action="store_true", help="Build diagnostics only; skip model inference")
    parser.add_argument(
        "--output_dir",
        default="/home/wuroderi/links/projects/def-rgrosse/wuroderi/reasoning_abstraction/intervention_token_results",
    )
    args = parser.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    hidden_var_key = cfg["hidden_var_key"]
    hidden_symbol = cfg.get("hidden_symbol", hidden_var_key)
    answer_key = cfg["answer_key"]
    final_symbol = cfg["final_symbol"]

    traces_dir = Path(args.traces_root) / args.experiment
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(traces_dir / "traces_metadata.json") as f:
        all_traces = json.load(f)

    base_indices = resolve_base_trace_indices(args, all_traces)

    print("=" * 70)
    print("EARLY COT TRUNCATION EXPERIMENT")
    print("=" * 70)
    print(f"Experiment         : {args.experiment}")
    print(f"Hidden var key     : {hidden_var_key}")
    print(f"Answer key         : {answer_key}")
    print(f"Hidden symbol      : {hidden_symbol}")
    print(f"Base trace indices : {base_indices}")
    print(f"Requested variants : {args.n_variations}")
    print(f"Match format only  : {args.match_format_only}")
    print(f"Dry run            : {args.dry_run}")
    print(f"Relative tolerance : {args.relative_tolerance}")
    print(f"Absolute tolerance : {args.absolute_tolerance}")
    print(f"Model              : {args.model_path}")
    print(f"Output             : {output_dir}")
    print()

    model = None
    tokenizer = None
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Model loaded: {model.config.num_hidden_layers} layers")
        print()

    all_base_summaries: List[Dict] = []
    aggregate_rows: List[Dict] = []

    for requested_base_idx in base_indices:
        base_idx = requested_base_idx
        if not trace_has_hidden_and_answer_literals(
            all_traces[base_idx],
            hidden_var_key,
            hidden_symbol,
            answer_key,
        ):
            if args.no_auto_fallback_base:
                raise RuntimeError(
                    f"Base trace {base_idx} is unparsable and --no_auto_fallback_base is set"
                )
            fallback = find_fallback_base_index(
                all_traces=all_traces,
                requested_base_idx=base_idx,
                hidden_var_key=hidden_var_key,
                hidden_symbol=hidden_symbol,
                answer_key=answer_key,
            )
            if fallback is None:
                raise RuntimeError(
                    f"No usable base trace found for requested base {base_idx}"
                )
            print(
                f"Requested base trace {requested_base_idx} is unparsable; "
                f"falling back to base trace {fallback}."
            )
            base_idx = fallback

        summary = run_for_base_trace(
            all_traces=all_traces,
            base_idx=base_idx,
            hidden_var_key=hidden_var_key,
            hidden_symbol=hidden_symbol,
            answer_key=answer_key,
            final_symbol=final_symbol,
            args=args,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )
        summary["requested_base_trace_idx"] = requested_base_idx
        all_base_summaries.append(summary)
        aggregate_rows.extend(summary["summary_rows"])

    aggregate_stub = f"cot_truncation_{args.experiment}_aggregate"
    if args.match_format_only:
        aggregate_stub += "_samefmt"
    else:
        aggregate_stub += "_allfmt"

    diag_path = output_dir / f"{aggregate_stub}_base_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(all_base_summaries, f, indent=2)
    print(f"Saved base diagnostics JSON: {diag_path}")

    if aggregate_rows:
        agg_csv = output_dir / f"{aggregate_stub}_summary.csv"
        with open(agg_csv, "w") as f:
            f.write("base_trace_idx,k,word_just_added,accuracy,n_correct,n_total\n")
            for row in aggregate_rows:
                f.write(
                    f"{row['base_trace_idx']},{row['k']},{row['word_just_added']},"
                    f"{row['accuracy']:.4f},{row['n_correct']},{row['n_total']}\n"
                )
        print(f"Saved aggregate CSV: {agg_csv}")

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Processed base traces: {base_indices}")
    print(f"Total base runs: {len(base_indices)}")
    print()


if __name__ == "__main__":
    main()
