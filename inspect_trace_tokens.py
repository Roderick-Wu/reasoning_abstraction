"""
Quick token inspection tool for saved prompt pairs.

Usage examples:
  python inspect_trace_tokens.py --pair-index 0
  python inspect_trace_tokens.py --pair-index 3 --show 220 --truncated-only
    python inspect_trace_tokens.py --trace-id 12345 --search "v\^2|e\+" --from-metadata
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import AutoTokenizer

MODEL_PATH = "/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B"
TRACES_METADATA_FILE = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/traces_metadata.json")
DEFAULT_PAIRS_JSON = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/prompt_pairs.json")


def format_value_like_reference(reference: str, value: float) -> str:
    if "e" in reference.lower():
        mantissa, _exp = re.split(r"[eE]", reference)
        decimal_places = len(mantissa.split(".")[1]) if "." in mantissa else 0
        formatted = f"{value:.{decimal_places}e}"
        return formatted.upper() if "E" in reference else formatted
    if "." in reference:
        decimal_places = len(reference.split(".")[1])
        return f"{value:.{decimal_places}f}"
    return f"{value:.0f}"


def find_value_occurrence_in_text(text: str, value: float) -> Optional[Tuple[int, int, str]]:
    numeric_pattern = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")
    for match in numeric_pattern.finditer(text):
        token = match.group()
        if format_value_like_reference(token, value) == token:
            return match.start(), match.end(), token
    return None


def find_velocity_in_text(text: str, velocity: float) -> Optional[int]:
    velocity_strs = [str(int(velocity)), f"{velocity:.1f}", f"{velocity:.2f}"]
    for vel_str in velocity_strs:
        pattern = rf"\b{re.escape(vel_str[0])}"
        match = re.search(pattern, text)
        if match:
            if text[match.start() :].startswith(vel_str):
                return match.start()
    return None


def build_prompt_pairs(traces: List[dict]) -> List[dict]:
    traces_by_format = defaultdict(list)
    for tr in traces:
        traces_by_format[tr["format_id"]].append(tr)

    min_traces_per_format = min(len(v) for v in traces_by_format.values())
    n_pairs_per_format = min_traces_per_format // 2

    pairs = []
    for fmt_id, fmt_traces in traces_by_format.items():
        for i in range(n_pairs_per_format):
            pairs.append(
                {
                    "format_id": fmt_id,
                    "source_trace": fmt_traces[i],
                    "base_trace": fmt_traces[i + n_pairs_per_format],
                }
            )
    return pairs


def normalize_pair_for_inspection(pair: dict) -> Tuple[dict, dict, int]:
    # v5 schema (current): cot_0/cot_1 with trace_0/trace_1 values
    if "values" in pair and "cot_0" in pair and "cot_1" in pair:
        trace_0 = {
            "id": pair.get("pair_id", -1),
            "format_id": pair.get("format_id", -1),
            "m": pair["values"]["trace_0"].get("m"),
            "ke": pair["values"]["trace_0"].get("ke"),
            "v": pair["values"]["trace_0"].get("v"),
            "d": pair["values"]["trace_0"].get("d"),
            "generated_text": pair["cot_0"].get("original", ""),
        }
        trace_1 = {
            "id": pair.get("pair_id", -1),
            "format_id": pair.get("format_id", -1),
            "m": pair["values"]["trace_1"].get("m"),
            "ke": pair["values"]["trace_1"].get("ke"),
            "v": pair["values"]["trace_1"].get("v"),
            "d": pair["values"]["trace_1"].get("d"),
            "generated_text": pair["cot_1"].get("original", ""),
        }
        return trace_0, trace_1, pair.get("pair_id", -1)

    # v4 schema (old): cot_source/cot_base structure
    if "values" in pair and "cot_source" in pair and "cot_base" in pair:
        source = {
            "id": pair.get("pair_id", -1),
            "format_id": pair.get("format_id", -1),
            "m": pair["values"]["source"].get("m"),
            "ke": pair["values"]["source"].get("ke"),
            "v": pair["values"]["source"].get("v"),
            "d": pair["values"]["source"].get("d"),
            "generated_text": pair["cot_source"].get("original", ""),
        }
        base = {
            "id": pair.get("pair_id", -1),
            "format_id": pair.get("format_id", -1),
            "m": pair["values"]["base"].get("m"),
            "ke": pair["values"]["base"].get("ke"),
            "v": pair["values"]["base"].get("v"),
            "d": pair["values"]["base"].get("d"),
            "generated_text": pair["cot_base"].get("original", ""),
        }
        return source, base, pair.get("pair_id", -1)

    # v3 schema: cot_raw structure
    if "values" in pair and "cot_raw" in pair:
        source = {
            "id": pair["cot_raw"]["source"].get("trace_id"),
            "format_id": pair.get("format_id", -1),
            "m": pair["values"]["source"].get("m"),
            "ke": pair["values"]["source"].get("ke"),
            "v": pair["values"]["source"].get("v"),
            "d": pair["values"]["source"].get("d"),
            "generated_text": pair["cot_raw"]["source"].get("text", ""),
        }
        base = {
            "id": pair["cot_raw"]["base"].get("trace_id"),
            "format_id": pair.get("format_id", -1),
            "m": pair["values"]["base"].get("m"),
            "ke": pair["values"]["base"].get("ke"),
            "v": pair["values"]["base"].get("v"),
            "d": pair["values"]["base"].get("d"),
            "generated_text": pair["cot_raw"]["base"].get("text", ""),
        }
        return source, base, pair.get("format_id", -1)

    # v2 schema from intervene_generate_pairs.py
    if "source" in pair and "base" in pair:
        source = {
            "id": pair["source"].get("trace_id"),
            "format_id": pair.get("format_id", pair["source"].get("format_id")),
            "m": pair["source"].get("values", {}).get("m"),
            "ke": pair["source"].get("values", {}).get("ke"),
            "v": pair["source"].get("values", {}).get("v"),
            "d": pair["source"].get("values", {}).get("d"),
            "generated_text": pair["source"].get("generated_text", ""),
        }
        base = {
            "id": pair["base"].get("trace_id"),
            "format_id": pair.get("format_id", pair["base"].get("format_id")),
            "m": pair["base"].get("values", {}).get("m"),
            "ke": pair["base"].get("values", {}).get("ke"),
            "v": pair["base"].get("values", {}).get("v"),
            "d": pair["base"].get("values", {}).get("d"),
            "generated_text": pair["base"].get("generated_text", ""),
        }
        return source, base, pair.get("format_id", -1)

    # v1 schema fallback
    if "source_trace" in pair and "base_trace" in pair:
        return pair["source_trace"], pair["base_trace"], pair.get("format_id", -1)

    raise ValueError("Unrecognized pair schema")


def print_tokens(tokenizer, text: str, max_tokens: int, search: Optional[str]) -> None:
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()

    if search:
        search_re = re.compile(search)
    else:
        search_re = None

    print(f"Token count: {len(ids)}")
    print("idx\tchar_start\tchar_end\ttoken")

    for i in range(min(len(ids), max_tokens)):
        tok = tokenizer.decode([ids[i].item()]).replace("\n", "\\n")
        c0, c1 = offsets[i]
        line = f"{i}\t{c0}\t{c1}\t{tok}"
        if search_re is not None:
            if search_re.search(tok):
                print(line + "    <MATCH>")
        else:
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Inspect token indices in trace text.")
    parser.add_argument("--pair-index", type=int, default=0, help="Pair index from grouped source/base pairing.")
    parser.add_argument("--trace-id", type=int, default=None, help="Inspect this trace id directly (overrides pair-index).")
    parser.add_argument("--pairs-json", type=Path, default=DEFAULT_PAIRS_JSON, help="Path to prompt pairs JSON.")
    parser.add_argument("--from-metadata", action="store_true", help="Build pairs directly from traces metadata instead of loading pairs JSON.")
    parser.add_argument("--show", type=int, default=200, help="How many tokens to print.")
    parser.add_argument("--search", type=str, default=None, help="Regex over decoded token text for highlighting.")
    parser.add_argument("--truncated-only", action="store_true", help="Truncate before v^2 (or velocity fallback) before printing tokens.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if args.trace_id is not None:
        with open(TRACES_METADATA_FILE, "r") as f:
            traces = json.load(f)
        selected = None
        for tr in traces:
            if int(tr["id"]) == args.trace_id:
                selected = tr
                break
        if selected is None:
            raise ValueError(f"Trace id {args.trace_id} not found")
        base_trace = selected
        print(f"Inspecting trace-id={base_trace['id']} format={base_trace['format_id']}")
    else:
        if args.from_metadata:
            with open(TRACES_METADATA_FILE, "r") as f:
                traces = json.load(f)
            pairs = build_prompt_pairs(traces)
        else:
            with open(args.pairs_json, "r") as f:
                payload = json.load(f)
            pairs = payload["pairs"]

        if args.pair_index < 0 or args.pair_index >= len(pairs):
            raise ValueError(f"pair-index out of range: {args.pair_index}, total={len(pairs)}")
        pair = pairs[args.pair_index]
        source_trace, base_trace, format_id = normalize_pair_for_inspection(pair)
        print(
            f"Inspecting pair-index={args.pair_index} format={format_id} "
            f"source_id={source_trace['id']} base_id={base_trace['id']}"
        )

    text = base_trace["generated_text"]

    if args.truncated_only:
        v2 = (2 * base_trace["ke"]) / base_trace["m"]
        match = find_value_occurrence_in_text(text, v2)
        if match is not None:
            pos, _end, ref = match
            print(f"Truncating at v^2 match: '{ref}' at char={pos}")
            text = text[:pos]
        else:
            pos = find_velocity_in_text(text, base_trace["v"])
            if pos is not None:
                print(f"v^2 not found; truncating at velocity char={pos}")
                text = text[:pos]
            else:
                print("No v^2 or velocity match found; printing full text")

    print_tokens(tokenizer, text, args.show, args.search)


if __name__ == "__main__":
    main()
