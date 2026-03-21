# Simple Causal Tracing via Activation Patching

## Quick Start

Run the script with target token indices:

```bash
cd /home/wuroderi/links/projects/def-rgrosse/wuroderi/reasoning_abstraction
python intervene_graph.py --target-indices 120 132 145
```

## Input Paths

| Input | Path |
|-------|------|
| **Traces Metadata** | `/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/traces_metadata.json` |
| **Model** | `/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B` |

## Output Paths

All outputs go to: `/home/wuroderi/links/scratch/causal_graph_plots/`

| Output Type | Pattern | Example |
|------------|---------|---------|
| **Heatmap PNG** | `causal_heatmap_pair_NNN_target_MMM.png` | `causal_heatmap_pair_001_target_120.png` |
| **Data (NPZ)** | `causal_data_pair_NNN_target_MMM.npz` | `causal_data_pair_001_target_120.npz` |
| **Summary JSON** | `causal_graph_summary.json` | — |

## Usage Examples

### Basic: Single pair, single target
```bash
python intervene_graph.py --target-indices 120
```

### Multiple targets on first pair
```bash
python intervene_graph.py --target-indices 100 120 150 200
```

### Different layers (default: every 4th layer)
```bash
python intervene_graph.py --target-indices 120 --layers 0 5 10 15 20
```

### Multiple pairs starting from pair index 2
```bash
python intervene_graph.py --pair-index 2 --n-pairs 3 --target-indices 120 150
```

### Skip heatmap plots (save only NPZ data)
```bash
python intervene_graph.py --target-indices 120 --no-plot
```

## CLI Arguments

```
--target-indices INDICES     Token indices to analyze (required). Example: 120 132 145
--pair-index N               Which pair to start at (default: 0)
--n-pairs N                  How many pairs to process (default: 1)
--layers LAYER_LIST          Layer indices to test. Example: 0 5 10 15 20 30
                             Default: every 4th layer 0-63
--no-plot                    Skip PNG generation (still save .npz)
--help                       Show all options
```

## Workflow

1. **Identify targets**: Use `inspect_trace_tokens.py` to find token indices of interest:
   ```bash
   python inspect_trace_tokens.py --pair-index 0 --truncated-only --show 50
   ```

2. **Run experiment**: Provide target indices and run patching:
   ```bash
   python intervene_graph.py --target-indices <IDX1> <IDX2> <IDX3>
   ```

3. **Inspect outputs**:
   - Open `.png` files in viewer to see heatmaps
   - Load `.npz` files to examine raw causal effect scores
   - Check `causal_graph_summary.json` for metadata

4. **Iterate**: Based on heatmaps, choose new target indices and repeat step 2.

## Output File Structure

### Heatmap PNG
- X-axis: tokens (truncated before target)
- Y-axis: layers tested
- Color: log P(source) - log P(base)
  - Red = favors source
  - Blue = favors base

### NPZ Data
Contains:
- `heatmap`: (n_layers, n_prefix_tokens) array of effects
- `layers`: tested layer indices
- `prefix_tokens`: token strings before target
- `target_index`: token index being targeted
- `target_token`: string of target token
- `source_velocity`, `base_velocity`: trace velocities

### Summary JSON
Per-pair metadata:
- `pair_idx`: pair number
- `source_trace_id`, `base_trace_id`: trace IDs
- `target_results`: list of targets analyzed + output file paths

## Tips

- Start with `--pair-index 0 --n-pairs 1` to test on one pair first
- Use smaller `--layers` list (e.g., `--layers 0 10 20 30 40 50 60`) for faster debugging
- The `--target-indices` indices refer to positions in the **full tokenized base text** (not truncated)
