# DTDO — Dynamic Topological-Dimensional Operator

The DTDO is the algorithmic core of V-BATTEN-X. It is the component that
allows the model to restructure the latent field between boosting stages —
changing dimensions, splitting and merging regions, and rewiring topology.

---

## The core operation

```
Oθ : (F, K, d, T) → (F', K', d', T')
```

At each boosting stage, before fitting residuals, the DTDO receives the current
field state and the physics residuals from the previous stage. It decides whether
and how to mutate the field structure.

---

## Mutation types

| MutationType | What changes | When it fires |
|---|---|---|
| `expand_1d_to_2d` | Embeds 1D field into 2D space | PDE residual > τ_expand in 1D region |
| `expand_2d_to_3d` | Embeds 2D field into 3D space | PDE residual > τ_expand in 2D region |
| `collapse_3d_to_2d` | Projects 3D field to 2D via PCA | PDE residual < τ_collapse or over budget |
| `collapse_2d_to_1d` | Projects 2D field to 1D via PCA | PDE residual < τ_collapse or over budget |
| `split_region` | Divides one region into two | Residual gradient magnitude > split_thresh |
| `merge_regions` | Fuses two adjacent regions | Residual gradient magnitude < merge_thresh |
| `add_connection` | Adds topology edge between regions | Connectivity increase needed |
| `remove_connection` | Removes topology edge | Connectivity reduction for pruning |
| `local_dim_change` | Changes dim of exactly one region (n→m) | General n→m via PCA |
| `no_op` | Does nothing | Budget exhausted or residuals within bounds |

---

## Dimension transitions

All dimension changes preserve the Frobenius norm of the parameter tensor:

- **Expand 1D→2D**: Appends a small-noise second axis. New axis initialised near zero, then normalised.
- **Expand 2D→3D**: Appends a third axis similarly.
- **Collapse 3D→2D**: PCA projection — keeps top-2 principal components.
- **Collapse 2D→1D**: PCA projection — keeps first principal component.
- **General n→m**: Uses PCA for any n→m transition. If m > n, zero-padding with small noise.

The Frobenius norm is restored after every transition so the boosting step size
is not disrupted by the mutation.

---

## Complexity budget

Every DTDO decision is checked against a `ComplexityCost` before firing:

```python
Booster({
    "dtdo":          "threshold",
    "max_total_dim": 64,   # hard cap on Σ local_dim across all regions
    "max_regions":   16,   # hard cap on number of regions
    "max_connections": 32, # hard cap on topology edges
})
```

`CanAfford(mutation_type)` returns false if the mutation would exceed any cap.
The `ComplexityPruner` collapses the lowest-residual regions when the model is
over budget.

---

## Rule-based DTDO strategies

### `"threshold"` (default simple strategy)

```python
Booster({"dtdo": "threshold", "tau_expand": 0.1, "tau_collapse": 0.01})
```

- If `pde_residual > tau_expand` → expand dimension of region 0
- If `pde_residual < tau_collapse` → collapse dimension of region 0
- Otherwise → `no_op`

Fast and deterministic. Good starting point.

### `"gradient"`

```python
Booster({"dtdo": "gradient", "tau_expand": 0.05, "tau_collapse": 0.005})
```

Splits regions where residual gradient magnitude is high (high complexity needed)
and merges adjacent regions where gradient is low (safely simplifiable).

### `"pruner"`

Fires a collapse every `prune_every` stages regardless of residuals.
Used when the complexity budget is the primary concern.

### `"router"` (recommended for physics problems)

Priority chain: `ComplexityPruner` (if over budget) → `GradientOperator` →
`ThresholdOperator`. Runs the most appropriate strategy automatically.

```python
Booster({
    "dtdo":          "router",
    "tau_expand":    0.1,
    "tau_collapse":  0.01,
    "max_total_dim": 32,
})
```

### `"none"` (disabled)

Default. DTDO is not called. Field structure is fixed throughout training.
Equivalent to v1/v2 behaviour.

---

## Reading the mutation log

After training and saving, the mutation log is available:

```python
b.save("model.json")
log = b.get_mutation_log()   # List[List[MutationEvent]] — one list per stage

for stage_idx, events in enumerate(log):
    for ev in events:
        print(f"stage={stage_idx}  {ev.type}  region={ev.region}  pde_r={ev.pde_r:.4f}")
```

PDE residuals per stage (before and after mutation):

```python
residuals = b.get_stage_pde_residuals()
# [{"before": 0.12, "after": 0.08}, ...]
```

---

## Tuning guide

| Scenario | Recommended setting |
|---|---|
| No physics knowledge | `dtdo="none"` |
| Noisy data, sparse physics | `dtdo="threshold"`, `tau_expand=0.05` |
| Dense data, strong physics | `dtdo="router"`, `lambda_pde=0.05` |
| Memory constrained | `max_total_dim=16`, `max_regions=4` |
| Research / ablation | `dtdo="threshold"` then `dtdo="gradient"` — compare RMSE |

---

## Model file format (v3)

The JSON model file now includes per-stage mutation logs and PDE residuals:

```json
{
  "version": "3.0.0",
  "total_mutations": 12,
  "stages": [
    {
      "weight": 0.1,
      "pde_before": 0.08,
      "pde_after": 0.06,
      "params": [...],
      "mutations": [
        {"type": "expand_1d_to_2d", "region": 0, "pde_r": 0.08}
      ]
    }
  ]
}
```
