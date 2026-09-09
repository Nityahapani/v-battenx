# Model Serialization Format

> **Format version:** frozen as of v5. Forward-compatible for all v5.x releases.

## File Layout

A saved model is a single `.json` file. It contains a JSON envelope and all
field parameters inline as arrays of full-precision (17-digit) doubles.

```json
{
  "version":         "5.0.0",
  "num_stages":      100,
  "learning_rate":   0.1,
  "reg_lambda":      1.0,
  "lambda_pde":      0.0,
  "total_mutations": 12,
  "stages": [ ... ]
}
```

## Stage Object

Each element of `"stages"` holds one boosting stage:

```json
{
  "weight":      0.1,
  "pde_before":  0.0842,
  "pde_after":   0.0613,
  "params":      [1.234567890123456789, -0.987654321098765, ...],
  "mutations":   [
    {
      "type":   "expand_1d_to_2d",
      "region": 0,
      "pde_r":  0.0842
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `weight` | double | Learning rate η for this stage |
| `pde_before` | double | PDE residual before DTDO mutation |
| `pde_after` | double | PDE residual after DTDO mutation |
| `params` | double[] | Full-precision field parameters (Frobenius-norm-preserving) |
| `mutations` | object[] | Mutation events fired this stage |

## Mutation Event Object

```json
{
  "type":   "expand_1d_to_2d",
  "region": 0,
  "pde_r":  0.0842
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | One of the 10 `MutationType` names |
| `region` | int | Region ID that was mutated |
| `pde_r` | double | PDE residual at mutation time |

## Mutation Type Names

| C++ enum | JSON string |
|----------|-------------|
| `NoOp` | `"no_op"` |
| `Expand1dTo2d` | `"expand_1d_to_2d"` |
| `Expand2dTo3d` | `"expand_2d_to_3d"` |
| `Collapse3dTo2d` | `"collapse_3d_to_2d"` |
| `Collapse2dTo1d` | `"collapse_2d_to_1d"` |
| `SplitRegion` | `"split_region"` |
| `MergeRegions` | `"merge_regions"` |
| `AddConnection` | `"add_connection"` |
| `RemoveConnection` | `"remove_connection"` |
| `LocalDimChange` | `"local_dim_change"` |

## Precision Guarantee

All `params` values are written with `std::setprecision(17)` — the minimum
precision required for exact IEEE 754 float64 round-trip. Save → load → predict
produces bit-identical outputs.

## Forward Compatibility Rules

1. New fields can be added to the envelope or stage object without breaking older loaders.
2. Loaders must ignore unknown fields (they do — the JSON parser discards unrecognised keys).
3. The `"version"` field is informational only — it does not gate loading behaviour.
4. Breaking changes require a major version bump (`6.0.0`).

## API

```python
booster.save("model.json")
b2 = Booster.load("model.json")
```

```r
vbx.save(booster, "model.json")
b2 <- vbx.load("model.json")
```

```c
vbx_save(handle, "model.json");
vbx_load(handle, "model.json");
```
