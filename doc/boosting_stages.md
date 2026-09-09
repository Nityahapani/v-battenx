# Variational Boosting in V-BATTEN-X

## Classical Gradient Boosting vs Variational Boosting

In classical gradient boosting (XGBoost, LightGBM), each stage fits a weak learner
(a decision tree) to the negative gradient of the loss. The ensemble is additive:

```
F_t(x) = F_{t-1}(x) + η · h_t(x)
```

where `h_t` is a new tree and `η` is the learning rate.

In V-BATTEN-X, each stage can mutate the **computational field** before fitting:

```
(F, K, d, T)_t = DTDO( (F, K, d, T)_{t-1}, residuals_{t-1}, budget )
pred_t(x)      = F_t(x) + η · LinearBooster(x, g_t, h_t)
```

Each stage in the ensemble owns a snapshot of `(F, K, d, T)` at the time of boosting,
along with the mutation log that produced it.

## The Variational Interpretation

The DTDO mutation is a **variational step** over the space of field structures.
At each stage, it solves:

```
(F', K', d', T') = argmin_{structure} E[loss | structure, data]
                   subject to complexity_budget
```

This is approximated by the rule-based or learned DTDO policy.

## Stage Model

Each stage in the `VariationalEnsemble` stores:

| Field | Type | Meaning |
|-------|------|---------|
| `state` | `FieldState` | Snapshot of `(F,K,d,T)` at boost time |
| `mutation_log` | `MutationLog` | What the DTDO changed to reach this state |
| `weight` | `vbx_float` | Learning rate η for this stage |
| `pde_residual_before` | `float` | PDE residual before mutation |
| `pde_residual_after` | `float` | PDE residual after mutation |

## Shrinkage and Ensemble Weights

The learning rate η controls how much each stage contributes:

- `η = 0.1` (default) — conservative, many stages needed, better generalisation
- `η = 0.3` — faster convergence, higher risk of overfitting
- `η = 1.0` — aggressive, equivalent to gradient descent without shrinkage

Stage weight is constant in v5. Adaptive weights per stage (complexity-penalised) are planned for v6.

## Convergence

Training stops when:

1. All `num_boost_round` stages are completed, **or**
2. `|loss_t - loss_{t-1}| < tol` for some `tol` (default `1e-6`)

The `EarlyStopping` callback provides external convergence control based on validation metrics.

## Physics-Constrained Boosting

When `lambda_pde > 0`, the gradient at each stage is augmented:

```
g_augmented = g_task + 2 * lambda_pde * pde_residual
```

This makes the boosting steps simultaneously minimise task loss and physics residuals.
The PDE residual is computed by the `PhysicsEvaluator` before each boost stage.

## Complexity Budget

The `ComplexityCost` prevents unbounded field growth:

```
Σ_r local_dim(r) ≤ max_total_dim     (default: 64)
num_regions       ≤ max_regions       (default: 16)
num_edges         ≤ max_connections   (default: 32)
```

When the budget is exceeded, `ComplexityPruner` fires before the regular DTDO.
