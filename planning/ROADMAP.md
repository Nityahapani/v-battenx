# V-BATTEN-X · Master Roadmap

This folder is the **single source of truth** for what we are building, in what order,
and what "done" means at each stage. Every team member should read this before touching code.

---

## What Is V-BATTEN-X?

V-BATTEN-X is a physics-informed variational boosting library. Unlike XGBoost or LightGBM,
which boost over fixed-dimensional decision trees, V-BATTEN-X boosts over a **dynamic
physical field** — a structure that can change its geometry, topology, and local dimensionality
at each boosting stage in response to physics residuals.

The core idea is the **operator**:

```
Oθ : (F, K, d, T) → (F', K', d', T')
```

where:
- **F** — Latent field (where states live)
- **K** — Topology (how regions connect)
- **d** — Dimension map (degrees of freedom per region)
- **T** — Tensor field (how components interact)

The component that performs this transformation is the **DTDO**
(Dynamic Topological-Dimensional Operator) — the algorithmic heart of the library.

---

## Version Overview

| Version | Theme | Key Deliverable | Status |
|---------|-------|-----------------|--------|
| [v1](./v1/README.md) | Foundation | End-to-end pipeline works; linear booster; sklearn API | 🔲 Not started |
| [v2](./v2/README.md) | Physics | Real PDE evaluation; physics-informed loss; symmetry encoder | 🔲 Not started |
| [v3](./v3/README.md) | DTDO | Rule-based topology/dimension mutation; variational booster | 🔲 Not started |
| [v4](./v4/README.md) | Scale | Learned DTDO (neural policy); GPU; distributed training | 🔲 Not started |
| [v5](./v5/README.md) | Release | Stable ABI; full docs; PyPI/CRAN/Maven; v1.0.0 | 🔲 Not started |

Update status to 🔄 In Progress / ✅ Done as work proceeds.

---

## Dependency Graph

```
v1 (foundation)
 └── v2 (physics layer)
      └── v3 (DTDO — rule-based)
           ├── v4a (learned DTDO)     ← can start after v3 is 50% done
           └── v4b (GPU + distributed) ← can start after v3 is done
                └── v5 (release)
```

v4a and v4b can be developed in parallel by different team members once v3 is stable.

---

## Guiding Principles

**1. Physics first, performance second.**
The library exists to incorporate physical knowledge. A physically correct slow model
beats a fast model that violates conservation laws.

**2. Every abstraction must be replaceable.**
All core types (`LatentField`, `FieldTopology`, `TopologicalOperator`, etc.) are ABCs.
The user can swap in custom implementations without touching the core loop.

**3. The DTDO is the contribution.**
Everything else (encoder, physics eval, booster) exists in some form in other libraries.
The DTDO does not. It is the algorithmic novelty. Every design decision should protect
and clarify the DTDO's interface.

**4. Complexity budget is always respected.**
The `ComplexityCost` is not optional. A model that grows unboundedly is not a model.
Every mutation must check `can_afford()` before executing.

**5. No new algorithmic ideas after v3.**
v4 and v5 are about scale and polish. New ideas go into a `v6/` design doc.
This keeps scope controlled.

---

## How to Use This Folder

- **Before writing code:** read the version doc for the current milestone
- **Before opening a PR:** check the exit criteria at the bottom of the version doc
- **When stuck:** re-read `doc/architecture.md` and `doc/dtdo.md`
- **When you have a new idea:** add it to `v5/README.md` under "Post-v5 / v6 Ideas"

---

## Team Conventions

- Commits: follow conventional commits — `feat(scope):`, `fix(scope):`, `test(scope):`, `docs:`, `perf:`
- Branch names: `v1/feature-name`, `v2/pde-ops`, `v3/dtdo-threshold`, etc.
- PR template: in `.github/ISSUE_TEMPLATE/` — always fill out the "Exit criteria affected" section
- Code style: `clang-format` (C++), `black` + `isort` (Python), `styler` (R)
- No `TODO` comments in merged code — open an issue instead

---

## Open Questions (resolve before v3)

| Question | Owner | Deadline |
|----------|-------|----------|
| Learned DTDO: native C++ autograd vs Python `torch.nn.Module` via C API? | — | Before v4 starts |
| Simplex complex: do we need Betti numbers in v3 or can they wait for v5? | — | Before v3 starts |
| GPU: A100 vs H100 as primary target? Affects memory layout choices. | — | Before v4 starts |
| R package: Rcpp or raw JNI? Rcpp is easier but adds a dependency. | — | Before v5 starts |
| Model format: binary + JSON envelope vs pure protobuf? | — | Before v2 serialisation work |

---

*Last updated: initial draft. Update this file whenever a version is marked done or a major decision is made.*
