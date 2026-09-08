from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .core import Booster


def _load_model_data(booster: "Booster") -> dict:
    if booster._last_model_json is None:
        raise RuntimeError("Call booster.save() before visualising — "
                           "model JSON must be available.")
    return json.loads(booster._last_model_json)


def plot_pde_residuals(booster: "Booster", ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    data   = _load_model_data(booster)
    stages = data.get("stages", [])
    before = [s.get("pde_before", 0.0) for s in stages]
    after  = [s.get("pde_after",  0.0) for s in stages]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(stages)))
    ax.plot(x, before, label="pde_before", linewidth=1.5)
    ax.plot(x, after,  label="pde_after",  linewidth=1.5, linestyle="--")
    ax.set_xlabel("Stage")
    ax.set_ylabel("PDE Residual")
    ax.set_title("PDE Residual per Boosting Stage")
    ax.legend()
    return ax


def plot_mutation_history(booster: "Booster", ax=None):
    import matplotlib.pyplot as plt

    data   = _load_model_data(booster)
    stages = data.get("stages", [])

    mutation_counts = [len(s.get("mutations", [])) for s in stages]
    mutation_types  = {}
    for s_idx, s in enumerate(stages):
        for ev in s.get("mutations", []):
            t = ev.get("type", "no_op")
            mutation_types.setdefault(t, [])
            mutation_types[t].append(s_idx)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.bar(range(len(mutation_counts)), mutation_counts, color="steelblue", alpha=0.7)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mutations")
    ax.set_title("Topology Mutations per Stage")

    if mutation_types:
        legend_y = 0
        for t, idxs in mutation_types.items():
            ax.scatter(idxs, [0.05] * len(idxs), label=t, s=20, zorder=3)
        ax.legend(fontsize=7, loc="upper right")

    return ax


def plot_dimension_map(booster: "Booster", stage: int = -1, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    data   = _load_model_data(booster)
    stages = data.get("stages", [])
    if not stages:
        raise ValueError("No stages in model.")

    s_idx  = stage if stage >= 0 else len(stages) + stage
    params = stages[s_idx].get("params", [])
    n      = len(params)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    side   = max(1, int(n ** 0.5))
    matrix = np.array(params[:side * side]).reshape(side, side)
    im     = ax.imshow(matrix, cmap="RdBu", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Field Parameters — Stage {s_idx}")
    return ax


def plot_field_slice(booster: "Booster", stage: int = -1,
                     axis: int = 0, value: float = 0.0, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    data   = _load_model_data(booster)
    stages = data.get("stages", [])
    s_idx  = stage if stage >= 0 else len(stages) + stage
    params = np.array(stages[s_idx].get("params", []))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.plot(params, linewidth=1.2)
    ax.axhline(value, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Value")
    ax.set_title(f"Field Slice — Stage {s_idx}, axis={axis}")
    return ax


def plot_topology(booster: "Booster", stage: int = -1, ax=None):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("networkx and matplotlib are required: pip install networkx matplotlib")

    data   = _load_model_data(booster)
    stages = data.get("stages", [])
    s_idx  = stage if stage >= 0 else len(stages) + stage
    num_mutations = len(stages[s_idx].get("mutations", []))

    G = nx.Graph()
    G.add_node(0, label=f"R0\n{num_mutations}mut")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue",
            node_size=800, font_size=8)
    ax.set_title(f"Field Topology — Stage {s_idx}")
    return ax


def plot_topology_evolution(booster: "Booster",
                             interval: int = 1,
                             save_gif: Optional[str] = None):
    import matplotlib.pyplot as plt

    data   = _load_model_data(booster)
    stages = data.get("stages", [])

    figs = []
    for i in range(0, len(stages), interval):
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_dimension_map(booster, stage=i, ax=ax)
        ax.set_title(f"Stage {i}")
        figs.append(fig)

    if save_gif and figs:
        try:
            import imageio, io
            frames = []
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=72)
                buf.seek(0)
                frames.append(imageio.imread(buf))
                plt.close(fig)
            imageio.mimsave(save_gif, frames, fps=2)
        except ImportError:
            pass

    return figs
