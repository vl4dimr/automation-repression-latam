"""Draw the causal diagram used in 01_design.md (vector PDF + SVG)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT

nodes = {
    "P":     (0.05, 0.80, "Commodity\nprices x initial\nexport mix\n(instrument)"),
    "Z":     (0.05, 0.20, "Donor robot\nimports x initial\nmanuf. share\n(instrument)"),
    "K":     (0.30, 0.80, "Capital per\nworker  K/L"),
    "A":     (0.30, 0.20, "Automation\n(robot stock,\nroutine share)"),
    "SL":    (0.52, 0.50, "Labour share\ns_L"),
    "THR":   (0.68, 0.80, "Revolt threat\n(beta: latent)"),
    "MOB":   (0.68, 0.20, "Mobilization /\nprotest"),
    "REP":   (0.90, 0.65, "Repression"),
    "RED":   (0.90, 0.35, "Redistribution"),
    "COUP":  (0.90, 0.05, "Coup /\nbacksliding"),
    "R":     (0.52, 0.95, "Cost of repression\n(coercive capacity,\nintl. pressure)"),
    "U":     (0.52, 0.05, "Unobserved:\nresource rents,\nCold War, state\ncapacity"),
}
edges = [
    ("P", "K", "solid"), ("Z", "A", "solid"),
    ("K", "A", "solid"), ("K", "SL", "solid"), ("A", "SL", "solid"),
    ("SL", "THR", "solid"), ("SL", "MOB", "solid"),
    ("THR", "REP", "solid"), ("THR", "RED", "solid"), ("MOB", "REP", "solid"),
    ("K", "REP", "solid"), ("K", "RED", "solid"), ("K", "COUP", "solid"),
    ("RED", "COUP", "solid"),
    ("R", "REP", "solid"),
    ("REP", "MOB", "dashed"), ("REP", "A", "dashed"), ("RED", "A", "dashed"),
    ("REP", "K", "dashed"),
    ("U", "K", "dotted"), ("U", "REP", "dotted"), ("U", "RED", "dotted"), ("U", "MOB", "dotted"),
]
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.08, 1.08); ax.axis("off")
for k, (x, y, lab) in nodes.items():
    fc = "#f2f2f2" if k in ("P", "Z") else ("#ffffff" if k not in ("THR", "U") else "#fff3d6")
    ax.text(x, y, lab, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec="#333333", lw=0.8))
for a, b, st in edges:
    xa, ya, _ = nodes[a]; xb, yb, _ = nodes[b]
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=12,
                                 lw=0.9, color="#222222" if st == "solid" else "#777777",
                                 linestyle=st, shrinkA=28, shrinkB=28,
                                 connectionstyle="arc3,rad=0.08"))
ax.text(0.0, -0.06, "Solid: model channels (Acemoglu, Gitmez & Shadmehr 2026). Dashed: feedback the model implies "
        "(state chooses automation; repression lowers observed protest; regime affects accumulation).\n"
        "Dotted: confounders. Grey boxes: excluded instruments. Yellow: latent or unobserved.",
        fontsize=8, va="top")
out = ROOT / "01_design_dag"
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".svg", bbox_inches="tight")
print("saved", out)
