#!/usr/bin/env python3
"""1200x630 social-preview card for the Retro-DARC release page."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

rcParams.update({"font.family": "STIX Two Text", "svg.fonttype": "none"})

fig = plt.figure(figsize=(12, 6.3), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), 12, 6.3, color="#f6f6f3"))
ax.add_patch(plt.Rectangle((0, 6.12), 12, 0.18, color="#3565E0"))

ax.text(0.75, 4.98, "MEASURED ON PUBLIC CHECKPOINTS · JULY 2026",
        fontsize=13, color="#5a5f6b", family="Helvetica Neue")
ax.text(0.72, 4.28, "Retro-DARC", fontsize=54, color="#16181d", fontweight="bold")
ax.text(0.75, 3.30, "Function-Preserving Residual-Memory Adapters\nfor Pretrained Language and World Models",
        fontsize=23, color="#16181d", va="top", linespacing=1.35)
ax.text(0.75, 1.90, "Bitwise-identity insertion · delta-key addressing · causal memory audits\nDepth reads that beat LoRA at matched parameters — replicated across accelerators",
        fontsize=15.5, color="#5a5f6b", va="top", linespacing=1.5)
ax.text(0.75, 0.62, "Ada Cyborg   ·   adacyb0rg.github.io/retro-darc-site", fontsize=14, color="#3565E0")

# the lab cat, sitting bottom-right on the baseline
def cat(ax, x0, y0, s, lw=2.6, color="#16181d"):
    import matplotlib.path as mpath
    kw = dict(fill=False, edgecolor=color, linewidth=lw, capstyle="round", joinstyle="round")
    P = mpath.Path
    def draw(verts, codes):
        ax.add_patch(matplotlib.patches.PathPatch(P(
            [(x0 + vx * s, y0 + vy * s) for vx, vy in verts], codes), **kw))
    draw([(24,63),(21,77),(33,70)], [P.MOVETO, P.LINETO, P.LINETO])           # left ear
    draw([(44,63),(47,77),(35,70)], [P.MOVETO, P.LINETO, P.LINETO])           # right ear
    ax.add_patch(matplotlib.patches.Circle((x0+34*s, y0+54*s), 14*s, **kw))   # head
    draw([(27,53),(30,50),(33,53)], [P.MOVETO, P.CURVE3, P.CURVE3])           # eye
    draw([(36,53),(39,50),(42,53)], [P.MOVETO, P.CURVE3, P.CURVE3])           # eye
    draw([(46,46),(62,42),(74,29),(76,8)], [P.MOVETO, P.CURVE4, P.CURVE4, P.CURVE4])  # back
    draw([(25,41),(22,29),(22,18),(24,8)], [P.MOVETO, P.CURVE4, P.CURVE4, P.CURVE4])  # chest
    draw([(24,8),(76,8)], [P.MOVETO, P.LINETO])                               # ground
    draw([(76,10),(96,6),(106,18),(98,28)], [P.MOVETO, P.CURVE4, P.CURVE4, P.CURVE4]) # tail

cat(ax, 9.55, 0.55, 0.022)
fig.savefig("og-card.png")
print("wrote og-card.png")
