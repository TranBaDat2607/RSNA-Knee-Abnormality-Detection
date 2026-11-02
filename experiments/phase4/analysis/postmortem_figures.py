"""Draw the two README figures from reports/submissions_final.csv.  Run from the repo root.

Input: reports/submissions_final.csv (one row per submission, read from Kaggle after the close).
Output: docs/figures/private_over_time.png, docs/figures/public_vs_private.png
"""
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
TRACKS = [  # label, colour, marker: identity is never colour alone
    ("own", "Own models alone", "#2a78d6", "o"),
    ("clean", "Blends, no OAI-trained weights", "#eb6834", "s"),
    ("oai", "Blends with OAI-trained public weights", "#1baf7a", "^"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def base(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def scatter(ax, df, x, y):
    for key, label, colour, marker in TRACKS:
        d = df[df.track == key]
        ax.scatter(d[x], d[y], s=64, c=colour, marker=marker, edgecolors=SURFACE, linewidths=1.5, label=label, zorder=3)


df = pd.read_csv("reports/submissions_final.csv", parse_dates=["date"])

fig, ax = base(9, 4.6)
scatter(ax, df, "date", "private")
notes = {55429575: ("Baseline + LLM labels  0.804", 8, -4), 56184308: ("E8, clean 4-arm stack  0.921", -80, 11),
         56814752: ("Best own models alone  0.913", -178, -3), 57186035: ("C3, clean-track best  0.931", -92, -19),
         57490753: ("A20, final  0.943", -100, 12)}
for ref, (text, dx, dy) in notes.items():
    r = df[df.ref == ref].iloc[0]
    ax.annotate(text, (r.date, r.private), xytext=(dx, dy), textcoords="offset points", fontsize=10, color=INK)
ax.set_ylabel("Private leaderboard (macro AUC)")
ax.set_title("Private score of every scored submission", loc="left", fontweight="bold", color=INK, pad=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend(frameon=False, loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig("docs/figures/private_over_time.png", facecolor=SURFACE)

drop = (df.public - df.private).mean()
fig, ax = base(6.4, 5)
scatter(ax, df, "public", "private")
xs = [df.public.min(), df.public.max()]
ax.plot(xs, [v - drop for v in xs], color=MUTED, lw=1.2, ls="--", zorder=2)
ax.annotate(f"public − {drop:.3f}", (0.885, 0.885 - drop), xytext=(10, -14), textcoords="offset points", fontsize=10, color=MUTED)
ax.set_xlabel("Public leaderboard")
ax.set_ylabel("Private leaderboard")
ax.set_title("The public leaderboard tracked private", loc="left", fontweight="bold", color=INK, pad=12)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
fig.tight_layout()
fig.savefig("docs/figures/public_vs_private.png", facecolor=SURFACE)
