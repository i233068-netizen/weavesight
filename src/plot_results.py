import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.expanduser("~/weavesight/results")
rows = list(csv.DictReader(open(f"{R}/manahil_merged_table.csv")))
cats = [c for c in dict.fromkeys(r["category"] for r in rows) if c != "aitex_05"]
models = ["cae", "dinomaly", "patchcore"]
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, (key, title) in zip(axes, [("image_auroc", "Image AUROC"), ("pixel_auroc", "Pixel AUROC"), ("aupro", "AUPRO (fpr limit 0.3)")]):
    for i, m in enumerate(models):
        vals = [float(next(r[key] for r in rows if r["category"] == c and r["model"] == m)) for c in cats]
        ax.bar(np.arange(len(cats)) + (i - 1) * 0.27, vals, 0.27, label=m)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(ncol=3, fontsize=8)
axes[-1].set_xticks(np.arange(len(cats)))
axes[-1].set_xticklabels(cats, rotation=45, ha="right")
fig.suptitle("WeaveSight: CAE vs Dinomaly vs PatchCore (single seed; aitex_05 omitted: one defect tile)", fontsize=10)
plt.tight_layout()
out = f"{R}/figures/manahil_results_overview.png"
plt.savefig(out, dpi=130)
print("saved", out)
