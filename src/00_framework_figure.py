"""Fig. 1: staged early-warning framework diagram (pure matplotlib, publication quality)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import savefig, COL2, INK, INK2, STAGE_COLORS

fig, ax = plt.subplots(figsize=(COL2, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

def box(x, y, w, h, title, body, fc="#f5f9fe", ec="#86b6ef", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=6.4, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.3)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 color=INK2, lw=0.9, shrinkA=0, shrinkB=0))

# Row 1
box(1, 25, 17, 17, "1. Data", "Realinho et al. (2022)\nKaggle release\n4,424 students\n34 predictors, 3 classes")
box(22, 25, 18, 17, "2. Audit & cleaning", "Missing, duplicates,\nlogic checks\nIQR & |z|>3 outliers\nStructural zeros flagged")
box(44, 25, 18, 17, "3. Inferential EDA", "Kruskal–Wallis ε²\nχ² + Cramér's V (Holm)\nSpearman, VIF\nWilson CIs")
# stage boxes
ax.add_patch(FancyBboxPatch((66, 25), 33, 17, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc="white", ec="#86b6ef", lw=1.0))
ax.text(82.5, 39.7, "4. Stage-specific feature sets", ha="center", va="top", fontsize=7.2, weight="bold", color=INK)
for i, (s, lbl, n) in enumerate([("S0", "Enrolment", "22 + 2 eng."), ("S1", "End sem. 1", "+6 + 3 eng."),
                                 ("S2", "End sem. 2", "+6 + 7 eng.")]):
    x = 67.5 + i * 10.6
    ax.add_patch(FancyBboxPatch((x, 26.5), 9.0, 9.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=STAGE_COLORS[s], ec="none"))
    ax.text(x + 4.5, 33.6, s, ha="center", fontsize=7.5, weight="bold", color="white" if s != "S0" else INK)
    ax.text(x + 4.5, 30.6, lbl, ha="center", fontsize=5.5, color="white" if s != "S0" else INK)
    ax.text(x + 4.5, 28.2, n, ha="center", fontsize=5.0, color="white" if s != "S0" else INK)
arrow(18.6, 33.5, 21.4, 33.5); arrow(40.6, 33.5, 43.4, 33.5); arrow(62.6, 33.5, 65.4, 33.5)

# Row 2
box(66, 2, 33, 18, "5. Model development (×3 stages)",
    "LR, SVM, RF, XGBoost,\nLightGBM, CatBoost, MLP\nIn-fold encoding, scaling, imputation\nClass re-weighting (SMOTE ablation)\nRandomised search: 15 candidates\n× 5-fold CV, macro-F1")
box(36, 2, 26, 18, "6. Inferential comparison",
    "5×3 repeated stratified CV\nFriedman + Nemenyi (CD)\nNadeau–Bengio corrected t\n+ Holm; held-out test with\n1,000× bootstrap 95% CI")
box(1, 2, 31, 18, "7. Deployment layer",
    "Isotonic calibration (Brier, ECE)\nTreeSHAP (global + local)\nFairness slicing (TPR/FPR gaps)\nCapacity / cumulative gains\n→ stage-wise early-warning policy",
    fc="#fdf3ee", ec="#eb6834")
arrow(82.5, 24.4, 82.5, 20.6); arrow(65.4, 11, 62.6, 11); arrow(35.4, 11, 32.6, 11)
ax.text(50, 43.6, "Train/test split 80/20 (stratified, seed 42) — test set used once, after all selection steps",
        ha="center", fontsize=6.2, color=INK2, style="italic")
savefig(fig, "fig00_framework")
print("ok")
