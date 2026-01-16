import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os

from sklearn.ensemble import IsolationForest

sns.set(style="whitegrid")

# -----------------------------
# 1️⃣ Données synthétiques
# -----------------------------
np.random.seed(42)

X_normal = 0.6 * np.random.randn(300, 2)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))

X = np.vstack([X_normal, X_outliers])

# -----------------------------
# 2️⃣ Isolation Forest
# -----------------------------
iso = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.06,
    random_state=42
)

iso.fit(X)

# Scores (plus grand = plus anormal)
scores = -iso.score_samples(X)

# -----------------------------
# 3️⃣ Grille pour frontière
# -----------------------------
xx, yy = np.meshgrid(
    np.linspace(-7, 7, 300),
    np.linspace(-7, 7, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

grid_scores = -iso.score_samples(grid)
Z = grid_scores.reshape(xx.shape)

# -----------------------------
# 4️⃣ Animation progressive
# -----------------------------
os.makedirs("frames", exist_ok=True)
frames = []

thresholds = np.linspace(scores.min(), scores.max(), 15)

for i, t in enumerate(thresholds):

    plt.figure(figsize=(7, 6))

    # Frontière d'isolement
    plt.contourf(
        xx, yy, Z,
        levels=50,
        cmap="coolwarm",
        alpha=0.35
    )

    # Points normaux / outliers selon seuil
    mask_outlier = scores >= t

    plt.scatter(
        X[~mask_outlier, 0],
        X[~mask_outlier, 1],
        c="steelblue",
        s=40,
        label="Points normaux"
    )

    plt.scatter(
        X[mask_outlier, 0],
        X[mask_outlier, 1],
        c="crimson",
        s=60,
        label="Outliers détectés"
    )

    plt.title(
        f"Isolation Forest – Détection d'anomalies\n"
        f"Seuil du score : {t:.2f}",
        fontsize=14
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.axis("equal")

    fname = f"frames/frame_{i:03d}.png"
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()

    frames.append(fname)

# -----------------------------
# 5️⃣ Création du GIF
# -----------------------------
with imageio.get_writer(
    "isolation_forest_outliers.gif",
    mode="I",
    duration=0.8,
    loop=0
) as writer:
    for frame in frames:
        writer.append_data(imageio.imread(frame))

# Nettoyage
for f in frames:
    os.remove(f)

print("GIF créé : isolation_forest_outliers.gif")
