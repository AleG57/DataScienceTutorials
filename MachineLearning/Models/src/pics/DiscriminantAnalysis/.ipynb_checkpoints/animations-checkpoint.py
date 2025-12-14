import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Choisir deux features (par exemple : petal length & petal width)
i, j = 2, 3  # colonnes 2 et 3
X_pair = X[:, [i, j]]

# Normaliser les données
X_pair = StandardScaler().fit_transform(X_pair)

# Couleurs pour les classes
colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# --------------------------
# 1️⃣ Figure "avant QDA"
# --------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, cmap=colors, edgecolors='k')
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.title("Nuage de points (Iris) — Avant QDA")

# --------------------------
# 2️⃣ Figure "après QDA"
# --------------------------
model = QuadraticDiscriminantAnalysis()
model.fit(X_pair, y)

# Créer une grille pour visualiser les frontières
x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, cmap=colors, alpha=0.4)
plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, cmap=colors, edgecolors='k')
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.title("Séparation par QDA (Iris)")

plt.tight_layout()
plt.savefig("qda_iris.png", dpi=300)
plt.show()
