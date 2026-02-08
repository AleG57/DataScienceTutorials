import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

sns.set_style("darkgrid")

# -----------------------
# Load Iris dataset
# -----------------------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# -----------------------
# Fit LDA and transform
# -----------------------
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# -----------------------
# Setup figure
# -----------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("LDA sur Iris (projection animée)")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")

# Axis limits (fixed for smooth animation)
margin = 1
x_min, x_max = X_lda[:, 0].min() - margin, X_lda[:, 0].max() + margin
y_min, y_max = X_lda[:, 1].min() - margin, X_lda[:, 1].max() + margin
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Colors per class
colors = ["tab:blue", "tab:orange", "tab:green"]

# Create scatter objects (one per class)
scatters = []
for k in range(3):
    scatter = ax.scatter([], [], label=target_names[k], color=colors[k], alpha=0.8)
    scatters.append(scatter)

ax.legend()

# -----------------------
# Animation update function
# -----------------------
def update(frame):
    # frame = number of points shown
    for k in range(3):
        mask = (y[:frame] == k)
        scatters[k].set_offsets(X_lda[:frame][mask])
    ax.set_title(f"LDA on Iris (points : {frame}/{len(X)})")
    return scatters

# -----------------------
# Create animation
# -----------------------
anim = FuncAnimation(
    fig,
    update,
    frames=np.arange(1, len(X) + 1),
    interval=50,
    blit=False,
    repeat=True
)

anim.save("lda_iris.gif", writer="pillow", fps=20)