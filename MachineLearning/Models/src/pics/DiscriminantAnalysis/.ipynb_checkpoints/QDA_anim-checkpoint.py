import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.patches import Ellipse
import seaborn as sns

sns.set_style("darkgrid")

# -----------------------
# Load Iris dataset
# -----------------------
iris = load_iris()
X = iris.data[:, :2]  # 2D pour visualisation
y = iris.target
target_names = iris.target_names

# -----------------------
# Fit QDA
# -----------------------
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)

# -----------------------
# Setup figure
# -----------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("QDA sur Iris (points + ellipses colorées)")
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")

# Axis limits
margin = 0.5
ax.set_xlim(X[:, 0].min() - margin, X[:, 0].max() + margin)
ax.set_ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)

# Colors per class
colors = ["tab:blue", "tab:orange", "tab:green"]

# Scatter points
scatters = []
for k in range(3):
    scatter = ax.scatter([], [], label=target_names[k], color=colors[k], alpha=0.8)
    scatters.append(scatter)

ax.legend()

# Ellipse objects (filled with semi-transparent color)
ellipses = []
for k in range(3):
    ellipse = Ellipse((0,0), width=0, height=0, angle=0,
                      edgecolor=colors[k],
                      facecolor=colors[k],
                      alpha=0.2, lw=2)
    ax.add_patch(ellipse)
    ellipses.append(ellipse)

# -----------------------
# Function to get ellipse parameters
# -----------------------
def get_ellipse_params(mean, cov, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    return width, height, angle

# -----------------------
# Animation update
# -----------------------
def update(frame):
    # Update points
    for k in range(3):
        mask = (y[:frame] == k)
        scatters[k].set_offsets(X[:frame][mask])

    # Update ellipses (pleines dès le début)
    for k in range(3):
        mean = X[y==k].mean(axis=0)
        cov = np.cov(X[y==k].T)
        width, height, angle = get_ellipse_params(mean, cov)
        ellipses[k].center = mean
        ellipses[k].width = width
        ellipses[k].height = height
        ellipses[k].angle = angle

    ax.set_title(f"QDA on Iris (points : {frame}/{len(X)})")
    return scatters + ellipses

# -----------------------
# Create animation
# -----------------------
anim = FuncAnimation(
    fig,
    update,
    frames=np.arange(1, len(X)+1),
    interval=50,
    blit=False,
    repeat=True
)

# Save GIF
anim.save("qda_iris.gif", writer="pillow", fps=20)
