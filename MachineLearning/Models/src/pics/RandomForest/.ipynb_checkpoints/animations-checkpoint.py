import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
import os


def decision_tree_2d_stepwise(save_path="decision_tree_2d_stepwise.gif",
                               max_depth=4,
                               fps=2,
                               noise=0.2,
                               random_state=42):
    """
    Animation montrant la construction progressive d'un arbre de décision 2D
    avec mise à jour des zones colorées à chaque étape.
    """
    # Jeu de données 2D
    X, y = make_moons(n_samples=200, noise=noise, random_state=random_state)

    # Arbre complet
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X, y)

    # Grille pour visualisation
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Préparer la figure
    fig, ax = plt.subplots(figsize=(7,6))

    # Récupération des noeuds
    tree = clf.tree_
    n_nodes = tree.node_count
    nodes_order = []

    def traverse(node_id=0):
        if node_id == -1:
            return
        nodes_order.append(node_id)
        traverse(tree.children_left[node_id])
        traverse(tree.children_right[node_id])
    traverse()

    # Fonction récursive pour prédire les zones avec les noeuds explorés
    def predict_partial(X_grid, nodes_to_use, node_id=0):
        """
        Retourne -1 pour les zones non encore explorées
        """
        if node_id not in nodes_to_use:
            return -1 * np.ones(X_grid.shape[0], dtype=int)

        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if left == -1 and right == -1:
            return np.full(X_grid.shape[0], tree.value[node_id].argmax(), dtype=int)

        # Découpage
        mask_left = X_grid[:, feature] <= threshold
        mask_right = ~mask_left

        y_pred = np.full(X_grid.shape[0], -1, dtype=int)
        if left != -1:
            y_pred[mask_left] = predict_partial(X_grid[mask_left], nodes_to_use, left)
        if right != -1:
            y_pred[mask_right] = predict_partial(X_grid[mask_right], nodes_to_use, right)
        return y_pred

    def update(frame):
        ax.clear()
        # Noeuds à utiliser
        nodes_to_use = nodes_order[:frame+1]

        # Prédiction partielle
        Z = predict_partial(grid_points, nodes_to_use).reshape(xx.shape)

        # Coloration des zones (ignore -1 pour zones non explorées)
        ax.contourf(xx, yy, np.where(Z==-1, np.nan, Z), alpha=0.3, cmap="coolwarm")

        # Points
        ax.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors="k")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title(f"Decision Tree — étape {frame+1}/{len(nodes_order)}")
        ax.grid(alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(nodes_order), interval=800)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")

def decision_tree_3d_animation(save_path="decision_tree_3d.gif",
                               max_depth=3,
                               fps=2,
                               n_samples=300,
                               random_state=42):
    """
    Animation montrant la construction progressive d'un arbre de décision
    sur un jeu de données 3D (x1, x2, x3) avec zones colorées.
    """
    # Jeu de données 3D
    X, y = make_classification(n_samples=n_samples,
                               n_features=3,
                               n_informative=3,
                               n_redundant=0,
                               n_clusters_per_class=1,
                               random_state=random_state)
    
    # Arbre complet
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X, y)
    
    # Grille pour visualisation 3D
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    z_min, z_max = X[:,2].min()-0.5, X[:,2].max()+0.5
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30),
                             np.linspace(y_min, y_max, 30),
                             np.linspace(z_min, z_max, 30))
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Récupération des noeuds
    tree = clf.tree_
    nodes_order = []
    def traverse(node_id=0):
        if node_id == -1:
            return
        nodes_order.append(node_id)
        traverse(tree.children_left[node_id])
        traverse(tree.children_right[node_id])
    traverse()
    
    # Fonction pour prédiction partielle
    def predict_partial(X_grid, nodes_to_use, node_id=0):
        if node_id not in nodes_to_use:
            return -1 * np.ones(X_grid.shape[0], dtype=int)
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        if left == -1 and right == -1:
            return np.full(X_grid.shape[0], tree.value[node_id].argmax(), dtype=int)
        mask_left = X_grid[:, feature] <= threshold
        mask_right = ~mask_left
        y_pred = np.full(X_grid.shape[0], -1, dtype=int)
        if left != -1:
            y_pred[mask_left] = predict_partial(X_grid[mask_left], nodes_to_use, left)
        if right != -1:
            y_pred[mask_right] = predict_partial(X_grid[mask_right], nodes_to_use, right)
        return y_pred

    def update(frame):
        ax.clear()
        nodes_to_use = nodes_order[:frame+1]
        Z = predict_partial(grid_points, nodes_to_use)
        
        # Couleurs semi-transparentes
        for cls in np.unique(y):
            mask = Z == cls
            ax.scatter(grid_points[mask,0], grid_points[mask,1], grid_points[mask,2],
                       c=np.array([plt.cm.coolwarm(cls/len(np.unique(y)))])*0.3,
                       marker='o', s=10, alpha=0.3)
        
        # Points du dataset
        ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='coolwarm', edgecolors='k', s=40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("x₃")
        ax.set_title(f"Decision Tree 3D — étape {frame+1}/{len(nodes_order)}")
    
    anim = FuncAnimation(fig, update, frames=len(nodes_order), interval=800)
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")

def random_forest_regression_animation(save_path="random_forest_regression.gif",
                                       n_trees=30,
                                       n_samples=200,
                                       fps=2,
                                       random_state=42):
    """
    Animation montrant l'évolution d'une Random Forest de régression
    au fur et à mesure qu'on ajoute des arbres.
    L'espace (x1, x2) est coloré selon la prédiction moyenne.
    """

    # ===== 1. Dataset =====
    np.random.seed(random_state)
    X = np.random.uniform(-3, 3, size=(n_samples, 2))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.2, n_samples)

    xx, yy = np.meshgrid(np.linspace(-3, 3, 200),
                         np.linspace(-3, 3, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # ===== 2. Génération progressive =====
    frames = []
    os.makedirs("frames_rf", exist_ok=True)
    cmap = cm.get_cmap("Spectral")

    for t in range(1, n_trees + 1):
        rf = RandomForestRegressor(n_estimators=t, random_state=random_state)
        rf.fit(X, y)
        y_pred = rf.predict(X_grid).reshape(xx.shape)

        # ===== 3. Plot =====
        plt.figure(figsize=(6, 5))
        plt.title(f"Random Forest Regression — {t} arbres")

        plt.contourf(xx, yy, y_pred, levels=25, cmap=cmap, alpha=0.85)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=40)
        plt.colorbar(label="Prédiction moyenne")

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.tight_layout()

        # Sauvegarde de la frame
        filename = f"frames_rf/frame_{t:03d}.png"
        plt.savefig(filename)
        frames.append(imageio.imread(filename))
        plt.close()

    # ===== 4. Création du GIF =====
    imageio.mimsave(save_path, frames, duration=1/fps, loop=0)  # loop=0 → boucle infinie
    print(f"✅ Animation sauvegardée sous '{save_path}' (boucle infinie).")


def random_forest_classification_animation(save_path="random_forest_classification.gif",
                                           n_trees=30,
                                           n_samples=200,
                                           n_classes=3,
                                           fps=2,
                                           random_state=42):
    """
    Animation montrant l'évolution d'une Random Forest de classification
    au fur et à mesure qu'on ajoute des arbres.
    L'espace (x1, x2) est coloré selon la classe prédite majoritaire.
    """

    # ===== 1. Dataset synthétique =====
    np.random.seed(random_state)
    X = np.random.uniform(-3, 3, size=(n_samples, 2))
    # Génère des classes non linéaires
    y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) > 0).astype(int)
    if n_classes == 3:
        # On ajoute une 3e classe pour complexifier un peu
        y[(X[:, 0]**2 + X[:, 1]**2) < 2.5] = 2

    xx, yy = np.meshgrid(np.linspace(-3, 3, 200),
                         np.linspace(-3, 3, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # ===== 2. Génération progressive =====
    frames = []
    os.makedirs("frames_rf_classif", exist_ok=True)
    cmap = cm.get_cmap("Spectral", n_classes)

    for t in range(1, n_trees + 1):
        rf = RandomForestClassifier(n_estimators=t, random_state=random_state)
        rf.fit(X, y)
        y_pred = rf.predict(X_grid).reshape(xx.shape)

        # ===== 3. Plot =====
        plt.figure(figsize=(6, 5))
        plt.title(f"Random Forest Classification — {t} arbres")

        plt.contourf(xx, yy, y_pred, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, alpha=0.7)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=40)
        plt.colorbar(scatter, ticks=range(n_classes), label="Classe prédite")

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.tight_layout()

        # Sauvegarde de la frame
        filename = f"frames_rf_classif/frame_{t:03d}.png"
        plt.savefig(filename)
        frames.append(imageio.imread(filename))
        plt.close()

    # ===== 4. Création du GIF =====
    imageio.mimsave(save_path, frames, duration=1/fps, loop=0)  # loop=0 → boucle infinie
    print(f"✅ Animation sauvegardée sous '{save_path}' (boucle infinie).")



if __name__ == "__main__":
    random_forest_classification_animation(
        save_path="random_forest_classification_evolution.gif",
        n_trees=25,
        n_samples=300,
        n_classes=1,
        fps=3
        )

    random_forest_regression_animation(
    save_path="random_forest_regression_evolution.gif",
    n_trees=25,
    n_samples=250,
    fps=1
    )