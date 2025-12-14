import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn import svm, datasets
from sklearn.svm import NuSVR

def SVM_training():
    # Jeu de données 2D
    X, y = datasets.make_moons(noise=0.2, random_state=42)
    
    # Paramètres SVM classique
    kernel = "linear"
    C = 10000
    gamma = 0.5
    
    # Grille pour visualisation
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 500), np.linspace(-1.0, 1.5, 500))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    def update(frame):
        ax.clear()
        
        # Ne prendre que les points jusqu'à frame+1
        X_train = X[:frame+1]
        y_train = y[:frame+1]
        
        # Vérifier qu'on a au moins 2 classes avant de fit
        if len(np.unique(y_train)) < 2:
            # Tracer juste les points
            ax.scatter(X_train[:,0], X_train[:,1], c=y_train, s=40, cmap="coolwarm", edgecolors="k")
            ax.set_title(f"Training step: {frame+1}/{len(X)} (pas assez de classes)")
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
            clf.fit(X_train, y_train)
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            # Frontière et marges
            ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap="coolwarm", alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")
            ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, linestyles='--', colors="k")
            
            # Points marginaux
            margin_points = np.abs(clf.decision_function(X_train)) <= 1 + 1e-3
            ax.scatter(X_train[~margin_points, 0], X_train[~margin_points, 1], c=y_train[~margin_points],
                       s=40, cmap="coolwarm", edgecolors="k")
            ax.scatter(X_train[margin_points, 0], X_train[margin_points, 1], c=y_train[margin_points],
                       s=100, cmap="coolwarm", edgecolors="yellow", linewidths=2, marker='o')
            
            ax.set_title(f"Training step: {frame+1}/{len(X)}")
    
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-1.0, 1.5)
    
    # Animation
    anim = FuncAnimation(fig, update, frames=len(X), interval=200)
    
    # Sauvegarde en GIF
    writer = PillowWriter(fps=2)
    anim.save("svm_training.gif", writer=writer)
    
    plt.close(fig)
    print("Animation enregistrée sous 'svm_training.gif'.")

def SVM_C_variation():
    # Jeu de données 2D non linéaire
    X, y = datasets.make_moons(noise=0.2, random_state=42)
    
    # Noyaux à comparer
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    gamma = 0.5
    
    # Valeurs de C pour l'animation (plus de frames pour plus de fluidité)
    C_values = np.logspace(-1, 2, 60)  # 60 frames de C=0.1 à C=100
    
    # Grille pour la visualisation
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 500), np.linspace(-1.0, 1.5, 500))
    
    # Configuration de la figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Fonction de mise à jour pour l'animation
    def update(frame):
        C = C_values[frame]
        for i, kernel in enumerate(kernels):
            ax = axes[i]
            ax.clear()
            
            clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=3)
            clf.fit(X, y)
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Frontière et marges
            ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap="coolwarm", alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")
            ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, linestyles='--', colors="k")
            
            # Points marginaux
            margin_points = np.abs(clf.decision_function(X)) <= 1 + 1e-3
            ax.scatter(X[~margin_points, 0], X[~margin_points, 1], c=y[~margin_points], s=40,
                       cmap="coolwarm", edgecolors="k")
            ax.scatter(X[margin_points, 0], X[margin_points, 1], c=y[margin_points], s=100,
                       cmap="coolwarm", edgecolors="yellow", linewidths=2, marker='o')
            
            ax.set_title(f"{kernel}, C={C:.2f}, points marginaux: {margin_points.sum()}")
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_xlim(-1.5, 2.5)
            ax.set_ylim(-1.0, 1.5)
    
    # Création de l'animation
    anim = FuncAnimation(fig, update, frames=len(C_values), interval=150)  # interval en ms
    
    # Sauvegarde en GIF
    writer = PillowWriter(fps=10)  # fps = 10 images par seconde
    anim.save("svm_c_variation.gif", writer=writer)
    
    plt.close(fig)  # Fermer la figure pour ne pas l'afficher dans le notebook
    print("Animation enregistrée sous 'svm_c_variation.gif'.")

def SVM_gamma_variation():
    # Jeu de données 2D non linéaire
    X, y = datasets.make_moons(noise=0.2, random_state=42)
    
    # Noyaux à comparer
    kernels = ["rbf", "poly", "sigmoid"]  # gamma utilisé dans ces noyaux
    C = 1.0  # Fixe C pour cette animation
    
    # Valeurs de gamma pour l'animation (log scale)
    gamma_values = np.logspace(-2, 1, 60)  # de 0.01 à 10, 60 frames
    
    # Grille pour visualisation
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 500), np.linspace(-1.0, 1.5, 500))
    
    # Configuration de la figure
    fig, axes = plt.subplots(1, len(kernels), figsize=(18, 5))
    
    # Fonction de mise à jour pour l'animation
    def update(frame):
        gamma = gamma_values[frame]
        for i, kernel in enumerate(kernels):
            ax = axes[i]
            ax.clear()
            
            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=3)
            clf.fit(X, y)
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Frontière et marges
            ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap="coolwarm", alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")
            ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, linestyles='--', colors="k")
            
            # Points marginaux
            margin_points = np.abs(clf.decision_function(X)) <= 1 + 1e-3
            ax.scatter(X[~margin_points, 0], X[~margin_points, 1], c=y[~margin_points], s=40,
                       cmap="coolwarm", edgecolors="k")
            ax.scatter(X[margin_points, 0], X[margin_points, 1], c=y[margin_points], s=100,
                       cmap="coolwarm", edgecolors="yellow", linewidths=2, marker='o')
            
            ax.set_title(f"{kernel}, gamma={gamma:.3f}, points marginaux: {margin_points.sum()}")
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_xlim(-1.5, 2.5)
            ax.set_ylim(-1.0, 1.5)
    
    # Création de l'animation
    anim = FuncAnimation(fig, update, frames=len(gamma_values), interval=150)
    
    # Sauvegarde en GIF
    writer = PillowWriter(fps=10)
    anim.save("svm_gamma_variation.gif", writer=writer)
    
    plt.close(fig)
    print("Animation enregistrée sous 'svm_gamma_variation.gif'.")

def NuSVM_nu_variation():
    # Jeu de données 2D non linéaire
    X, y = datasets.make_moons(noise=0.2, random_state=42)
    
    # Noyaux à comparer
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    gamma = 0.5
    
    # Valeurs de nu pour l'animation (nu ∈ (0,1])
    nu_values = np.linspace(0.01, 0.99, 60)  # 60 frames
    
    # Grille pour la visualisation
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 500), np.linspace(-1.0, 1.5, 500))
    
    # Configuration de la figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Fonction de mise à jour pour l'animation
    def update(frame):
        nu = nu_values[frame]
        for i, kernel in enumerate(kernels):
            ax = axes[i]
            ax.clear()
            
            clf = svm.NuSVC(kernel=kernel, gamma=gamma, nu=nu, degree=3)
            clf.fit(X, y)
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Frontière et marges
            ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap="coolwarm", alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")
            ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, linestyles='--', colors="k")
            
            # Points marginaux
            margin_points = np.abs(clf.decision_function(X)) <= 1 + 1e-3
            ax.scatter(X[~margin_points, 0], X[~margin_points, 1], c=y[~margin_points], s=40,
                       cmap="coolwarm", edgecolors="k")
            ax.scatter(X[margin_points, 0], X[margin_points, 1], c=y[margin_points], s=100,
                       cmap="coolwarm", edgecolors="yellow", linewidths=2, marker='o')
            
            ax.set_title(f"{kernel}, nu={nu:.2f}, points marginaux: {margin_points.sum()}")
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_xlim(-1.5, 2.5)
            ax.set_ylim(-1.0, 1.5)
    
    # Création de l'animation
    anim = FuncAnimation(fig, update, frames=len(nu_values), interval=150)  # interval en ms
    
    # Sauvegarde en GIF
    writer = PillowWriter(fps=10)  # fps = 10 images par seconde
    anim.save("nusvm_nu_variation.gif", writer=writer)
    
    plt.close(fig)  # Fermer la figure pour ne pas l'afficher dans le notebook
    print("Animation enregistrée sous 'nusvm_nu_variation.gif'.")

def SVR_training(save_path="svr_training.gif",
                 kernel="rbf",
                 C=1.0,
                 epsilon=0.25,
                 gamma='scale',
                 frames_per_point=1,
                 fps=10,
                 noise=0.2,
                 random_state=42):
    """
    Animation montrant l'entraînement progressif d'un SVR en ajoutant les points un par un.
    - kernel, C, epsilon, gamma : hyperparamètres du SVR
    - frames_per_point : multiplier le nombre de frames par point (utile si tu veux ralentir)
    """
    # Jeu de données 1D (x -> y) : sin bruité, plus pédagogique pour SVR
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(-3, 3, 80))[:, None]          # shape (n_samples, 1)
    y = np.sin(X).ravel() + rng.normal(scale=noise, size=X.shape[0])

    # grille pour la prédiction (domaine continu en 1D)
    xx = np.linspace(X.min() - 0.5, X.max() + 0.5, 1000)[:, None]

    fig, ax = plt.subplots(figsize=(8, 5))

    title = ax.set_title("")
    ax.set_xlim(float(xx.min()), float(xx.max()))
    y_min = min(y.min(), -1.5)
    y_max = max(y.max(), 1.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # éléments graphiques initiaux (on gardera les références pour une mise à jour plus propre)
    pred_line, = ax.plot([], [], lw=2, label="SVR prediction")
    data_scatter = ax.scatter([], [], s=40, edgecolors="k")
    sv_scatter = ax.scatter([], [], s=160, facecolors='none', edgecolors='yellow', linewidths=2, label="Support vectors")
    tube_fill = None

    def update(frame):
        nonlocal tube_fill

        # on ajoute un point tous les frames_per_point frames
        n_points = min(1 + frame // frames_per_point, len(X))
        X_train = X[:n_points]
        y_train = y[:n_points]

        ax.clear()
        ax.set_xlim(float(xx.min()), float(xx.max()))
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Tracer les vrais points (tout le dataset en fond léger)
        ax.scatter(X, y, c='lightgray', s=30, label="Données (globales)")

        # Tracer les points utilisés pour l'entraînement (ceux qui ont été ajoutés)
        ax.scatter(X_train, y_train, c='C0', s=50, edgecolors='k', label="Points entraînement (ajoutés)")

        # Si on a au moins 2 points, fit le SVR
        if len(X_train) >= 2:
            svr = svm.SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
            svr.fit(X_train, y_train)

            # prédiction sur la grille
            y_pred = svr.predict(xx)

            # tracer la prédiction
            ax.plot(xx, y_pred, lw=2, label="Prediction SVR")

            # epsilon-tube (y_pred ± epsilon)
            if tube_fill is not None:
                # pas strictement nécessaire à cause de ax.clear(), mais gardé pour structure
                tube_fill.remove()
            ax.fill_between(xx.ravel(),
                            y_pred - epsilon,
                            y_pred + epsilon,
                            alpha=0.25,
                            label=f"ε-tube (ε={epsilon})")

            # support vectors (indices renvoyés par l'attribut support_)
            sv_idx = svr.support_
            sv_X = X_train[sv_idx]
            sv_y = y_train[sv_idx]
            # Mettre les vecteurs de support en surbrillance
            ax.scatter(sv_X, sv_y, s=160, facecolors='none', edgecolors='yellow', linewidths=2, label="Support vectors")

            # afficher info utiles dans le titre
            ax.set_title(f"SVR training step: {n_points}/{len(X)} — kernel={kernel}, C={C}, ε={epsilon}, support_vectors={len(sv_idx)}")
        else:
            ax.set_title(f"SVR training step: {n_points}/{len(X)} — pas assez de points pour fit (>=2 requis)")

        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    total_frames = len(X) * frames_per_point
    anim = FuncAnimation(fig, update, frames=total_frames, interval=200)

    # Sauvegarde en GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)

    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")

def SVR_training_multi_kernel(save_path="svr_training_multi_kernel.gif",
                              kernels=("linear", "poly", "rbf", "sigmoid"),
                              C=1.0,
                              epsilon=0.1,
                              gamma="scale",
                              frames_per_point=2,
                              fps=10,
                              noise=0.15,
                              random_state=42):
    """
    Animation : apprentissage progressif d'un SVR sur plusieurs noyaux.
    Chaque sous-graphe montre le modèle correspondant à un kernel différent.
    """
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(-3, 3, 150))[:, None]
    y = np.sin(X).ravel() + rng.normal(scale=noise, size=X.shape[0])

    # grille pour la prédiction
    xx = np.linspace(X.min() - 0.5, X.max() + 0.5, 500)[:, None]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    def update(frame):
        n_points = min(1 + frame // frames_per_point, len(X))
        X_train = X[:n_points]
        y_train = y[:n_points]

        for i, kernel in enumerate(kernels):
            ax = axes[i]
            ax.clear()
            ax.set_xlim(float(xx.min()), float(xx.max()))
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # Données globales
            ax.scatter(X, y, c="lightgray", s=30, label="Données (globales)")

            # Données utilisées jusqu'à présent
            ax.scatter(X_train, y_train, c="C0", s=40, edgecolors="k", label="Points appris")

            if len(X_train) >= 2:
                svr = svm.SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
                svr.fit(X_train, y_train)

                y_pred = svr.predict(xx)

                # Prédiction + tube
                ax.plot(xx, y_pred, lw=2, label="Prediction SVR")
                ax.fill_between(xx.ravel(),
                                y_pred - epsilon,
                                y_pred + epsilon,
                                alpha=0.25,
                                label=f"ε-tube (ε={epsilon})")

                # Support vectors
                sv_X = X_train[svr.support_]
                sv_y = y_train[svr.support_]
                ax.scatter(sv_X, sv_y, s=160, facecolors="none",
                           edgecolors="yellow", linewidths=2, label="Support vectors")

                ax.set_title(f"{kernel} — step {n_points}/{len(X)} — SV: {len(svr.support_)}")
            else:
                ax.set_title(f"{kernel} — pas assez de points")

            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

    total_frames = len(X) * frames_per_point
    anim = FuncAnimation(fig, update, frames=total_frames, interval=200)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")

def SVR_C_variation_rbf(save_path="svr_C_variation_rbf.gif",
                           epsilon=0.1,
                           gamma="scale",
                           fps=10,
                           noise=0.15,
                           random_state=42):
    """
    Animation montrant l'influence de C sur un SVR linéaire.
    """
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(-3, 3, 80))[:, None]
    y = np.sin(X).ravel() + rng.normal(scale=noise, size=X.shape[0])

    # Grille de test pour les prédictions
    xx = np.linspace(X.min() - 0.5, X.max() + 0.5, 400)[:, None]
    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    # Valeurs de C à explorer (échelle logarithmique)
    C_values = np.logspace(-2, 2, 60)  # 60 frames, C de 0.01 à 100

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        C = C_values[frame]
        ax.clear()

        # Fit du modèle
        svr = svm.SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(X, y)
        y_pred = svr.predict(xx)

        # Tracés
        ax.scatter(X, y, color="C0", edgecolors="k", s=40, label="Données")
        ax.plot(xx, y_pred, color="C1", lw=2, label="Prédiction SVR")

        # Tube ε
        ax.fill_between(xx.ravel(),
                        y_pred - epsilon,
                        y_pred + epsilon,
                        alpha=0.25,
                        color="C1",
                        label=f"ε-tube (ε={epsilon})")

        # Vecteurs de support
        sv_X = X[svr.support_]
        sv_y = y[svr.support_]
        ax.scatter(sv_X, sv_y, s=120, facecolors="none",
                   edgecolors="yellow", linewidths=2, label="Support vectors")

        # Mise en forme
        ax.set_xlim(float(xx.min()), float(xx.max()))
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"SVR linéaire — C = {C:.2f} — SV: {len(sv_X)}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    # Animation
    anim = FuncAnimation(fig, update, frames=len(C_values), interval=150)

    # Sauvegarde en GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)

    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")

def SVR_epsilon_variation_rbf(save_path="svr_epsilon_variation_rbf.gif",
                              C=1.0,
                              gamma="scale",
                              fps=10,
                              noise=0.15,
                              random_state=42):
    """
    Animation montrant l'influence de ε (epsilon) sur un SVR avec noyau RBF.
    """
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(-3, 3, 80))[:, None]
    y = np.sin(X).ravel() + rng.normal(scale=noise, size=X.shape[0])

    # Grille de test pour les prédictions
    xx = np.linspace(X.min() - 0.5, X.max() + 0.5, 400)[:, None]
    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    # Valeurs de epsilon à explorer
    epsilon_values = np.linspace(0.01, 1.0, 60)  # 60 frames, de 0.01 à 1.0

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        epsilon = epsilon_values[frame]
        ax.clear()

        # Fit du modèle SVR
        svr = svm.SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(X, y)
        y_pred = svr.predict(xx)

        # Tracés
        ax.scatter(X, y, color="C0", edgecolors="k", s=40, label="Données")
        ax.plot(xx, y_pred, color="C1", lw=2, label="Prédiction SVR")

        # Tube ε
        ax.fill_between(xx.ravel(),
                        y_pred - epsilon,
                        y_pred + epsilon,
                        alpha=0.25,
                        color="C1",
                        label=f"ε-tube (ε={epsilon:.2f})")

        # Vecteurs de support
        sv_X = X[svr.support_]
        sv_y = y[svr.support_]
        ax.scatter(sv_X, sv_y, s=120, facecolors="none",
                   edgecolors="yellow", linewidths=2, label="Support vectors")

        # Mise en forme
        ax.set_xlim(float(xx.min()), float(xx.max()))
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"SVR RBF — ε = {epsilon:.2f} — SV: {len(sv_X)}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    # Animation
    anim = FuncAnimation(fig, update, frames=len(epsilon_values), interval=150)

    # Sauvegarde en GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)

    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")

def NuSVR_nu_variation_multi_kernel(save_path="nusvr_nu_variation_multi_kernel.gif",
                                    kernels=("linear", "poly", "rbf", "sigmoid"),
                                    C=1.0,
                                    gamma="scale",
                                    fps=10,
                                    noise=0.15,
                                    random_state=42):
    """
    Animation montrant l'influence du paramètre ν (nu) sur un NuSVR
    pour plusieurs noyaux (kernels).
    """
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(-3, 3, 80))[:, None]
    y = np.sin(X).ravel() + rng.normal(scale=noise, size=X.shape[0])

    # Grille pour la prédiction
    xx = np.linspace(X.min() - 0.5, X.max() + 0.5, 400)[:, None]
    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    # Valeurs de ν à explorer
    nu_values = np.linspace(0.01, 0.9, 60)  # 60 frames

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    def update(frame):
        nu = nu_values[frame]
        for i, kernel in enumerate(kernels):
            ax = axes[i]
            ax.clear()
            ax.set_xlim(float(xx.min()), float(xx.max()))
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # Données
            ax.scatter(X, y, color="C0", edgecolors="k", s=40, label="Données")

            # Entraînement du NuSVR
            svr = NuSVR(kernel=kernel, C=C, nu=nu, gamma=gamma)
            svr.fit(X, y)
            y_pred = svr.predict(xx)

            # Tracé de la prédiction
            ax.plot(xx, y_pred, color="C1", lw=2, label="Prédiction NuSVR")

            # Vecteurs de support
            sv_X = X[svr.support_]
            sv_y = y[svr.support_]
            ax.scatter(sv_X, sv_y, s=120, facecolors="none",
                       edgecolors="yellow", linewidths=2, label="Support vectors")

            # Titre et mise en forme
            ax.set_title(f"{kernel} — ν={nu:.2f} — SV: {len(sv_X)}")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

    # Animation
    anim = FuncAnimation(fig, update, frames=len(nu_values), interval=150)

    # Sauvegarde du GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)

    plt.close(fig)
    print(f"Animation enregistrée sous '{save_path}'.")




if __name__ == "__main__":
    SVR_training()
    SVR_training_multi_kernel()
    SVR_C_variation_rbf()
    SVR_epsilon_variation_rbf()
    NuSVR_nu_variation_multi_kernel()
