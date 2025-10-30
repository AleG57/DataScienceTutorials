import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
    
def ACP_interp(data_acp,
               acp_norm='oui',
               choix_n_cp='oui', n_cp_max=10, ech_n_pc_x=6, ech_n_pc_y=4,
               cercle='oui', ech_cer=6,
               projection='oui', label_indiv='non', ech_pr_x=5, ech_pr_y=4,
               cp_x=1, cp_y=2):
    """
    Cette fonction renvoie des éléments pour interpréter les résultats d'une ACP.

    Arguments :
    data_acp    -- Le dataframe contenant les colonnes en entrée de l'ACP
    acp_norm    -- Le choix d'une ACP normée ('oui' par défaut) ou non
    choix_n_cp  -- La production des graphiques de 'choix' du nombre de composantes principales ('oui' par défaut) ou non
    n_cp_max    -- Le nombre maximum de composantes affichées sur les graphiques de 'choix' (10 par défaut)
    ech_n_pc_x  -- La largeur des graphiques de 'choix' en pouces-inches (6 par défaut)
    ech_n_pc_y  -- La hauteur des graphiques de 'choix' en pouces-inches (4 par défaut)
    cercle      -- La production du 'cercle des corrélations' ('oui' par défaut) ou non
    ech_cer     -- La taille du graphique 'cercle des corrélations' en pouces-inches (6 par défaut)
    projection  -- La production de la 'projection des individus' ('oui' par défaut) ou non
    label_indiv -- L'ajout des labels des individus sur le graphique 'projection des individus' ou non ('non' par défaut) 
    ech_pr_x    -- La largeur du graphique 'projection des individus' en pouces-inches (5 par défaut)
    ech_pr_y    -- La hauteur du graphique 'projection des individus' en pouces-inches (4 par défaut)
    cp_x        -- Le numéro de la composante principale sur l'axe des abscisses pour les graphiques 'cercle des corrélations' et 'projection des individus'
    cp_y        -- Le numéro de la composante principale sur l'axe des ordonnées pour les graphiques 'cercle des corrélations' et 'projection des individus'

    Retourne :
    acp_out      -- L'objet ACP en sortie de sklearn
    data_acp_out -- Le dataframe d'entrée augmenté des composantes principes, des cosinus carrés et des contributions
    """
    n = data_acp.shape[0]
    p = data_acp.shape[1]
    n_cp = min(n_cp_max, p)
    
    if acp_norm == 'oui':
        norm = StandardScaler(with_mean=True, with_std=True)
        data_acp_norm = norm.fit_transform(data_acp)
    else:
        norm = StandardScaler(with_mean=True, with_std=False)
        data_acp_norm = norm.fit_transform(data_acp)
    
    acp_model = decomposition.PCA(svd_solver='full')
    acp_out = acp_model.fit(data_acp_norm)

    val_prop = acp_out.explained_variance_ * (n-1) / n
    part_inertie_expl = acp_out.explained_variance_ratio_
    
    if choix_n_cp == 'oui':
        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(ech_n_pc_x, ech_n_pc_y))
        sns.barplot(x=np.arange(1, n_cp+1), y=val_prop[0:n_cp])
        ax.set_xlabel('Composante principale')
        ax.set_ylabel('Valeur propre')
        ax.set_title('Eboulis des valeurs propres')
        plt.show()
        
        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(ech_n_pc_x, ech_n_pc_y))
        sns.barplot(x=np.arange(1, n_cp+1), y=100 * val_prop[0:n_cp] / np.sum(val_prop))
        ax.set_xlabel('Composante principale')
        ax.set_ylabel('Pourcentage')
        ax.set_title("Part d'inertie expliquée (%)")
        plt.show()
        
        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(ech_n_pc_x, ech_n_pc_y))
        sns.barplot(x=np.arange(1, n_cp+1), y=np.cumsum(part_inertie_expl[0:n_cp])*100)
        ax.set_xlabel('Nombre de composantes principales')
        ax.set_ylabel('Pourcentage')
        ax.set_title("Part d'inertie expliquée cumulée (%)")
        plt.show()
    
    coord = acp_out.fit_transform(data_acp_norm)

    data_acp_out = data_acp.copy()
    
    for i in range(p):
        data_acp_out['CP_' + str(i+1)] = coord[:,i]
    
    for i in range(p):
        data_acp_out['Cos2_' + str(i+1)] = coord[:,i]**2 / np.sum(data_acp_norm**2, axis=1)
    
    ctr = coord**2

    for i in range(p):
        data_acp_out['CTR_' + str(i+1)] = coord[:,i]**2 / (n * val_prop[i])
    
    if cercle == 'oui':
        cor_var = np.zeros((p, p))
        for i in range(p):
            cor_var[:,i] = acp_out.components_[i,:] * np.sqrt((n-1)/n * acp_out.explained_variance_)[i]
        an = np.linspace(0, 2 * np.pi, 100)

        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(ech_cer, ech_cer))
        plt.plot(np.cos(an), np.sin(an))
        ax.axhline(y=0)
        ax.axvline(x=0)
        for i in range(p):
            ax.arrow(0,
                     0,
                     cor_var[i, cp_x-1],
                     cor_var[i, cp_y-1],
                     head_width=0.03,
                     head_length=0.03,
                     length_includes_head=True,
                     color='black') 
            ax.text(cor_var[i, cp_x-1] + 0.01,
                     cor_var[i, cp_y-1],
                     data_acp.columns.values[i],
                     c='red')
        ax.axis('equal')
        ax.set_xlabel('CP {}'.format(cp_x))
        ax.set_ylabel('CP {}'.format(cp_y))
        ax.set_title('Cercle de corrélations')
        plt.show()
    
    if projection == 'oui':
        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(2 * ech_pr_x, 2 * ech_pr_y))
        ax.scatter(data_acp_out['CP_'+str(cp_x)], data_acp_out['CP_'+str(cp_y)], s=40)
        ax.set_xlim(-ech_pr_x, ech_pr_y)
        ax.set_ylim(-ech_pr_x, ech_pr_y)
        ax.axhline(y=0)
        ax.axvline(x=0)
        ax.set_xlabel('CP {}'.format(cp_x))
        ax.set_ylabel('CP {}'.format(cp_y))
        ax.set_title('Projection des individus')
        if label_indiv == 'oui':
            for i in range(n):
                ax.annotate(data_acp.index[i], (data_acp_out['CP_' + str(cp_x)].iloc[i]+0.1, data_acp_out['CP_' + str(cp_y)].iloc[i]+0.1))
        plt.show()
    
    return acp_out, data_acp_out
