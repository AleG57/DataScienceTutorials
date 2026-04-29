import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gc

def missing_values_summary(df):

    # -------------- Init --------------#
    dtypes = df.select_dtypes(include=[np.number]).columns.tolist()
    
    nan_counts = df[dtypes].isna().sum()
    zero_counts = (df[dtypes] == 0).sum()
    total_len = len(df)

    # -------------- Summary --------------#
    summary = pd.DataFrame({
        "Nb NaN (Manquant)": nan_counts,
        "% NaN": (nan_counts / total_len).round(4) * 100,
        "Nb zéros": zero_counts,
        "% zéros": (zero_counts / total_len).round(4) * 100,
        "Total Lignes": total_len
    })
    
    return summary.sort_values(by="Nb NaN (Manquant)", ascending=False)

def missing_values_heatmap(
    df,
    method='nan', 
    figname = "MissingValues.png"
):
        
    # -------------- Init --------------#
    date_labels = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    nb_labels_desires = 20
    step = max(1, int(len(df) / nb_labels_desires))
    plt.figure(figsize=(12, 8))
    
    if method == 'zeros':

        data_to_plot = (df == 0).set_index(date_labels)
        title = "Missing values (Zeros)"
        label_legend = '1 : value is 0 | 0 : value != 0'
    
    elif method == 'nan':

        data_to_plot = df.isna().set_index(date_labels)
        title = "Missing values (NaN)"
        label_legend = '1 : missing value (NaN) | 0 : != NaN values'

    else: 
        return "Error : unknown method. method = {'zeros', 'nan'}"
        
    # -------------- Heatmap --------------#
    sns_plot = sns.heatmap(
            data_to_plot.astype(float), 
            yticklabels=step,
            cbar=True,
            cmap=sns.color_palette(["#440154", "#fde725"], as_cmap=True), 
            vmin=0,
            vmax=1,
            cbar_kws={'label': label_legend}
        )
    
    sns_plot.set_title(title, fontsize=18, weight='bold')
    sns_plot.set_xlabel("Features (variables)", fontsize=14)
    sns_plot.set_ylabel("Date", fontsize=14)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0)
    
    # -------------- Save --------------#
    fig = sns_plot.get_figure()
    fig.savefig(figname) 
    plt.show()

    plt.close('all')
    gc.collect()

    return None

def status_heatmap(
    df,
    figname="DataStatus_Heatmap.png"
):
    # -------------- Init --------------#
    date_labels = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    nb_labels_desires = 20
    step = max(1, int(len(df) / nb_labels_desires))
    
    plt.figure(figsize=(15, 8))
    
    # -------------- Status matrix --------------#
    status_matrix = pd.DataFrame(0, index=df.index, columns=df.columns)
    
    status_matrix[df == 0] = 1                           # Maintenance / stop
    status_matrix[df.isna()] = 2                         # NaN values
    status_matrix = status_matrix.set_index(date_labels) # Date labels

    # -------------- Colors --------------#
    # Couleur 0 : clean data
    # Couleur 1 : (blue) stop / maintenance
    # Couleur 2 : (red) lost data
    colors = ["#f0f0f0", "#21918c", "#d62728"] 
    cmap = ListedColormap(colors)
    
    # -------------- Heatmap --------------#
    ax = sns.heatmap(
        status_matrix,
        yticklabels=step,
        cbar=True,
        cmap=cmap,
        vmin=0, 
        vmax=3,
    )
    
    # -------------- Legend (Colorbar) --------------#
    colorbar = ax.collections[0].colorbar
    # On place les ticks au milieu de chaque couleur
    colorbar.set_ticks([0.5, 1.5, 2.5])
    colorbar.set_ticklabels([
        'Clean data', 
        'Zeros (0)', 
        'Data loss (NaN)'
    ])
    
    # -------------- Ttiles and Labels --------------#
    ax.set_title("État des variables en fonction du temps", fontsize=18, weight='bold')
    ax.set_xlabel("Features (variables)", fontsize=14)
    ax.set_ylabel("Date", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # -------------- Save --------------#
    fig = ax.get_figure()
    # Création du dossier si nécessaire
    # import os; os.makedirs("figs/01/", exist_ok=True)
    fig.savefig(figname, bbox_inches='tight')
    plt.show()

    plt.close('all')
    gc.collect()

    return None

# Handle missing value by replacing NaN values to -1 (by default) and 
# by creating new feature as a flag : is a missing value or not (boolean) 
    
def handle_missing_values_with_flag(df, fill_value=-1, exclude_cols=None):

    # -------------- Init --------------#
    df_clean = df.copy()
    
    if exclude_cols is None:
        exclude_cols = []
    
    # -------------- NaN values on numerical columns --------------#
    cols_with_nan = [
        col for col in df_clean.columns 
        if df_clean[col].isna().any() 
        and pd.api.types.is_numeric_dtype(df_clean[col])
        and col not in exclude_cols
    ]
    
    # -------------- Creating flag --------------#
    for col in cols_with_nan:
        flag_col_name = f"is_missing_{col}"
        df_clean[flag_col_name] = df_clean[col].isna().astype('uint8') # Memory usage
        df_clean[col] = df_clean[col].fillna(fill_value)
        
    return df_clean