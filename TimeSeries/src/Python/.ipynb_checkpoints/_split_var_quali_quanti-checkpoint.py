def _split_var_quanti_quali(df):
    """
    takes : 
    
    [pd.DataFrame] dataframe 
    
    returns : 
    
    [list] var_quali : covariables qualitatives
    [list] var_quanti : covariables quantitatives
    """

    # Init
    var_quali, var_quanti = df.select_dtypes(include=['object', 'category']), df.select_dtypes(include=['int64', 'float64'])
    
    # Set index
    var_quali = var_quali.set_index(var_quali.columns[0])
    var_quanti = var_quanti.set_index(var_quanti.columns[0])
    
    return var_quali, var_quanti