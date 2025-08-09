# Source : https://www.kaggle.com/code/goyami/steps-for-stepwise-feature-selection

import itertools
import pandas as pd
from statsmodels.api import Logit
from statsmodels.tools import add_constant

# Calculs des AIC et BIC
 
def AIC_BIC_logreg(X, y, features):
    logreg_model = Logit(y, X[list(features)])
    logreg = logreg_model.fit(disp=0)
    AIC = logreg.aic
    BIC = logreg.bic
    return {'model':logreg, 'AIC':AIC, 'BIC':BIC}

# Procédure backward

def logreg_backward_predictors(X, y, predictors, crit='BIC', verbose=True):
    results = []
    
    for combi in itertools.combinations(predictors, len(predictors)-1):
        results.append(AIC_BIC_logreg(X=X, y=y, features=list(combi)+['const']))
    models = pd.DataFrame(results)
    
    if crit == 'AIC':
        best_model = models.loc[models['AIC'].argmin()]
    else:
        best_model = models.loc[models['BIC'].argmin()]
    
    if verbose == True:
        print('Selected predictors:', best_model['model'].model.exog_names, 'AIC:', best_model['model'].aic, 'BIC:', best_model['model'].bic)
    
    return best_model

def logreg_backward_proc(X, y, crit='BIC', verbose=True):
    backward_models = pd.DataFrame(columns=['AIC', 'BIC', 'model'])
    predictors = list(X.columns.difference(['const']))
    
    for i in range(1, len(X.columns.difference(['const']))+1):
        backward_result = logreg_backward_predictors(X=X, y=y, predictors=predictors, crit=crit, verbose=verbose)
        if i > 1:
            if backward_result[crit] > backward_model_before:
                break
        backward_models.loc[i] = backward_result
        predictors = backward_models.loc[i]['model'].model.exog_names
        backward_model_before = backward_models.loc[i][crit]
        predictors = [k for k in predictors if k != 'const']
    
    return(backward_models['model'][len(backward_models['model'])])

# Procédure forward

def logreg_forward_predictors(X, y, predictors, crit='BIC', verbose=True):
    results=[]
    remaining_predictors = [p for p in X.columns.difference(['const']) if p not in predictors]

    for p in remaining_predictors:
        results.append(AIC_BIC_logreg(X=X, y=y, features=predictors+[p]+['const']))
    models = pd.DataFrame(results)
    
    if crit == 'AIC':
        best_model = models.loc[models['AIC'].argmin()]
    else:
        best_model = models.loc[models['BIC'].argmin()]
    
    if verbose==True:
        print('Selected predictors:', best_model['model'].model.exog_names, 'AIC:', best_model['model'].aic, 'BIC:', best_model['model'].bic)
    
    return best_model

def logreg_forward_proc(X, y, crit='BIC', verbose=True):
    forward_models = pd.DataFrame(columns=['AIC', 'BIC', 'model'])
    predictors = []
    
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        forward_result = logreg_forward_predictors(X=X, y=y, predictors=predictors, crit=crit, verbose=verbose)
        if i > 1:
            if forward_result[crit] > forward_model_before:
                break
        forward_models.loc[i] = forward_result
        predictors = forward_models.loc[i]['model'].model.exog_names
        forward_model_before = forward_models.loc[i][crit]
        predictors = [k for k in predictors if k != 'const']
    
    return(forward_models['model'][len(forward_models['model'])])

# Procédure stepwise

def logreg_stepwise_proc(X, y, crit='BIC', verbose=True):
    stepwise_models = pd.DataFrame(columns=['AIC', 'BIC', 'model'])
    predictors = []
    stepwise_model_before = AIC_BIC_logreg(X, y, predictors+['const'])[crit]
    
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        forward_result = logreg_forward_predictors(X=X, y=y, predictors=predictors, crit=crit, verbose=verbose)
        stepwise_models.loc[i] = forward_result
        predictors = stepwise_models.loc[i]['model'].model.exog_names
        predictors = [k for k in predictors if k != 'const']
        backward_result = logreg_backward_predictors(X=X, y=y, predictors=predictors, crit=crit, verbose=verbose)
        if backward_result[crit]< forward_result[crit]:
            stepwise_models.loc[i] = backward_result
            predictors = stepwise_models.loc[i]['model'].model.exog_names
            stepwise_model_before = stepwise_models.loc[i][crit]
            predictors = [k for k in predictors if k != 'const']
        if stepwise_models.loc[i][crit]> stepwise_model_before:
            break
        else:
            stepwise_model_before = stepwise_models.loc[i][crit]
    
    return(stepwise_models['model'][len(stepwise_models['model'])])
