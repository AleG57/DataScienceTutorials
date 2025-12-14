import pandas as pd

def InverseDifferenciation(forecast_diff, original_series, seasonal_periods=12, d=1, D=1):
    """
    Reconstruit les prévisions SARIMA de la série non-stationnaire à partir des prévisions de la série non-stationnaire.
    
    Args:
        forecast_diff (pd.Series): Prévisions dans l'espace différencié.
        original_series (pd.Series): Série d'origine avant modélisation.
        seasonal_periods (int): Période de saisonnalité (ex. 12 pour des données mensuelles).
        d (int): Ordre de différenciation simple.
        D (int): Ordre de différenciation saisonnière.
    
    Returns:
        pd.Series: Prévisions reconstruites dans l’échelle d’origine.
    """
    if D > 0:
        last_seasonal_vals = list(original_series[-seasonal_periods:].values)
    else:
        last_seasonal_vals = [0] * seasonal_periods  # si pas de saisonnalité

    if d > 0:
        last_val_d = original_series.iloc[-1]
    else:
        last_val_d = 0  # pas de diff simple

    reconstructed = []
    for i in range(len(forecast_diff)):
        val = forecast_diff.iloc[i]
        
        if D > 0:
            seasonal_lag = i - seasonal_periods if i >= seasonal_periods else -seasonal_periods + i
            val += last_seasonal_vals[seasonal_lag]
        
        if d > 0:
            if i == 0:
                val += last_val_d
            else:
                val += reconstructed[-1]
        
        reconstructed.append(val)

    start_date = original_series.index[-1] + pd.offsets.MonthEnd(1)
    return pd.Series(reconstructed, index=pd.date_range(start=start_date, periods=len(forecast_diff), freq='M'))