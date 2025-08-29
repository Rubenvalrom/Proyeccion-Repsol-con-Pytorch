def calcular_rsi(serie, periodo=14):
    """
    Parámetros:
    serie (pd.Series): Serie de pandas con valores numéricos.
    periodo (int): Número de periodos para el cálculo del RSI.

    Devuelve:
    pd.Series: Serie con el RSI calculado.

    Calcula el Relative Strength Index (RSI) de una serie temporal.
    """    
    delta = serie.diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    
    rs = ganancia / perdida
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
    
def calcular_spread(serie_1, serie_2):
    """
    Parámetros:
    serie_1 (pd.Series): La primera serie temporal.
    serie_2 (pd.Series): La segunda serie temporal.

    Devuelve:
    spread (pd.Series): los residuos de la regresión lineal.
    """
    from sklearn.linear_model import LinearRegression

    X = serie_2.values.reshape(-1, 1)
    y = serie_1.values

    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]

    spread = serie_1 - beta * serie_2

    return spread

def crear_secuencias(df, seq_length):
    """
    Parámetros:
    - df (pd.DataFrame): Datos de entrada
    - seq_length (int): Longitud de la secuencia
    Devuelve:
    - x (np.ndarray): Array de datos secuenciales
    """
    import numpy as np

    x = []

    for i in range(len(df) - (seq_length)):
        
        x_secuencial = df.iloc[i:(i+seq_length)].values
        x.append(x_secuencial)

    return np.array(x)