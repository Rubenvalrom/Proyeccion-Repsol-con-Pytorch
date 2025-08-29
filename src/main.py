import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from funciones.utilidades_datos import calcular_spread, crear_secuencias
from modelos.gru.Clase_GRU import modelo_GRU

def cargar_variables_seleccionadas():
    """Carga las variables seleccionadas por SHAP"""
    with open("data/shap/variables.csv", mode="r", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        variables = next(lector)
    return variables

def procesar_datos(df):
    """Aplica las transformaciones de ingeniería de características"""
    # Spread
    df['Spread_petroleo'] = calcular_spread(df['Repsol'], df['Petroleo'])
    df['Spread_ibex_35'] = calcular_spread(df['Repsol'], df['IBEX_35'])
    
    # EMA
    df['EMA_semana_repsol'] = df['Repsol'].ewm(span=5, adjust=False).mean()
       
    return df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar variables seleccionadas
    variables_seleccionadas = cargar_variables_seleccionadas()
    
    # Cargar modelo entrenado
    model = torch.load('modelos/gru/forecast_repsol.pt', map_location=device)
    model.eval()
    
    # Cargar scalers
    scaler_X = joblib.load('modelos/scalers/scaler_x.pkl')
    scaler_y = joblib.load('modelos/scalers/scaler_y.pkl')
    
    # Cargar datos limpios
    repsol = pd.read_csv('data/clean/repsol_clean.csv', index_col=0)
    petroleo = pd.read_csv('data/clean/petroleo_clean.csv', index_col=0)
    ibex_35 = pd.read_csv('data/clean/ibex_35_clean.csv', index_col=0)

    # Crear dataframe base
    df = pd.DataFrame({
        'Repsol': repsol['Close'],
        'Petroleo': petroleo['Close_Euro'],
        'IBEX_35': ibex_35['Close']
    })
    print(f"Datos cargados. Número de filas: {len(df)}")
    print(f"Número de nulos antes de limpiar: {df.isnull().sum().sum()}")

    df.index = pd.to_datetime(df.index)
    df.interpolate(method='time', inplace=True)
    print(f"Número de nulos después de limpiar: {df.isnull().sum().sum()}")

    # Aplicar transformaciones
    df = procesar_datos(df)
    df.dropna(inplace=True)

    print(f'Columnas del DataFrame: {sorted(df.columns)}')

    # Seleccionar solo variables relevantes
    assert(sorted(df.columns) == sorted(variables_seleccionadas))
    
    # Cambiar el orden de las columnas
    df = df[variables_seleccionadas]

    # Escalar datos
    X_scaled = scaler_X.transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=df.columns)

    # Crear secuencias
    seq_len = 3
    X_sequences = crear_secuencias(X_scaled_df, seq_length=seq_len)
    
    # Convertir a tensor
    X_tensor = torch.from_numpy(X_sequences).float().to(device)
    
    # Realizar predicciones
    with torch.inference_mode():
        y_pred_scaled = model(X_tensor).squeeze().cpu().numpy()
    
    # Desescalar predicciones
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Crear índice para las predicciones
    pred_index = df.index[seq_len:]

    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'Fecha': pred_index,
        'Precio_Actual': df.loc[pred_index, 'Repsol'].values,
        'Prediccion_Semana': y_pred
    })
    
    # Calcular error porcentual
    resultados['Error_Porcentual'] = ((resultados['Prediccion_Semana'] - resultados['Precio_Actual']) / resultados['Precio_Actual']) * 100
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f'data/predicciones/predicciones_{timestamp}.csv'
    resultados.to_csv(nombre_archivo, index=False)
    
    # Imprimir resultados
    print(f"Predicciones guardadas en: {nombre_archivo}")
    print("\nÚltimas 10 predicciones:")
    print(resultados.tail(10).to_string(index=False))
    
    # Estadísticas de rendimiento
    mae = np.mean(np.abs(resultados['Error_Porcentual']))
    print(f"\nMAE (Error Absoluto Medio): {mae:.4f}%")
    print(f"Predicciones realizadas: {len(resultados)}")

if __name__ == "__main__":
    main()
