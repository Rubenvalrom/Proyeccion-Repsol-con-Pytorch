# Proyección del Precio de la Acción de Repsol con Modelos GRU y SHAP

Proyecto para la proyección semanal (horizonte corto) del precio de la acción de Repsol mediante una red recurrente GRU (Gated Recurrent Unit) y un flujo de trabajo que integra: ingestión de datos, limpieza, ingeniería de características, escalado, generación de secuencias temporales, búsqueda de hiperparámetros, selección de variables con SHAP, persistencia de artefactos y predicción reproducible.

## Objetivos

1. Consolidar series temporales de fuentes financieras (acción Repsol, IBEX 35, precio del petróleo, divisa EUR/USD, etc.).
2. Generar variables derivadas (spreads, medias móviles exponenciales) que capturen relaciones y microestructura.
3. Entrenar un modelo GRU para pronosticar el precio (horizonte de 1 semana / próximo punto temporal tras una ventana corta).
4. Optimizar el modelo mediante búsqueda (grid search) y seleccionar atributos relevantes con interpretabilidad SHAP.
5. Producir un pipeline de inferencia reproducible que emplee artefactos serializados (scalers, modelo, variables seleccionadas) y registre resultados.

## Alcance y Advertencia

El proyecto tiene fines educativos y exploratorios. No constituye recomendación de inversión. Las predicciones están sujetas a incertidumbre y a cambios estructurales del mercado.

## Estructura de Directorios

Árbol lógico (raíz del repositorio):

```text
.
├── LICENSE                       # Licencia MIT
├── README.md                     # Documentación principal
├── requirements.txt              # Dependencias del entorno
└── src/
	├── main.py                   # Script de inferencia / generación de predicciones
	├── notebook.ipynb            # Exploración y análisis interactivo
	├── funciones/
	│   ├── __init__.py
	│   └── utilidades_datos.py   # Utilidades: spread, RSI, creación de secuencias
	├── modelos/
	│   ├── gru/
	│   │   ├── __init__.py
	│   │   ├── Clase_GRU.py      # Definición de la arquitectura GRU
	│   │   ├── forecast_repsol.pt# Modelo entrenado (inferencias)
	│   │   └── gru_sel.pt        # Modelo utilizado para selección de variables con SHAP
	│   ├── scalers/
	│   │   ├── scaler_x.pkl      # Escalador de variables independientes
	│   │   └── scaler_y.pkl      # Escalador de variable objetivo
	│   └── shap/
	│       └── explainer_shap.pkl# Objeto explicador SHAP serializado
	└── data/
		├── raw/                  # Datos brutos (ingesta original)
		│   ├── repsol_raw.csv
		│   ├── repsol_yahoo.csv
		│   ├── ibex_35_raw.csv
		│   ├── petroleo_raw.csv
		│   └── dolar_euro.csv
		├── clean/                # Datos limpios y consistentes
		│   ├── repsol_clean.csv
		│   ├── ibex_35_clean.csv
		│   ├── petroleo_clean.csv
		│   ├── train_escalado.csv
		│   ├── val_escalado.csv
		│   ├── test_escalado.csv
		│   ├── y_train_escalado.csv
		│   ├── y_val_escalado.csv
		│   └── y_test_escalado.csv
		├── tensor/               # Tensores para entrenamiento, validación y test
		│   ├── X_tensor_train.pt
		│   ├── X_tensor_val.pt
		│   ├── X_tensor_test.pt
		│   ├── y_tensor_train.pt
		│   ├── y_tensor_val.pt
		│   └── y_tensor_test.pt
		├── grid_search/          # Resultados de búsqueda de hiperparámetros
		│   ├── df_search_results.csv
		│   └── df_sel_search_results.csv
		├── shap/                 # Artefactos de selección de variables con SHAP
		│   ├── variables.csv     # Variables seleccionadas finales
		│   └── shap_values.pkl   # Valores SHAP (Tensor tridimensional con los periodos y variables)
		└── predicciones/         # Predicciones generadas por main.py
			└── predicciones_YYYYMMDD_HHMMSS.csv
```

## Flujo de Trabajo (Pipeline)

1. Ingesta: Descarga de series (p.ej. `repsol_raw.csv`, índices, commodities, divisas) desde APIs (yfinance, investpy u otras).
2. Limpieza y Normalización: Tratamiento de valores nulos (interpolación temporal), alineación de índices y formatos en `data/clean`.
3. Ingeniería de Características:
   - Cálculo de spreads (Repsol vs. Petróleo; Repsol vs. IBEX 35) mediante regresión lineal para residuales.
   - Medias móviles exponenciales (EMA semanal) y potencialmente otros indicadores (ej. RSI disponible en utilidades).
4. Escalado: Ajuste de `scaler_x.pkl` (variables independientes) y `scaler_y.pkl` (target) y persistencia para inferencia reproducible.
5. Creación de Secuencias: Conversión de las series escaladas en ventanas temporales (longitud `seq_len=3` en el script actual) con `crear_secuencias`.
6. División del Conjunto: Generación de tensores train / val / test (`data/tensor`).
7. Búsqueda de Hiperparámetros: Exploración sistemática (GRU: capas, unidades ocultas, dropout, bidireccionalidad) con resultados en `grid_search/`.
8. Entrenamiento: Ajuste del modelo GRU (definido en `Clase_GRU.py`) y selección del mejor checkpoint (`forecast_repsol.pt`).
9. Interpretabilidad: Cálculo de contribuciones SHAP, filtrado de variables relevantes y exportación a `variables.csv` (orden esperado por el pipeline de inferencia).
10. Inferencia: Ejecución de `main.py` que reconstruye el conjunto de características, verifica el conjunto de variables, escala, forma secuencias y genera la predicción del siguiente paso temporal.
11. Evaluación: Cálculo de error porcentual y MAE (Error Absoluto Medio) sobre el rango de predicciones reconstruidas.
12. Registro y Persistencia: Exportación de un CSV fechado en `data/predicciones/` para trazabilidad.

## Descripción de Componentes Clave

| Componente | Descripción |
|------------|-------------|
| `utilidades_datos.py` | Funciones auxiliares para cálculo de RSI, spreads y creación de secuencias (ventanas deslizantes). |
| `Clase_GRU.py` | Arquitectura GRU configurable: bidireccional opcional, BatchNorm, capa totalmente conectada con activación ReLU y dropout. |
| `variables.csv` | Lista ordenada de variables finales utilizadas durante la inferencia (control de consistencia). |
| `forecast_repsol.pt` | Modelo entrenado final listo para predicción. |
| `explainer_shap.pkl` | Objeto SHAP para reanálisis interpretativo. |
| `grid_search/*.csv` | Métricas consolidadas de exploración de hiperparámetros y selección. |
| `predicciones_*.csv` | Archivo de salida con columnas Fecha, Precio_Actual, Prediccion_Semana, Error_Porcentual. |

## Modelo GRU

La red GRU procesa secuencias de longitud fija (actualmente 3 pasos temporales) y produce una predicción escalar. Características técnicas:

- Bidireccionalidad para capturar patrones de dependencia contextual.
- Normalización por lotes en la salida temporal agregada (`out[:, -1, :]`).
- Capa intermedia densa (linear_size) con ReLU y dropout para regularización.
- Objetivo: estimar el valor futuro del precio de Repsol (escala invertida tras la predicción).

## Interpretabilidad con SHAP

Se emplea SHAP para:

1. Cuantificar contribuciones marginales de cada variable al output.
2. Filtrar y ordenar las variables por importancia agregada.
3. Garantizar reproducibilidad: la inferencia afirma (`assert`) que el conjunto y orden de columnas coincide con `variables.csv` (Los scalers no funcionan con columnas adicionales o en diferente orden).

## Requisitos del Entorno

El archivo `requirements.txt` fija versiones (principalmente Python 3.12 y librerías científicas). Para una instalación más ligera, pueden seleccionarse únicamente dependencias mínimas (ej.: `pandas`, `numpy`, `scikit-learn`, `torch`, `shap`, `joblib`, `yfinance`, `investpy`). Sin embargo, para reproducibilidad exacta se recomienda instalar el listado completo.

## Instalación

Clonar el repositorio oficial:

```bash
git clone https://github.com/Rubenvalrom/Proyeccion-Repsol-con-Pytorch.git
cd Proyeccion-Repsol-con-Pytorch
```

## Datos

Los ficheros bajo `data/raw` no se regeneran automáticamente en el script de inferencia actual. Si se desea reconstruir el pipeline desde origen, será necesario:

1. Descargar/actualizar series históricas (ej. vía yfinance / investpy).
2. Aplicar limpieza y alineación temporal.
3. Recalcular spreads y EMAs.
4. Reajustar escaladores y regenerar tensores.
5. Reentrenar el modelo y volver a calcular SHAP.

Estas etapas pueden implementarse en notebooks o scripts adicionales (no incluidos explícitamente en este repositorio en su forma actual).

## Uso (Inferencia de Predicciones)

Ejecutar el script principal (asumiendo que las rutas relativas se mantienen y que los artefactos ya existen):

```powershell
python src/main.py
```

Salida esperada:

1. Conteo de filas y nulos antes/después de interpolación.
2. Confirmación de columnas y verificación de variables seleccionadas.
3. Generación de archivo en `src/data/predicciones/predicciones_YYYYMMDD_HHMMSS.csv`.
4. Cálculo de MAE porcentual agregado.

## Extensión / Personalización

- Añadir indicadores técnicos: Incorporar cálculos (RSI ya disponible) en `procesar_datos` y actualizar `variables.csv` tras un nuevo análisis SHAP.
- Optimización avanzada: Repetir grid search con métricas adicionales (RMSE, Huber Loss) y agregar validación cruzada temporal (rolling / expanding window), aunque eso necesitaría una mayor potencia computacional.
- Despliegue: Empaquetar el pipeline de inferencia como servicio o tarea programada.

## Buenas Prácticas Implementadas

- Separación de responsabilidades (utilidades, modelo, artefactos).
- Persistencia explícita de escaladores y variables seleccionadas (control de deriva de esquema).
- Interpretabilidad integrada con SHAP para trazabilidad de decisiones del modelo.
- Versionado reproducible vía `requirements.txt`.

## Licencia

Distribuido bajo la Licencia MIT. Véase `LICENSE` para más detalles.
