# 🎓 Alerta Temprana de Deserción Estudiantil — Spark ML + Streamlit

Aplicación web que predice si un estudiante universitario va a **desertar (Dropout)**, **seguir matriculado sin graduarse a tiempo (Enrolled)** o **graduarse (Graduate)**, usando un modelo de clasificación multiclase entrenado con **Spark ML**.

🔗 **App desplegada:** https://dropout-sparkml-mhwghwptazajzmsycfj4aj.streamlit.app/

![Banner](assets/banner.png)

## Descripción

- **Dataset:** *Predict Students' Dropout and Academic Success* (Kaggle / UCI), con 4.424 estudiantes y variables demográficas, socioeconómicas y de rendimiento académico.
- **Notebook (Google Colab):** carga y exploración, limpieza, 5 variables derivadas (tasas de aprobación por semestre, variación de la nota, variación de la tasa de aprobación y eficiencia en evaluaciones), selección de variables con `featureImportances`, pipelines completos (`StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `StandardScaler` → clasificador), validación cruzada con `ParamGridBuilder` + `CrossValidator` para **Regresión Logística multinomial, Random Forest y Árbol de Decisión**, y evaluación en el conjunto de prueba.
- **Desbalance de clases:** pesos por clase (`weightCol`) y F1 ponderado como métrica de selección.
- **Modelo final:** el mejor `PipelineModel`, guardado en `modelo_final/` junto con `labels.json` (orden de clases) y `schema_entrada.json` (tipos de las variables de entrada).

La app:
1. Carga el modelo con `PipelineModel.load` y mantiene Spark y el modelo en memoria con `@st.cache_resource`.
2. Muestra un formulario (`st.form`) con todas las variables que usa el modelo y el botón **Predecir**.
3. Calcula dentro de la app las mismas variables derivadas del notebook.
4. Convierte los datos en un DataFrame de Spark y aplica el modelo.
5. Muestra la clase predicha, traducida con el orden de `labels` del `StringIndexer`, y la probabilidad de cada clase en una tabla y un gráfico.

## Estructura del repositorio

```
├── app.py                    # Aplicación de Streamlit
├── modelo_final/             # PipelineModel de Spark ML + labels.json + schema_entrada.json
├── modelo_final.zip          # Carpeta del modelo comprimida (entregable)
├── assets/banner.png         # Banner de la app
├── notebook/                 # Notebook ejecutado de Google Colab
├── capturas/                 # Capturas de la app funcionando
├── .streamlit/config.toml    # Tema visual
├── requirements.txt          # streamlit, pyspark, pandas
└── packages.txt              # default-jre (Java para Spark)
```

## Cómo ejecutarla localmente

Requisitos: Python 3.10 a 3.12 y **Java 17 o superior** instalado (Spark lo necesita).

```bash
git clone https://github.com/Codemikey21/dropout-sparkml.git
cd dropout-sparkml
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io): **Create app** → elegir el repositorio, la rama `main` y el archivo `app.py`.
3. En **Advanced settings**, elegir **Python 3.12**.
4. `packages.txt` instala Java (`default-jre`) y `requirements.txt` instala PySpark 4.0.5, la misma versión con la que se entrenó el modelo.

## Autor

**Miguel Ángel Solano Díaz** — Ingeniería de Sistemas, Universidad Autónoma de Bucaramanga (UNAB)
