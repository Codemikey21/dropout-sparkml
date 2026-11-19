"""
Alerta Temprana de Deserción Estudiantil — App de Streamlit
Predice si un estudiante va a desertar (Dropout), seguir matriculado (Enrolled)
o graduarse (Graduate) usando el PipelineModel de Spark ML entrenado en Colab.

Autor: Miguel Ángel Solano Díaz — Universidad Autónoma de Bucaramanga (UNAB)
"""
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO = BASE_DIR / "modelo_final"
RUTA_BANNER = BASE_DIR / "assets" / "banner.png"
AUTOR = "Miguel Ángel Solano Díaz"

st.set_page_config(page_title="Alerta Temprana de Deserción", page_icon="🎓", layout="wide")

COLORES_CLASE = {"Dropout": "#e76f51", "Enrolled": "#e9c46a", "Graduate": "#2a9d8f"}
NOMBRES_CLASE = {
    "Dropout": "Deserción (Dropout)",
    "Enrolled": "Sigue matriculado (Enrolled)",
    "Graduate": "Graduado (Graduate)",
}
MENSAJES_CLASE = {
    "Dropout": "Alto riesgo de abandono. Se recomienda contactar al estudiante y activar el acompañamiento académico y financiero.",
    "Enrolled": "Riesgo intermedio: el estudiante sigue activo pero va atrasado. Conviene hacer seguimiento a su avance.",
    "Graduate": "Trayectoria favorable: el estudiante tiene un perfil típico de graduación a tiempo.",
}

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f4f7f6 0%, #ffffff 40%); }
    h1, h2, h3 { color: #1d3557; }
    .tarjeta-resultado {
        border-radius: 16px; padding: 1.4rem 1.6rem; color: #ffffff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12); margin-bottom: 1rem;
    }
    .tarjeta-resultado .etiqueta { font-size: 0.95rem; opacity: 0.9; letter-spacing: 0.04em; text-transform: uppercase; }
    .tarjeta-resultado .clase { font-size: 2.1rem; font-weight: 800; margin: 0.2rem 0; }
    .tarjeta-resultado .prob { font-size: 1.05rem; }
    .pie-pagina {
        text-align: center; color: #5c677d; padding: 1.2rem 0 0.5rem 0;
        border-top: 1px solid #dfe5ec; margin-top: 2rem; font-size: 0.95rem;
    }
    div[data-testid="stForm"] { background: #ffffff; border-radius: 14px; border: 1px solid #dfe5ec; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Diccionarios de códigos -> descripción (documentación oficial del dataset)
# El dataset tiene dos versiones: la original (códigos como 9254) y la de
# Kaggle thedevastator (códigos consecutivos 1, 2, 3...). Se usan según el modelo.
# ---------------------------------------------------------------------------
CURSOS_V1 = {
    1: "Tecnologías de producción de biocombustibles", 2: "Diseño de animación y multimedia",
    3: "Servicio social (nocturno)", 4: "Agronomía", 5: "Diseño de comunicación",
    6: "Enfermería veterinaria", 7: "Ingeniería informática", 8: "Equinocultura",
    9: "Gestión / Administración", 10: "Servicio social", 11: "Turismo", 12: "Enfermería",
    13: "Higiene oral", 14: "Publicidad y marketing", 15: "Periodismo y comunicación",
    16: "Educación básica", 17: "Gestión / Administración (nocturno)",
}
CURSOS_V2 = dict(zip([33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500,
                      9556, 9670, 9773, 9853, 9991], CURSOS_V1.values()))

MODOS_V1 = {
    1: "1.ª fase – contingente general", 2: "Ordenanza n.º 612/93",
    3: "1.ª fase – contingente especial (Azores)", 4: "Titulado de otro curso superior",
    5: "Ordenanza n.º 854-B/99", 6: "Estudiante internacional (pregrado)",
    7: "1.ª fase – contingente especial (Madeira)", 8: "2.ª fase – contingente general",
    9: "3.ª fase – contingente general", 10: "Ordenanza n.º 533-A/99 b2 (otro plan)",
    11: "Ordenanza n.º 533-A/99 b3 (otra institución)", 12: "Mayores de 23 años",
    13: "Transferencia", 14: "Cambio de carrera", 15: "Diploma de especialización tecnológica",
    16: "Cambio de institución/carrera", 17: "Diploma de ciclo corto",
    18: "Cambio de institución/carrera (internacional)",
}
MODOS_V2 = dict(zip([1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57], MODOS_V1.values()))

ETIQUETAS = {
    "application_mode": "Modalidad de ingreso",
    "course": "Carrera",
    "mothers_qualification": "Nivel educativo de la madre (código)",
    "fathers_qualification": "Nivel educativo del padre (código)",
    "mothers_occupation": "Ocupación de la madre (código)",
    "fathers_occupation": "Ocupación del padre (código)",
    "marital_status": "Estado civil (código)",
    "previous_qualification": "Formación previa (código)",
    "nacionality": "Nacionalidad (código)",
    "daytime_evening_attendance": "Jornada diurna",
    "displaced": "Desplazado",
    "educational_special_needs": "Necesidades educativas especiales",
    "debtor": "¿Tiene deudas con la universidad?",
    "tuition_fees_up_to_date": "¿Matrícula al día?",
    "gender": "Género masculino",
    "scholarship_holder": "¿Es becado?",
    "international": "Estudiante internacional",
    "age_at_enrollment": "Edad al ingresar",
    "application_order": "Orden de preferencia de la carrera (0 = 1.ª opción)",
    "previous_qualification_grade": "Nota de la formación previa (0–200)",
    "admission_grade": "Nota de admisión (0–200)",
    "curricular_units_1st_sem_credited": "Materias convalidadas",
    "curricular_units_1st_sem_enrolled": "Materias inscritas",
    "curricular_units_1st_sem_evaluations": "Evaluaciones presentadas",
    "curricular_units_1st_sem_approved": "Materias aprobadas",
    "curricular_units_1st_sem_grade": "Nota promedio (0–20)",
    "curricular_units_1st_sem_without_evaluations": "Materias sin evaluar",
    "curricular_units_2nd_sem_credited": "Materias convalidadas",
    "curricular_units_2nd_sem_enrolled": "Materias inscritas",
    "curricular_units_2nd_sem_evaluations": "Evaluaciones presentadas",
    "curricular_units_2nd_sem_approved": "Materias aprobadas",
    "curricular_units_2nd_sem_grade": "Nota promedio (0–20)",
    "curricular_units_2nd_sem_without_evaluations": "Materias sin evaluar",
    "unemployment_rate": "Tasa de desempleo (%)",
    "inflation_rate": "Tasa de inflación (%)",
    "gdp": "PIB (variación %)",
}
BINARIAS = {"daytime_evening_attendance", "displaced", "educational_special_needs", "debtor",
            "tuition_fees_up_to_date", "gender", "scholarship_holder", "international"}
VALORES_DEFECTO = {
    "age_at_enrollment": 20, "application_order": 1, "previous_qualification_grade": 130.0,
    "admission_grade": 125.0, "curricular_units_1st_sem_enrolled": 6,
    "curricular_units_1st_sem_evaluations": 8, "curricular_units_1st_sem_approved": 5,
    "curricular_units_1st_sem_grade": 12.5, "curricular_units_2nd_sem_enrolled": 6,
    "curricular_units_2nd_sem_evaluations": 8, "curricular_units_2nd_sem_approved": 5,
    "curricular_units_2nd_sem_grade": 12.5, "unemployment_rate": 11.0, "inflation_rate": 1.2,
    "gdp": 0.3, "tuition_fees_up_to_date": 1, "daytime_evening_attendance": 1,
}
LIMITES = {
    "age_at_enrollment": (15, 80), "application_order": (0, 9),
    "previous_qualification_grade": (0.0, 200.0), "admission_grade": (0.0, 200.0),
    "unemployment_rate": (0.0, 30.0), "inflation_rate": (-5.0, 10.0), "gdp": (-10.0, 10.0),
}


# ---------------------------------------------------------------------------
# (a) Spark y modelo en memoria con @st.cache_resource
# ---------------------------------------------------------------------------
def _configurar_java():
    # En Streamlit Cloud, packages.txt (default-jre) instala Java; se define JAVA_HOME si falta.
    if os.environ.get("JAVA_HOME"):
        return
    java = shutil.which("java")
    if java:
        os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)


@st.cache_resource(show_spinner="Iniciando Spark (la primera vez tarda unos segundos)...")
def obtener_spark():
    _configurar_java()
    from pyspark.sql import SparkSession

    return (SparkSession.builder
            .appName("app_desercion_streamlit")
            .master("local[1]")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def cargar_modelo():
    from pyspark.ml import PipelineModel

    obtener_spark()
    modelo = PipelineModel.load(str(RUTA_MODELO))
    labels = json.loads((RUTA_MODELO / "labels.json").read_text(encoding="utf-8"))
    schema = json.loads((RUTA_MODELO / "schema_entrada.json").read_text(encoding="utf-8"))

    # Códigos válidos de cada categórica, leídos del StringIndexer guardado en el pipeline
    indexador = modelo.stages[0]
    categorias = {col: sorted(int(float(v)) for v in valores)
                  for col, valores in zip(indexador.getInputCols(), indexador.labelsArray)}
    nombre_clasificador = type(modelo.stages[-1]).__name__.replace("ClassificationModel", "").replace("Model", "")
    return modelo, labels, schema, categorias, nombre_clasificador


# ---------------------------------------------------------------------------
# (c) Variables derivadas: MISMA función que en el notebook (tarea 3)
# ---------------------------------------------------------------------------
def division_segura(numerador, denominador):
    from pyspark.sql import functions as F
    # Devuelve numerador/denominador, o 0.0 cuando el denominador es 0 (control de división por cero)
    return F.when(denominador > 0, numerador / denominador).otherwise(F.lit(0.0))


def agregar_variables_derivadas(df):
    from pyspark.sql import functions as F
    c = F.col
    df = df.withColumn("tasa_aprob_1sem", division_segura(c("curricular_units_1st_sem_approved"),
                                                          c("curricular_units_1st_sem_enrolled")))
    df = df.withColumn("tasa_aprob_2sem", division_segura(c("curricular_units_2nd_sem_approved"),
                                                          c("curricular_units_2nd_sem_enrolled")))
    df = df.withColumn("variacion_nota", (c("curricular_units_2nd_sem_grade")
                                          - c("curricular_units_1st_sem_grade")).cast("double"))
    df = df.withColumn("variacion_tasa_aprob", c("tasa_aprob_2sem") - c("tasa_aprob_1sem"))
    df = df.withColumn("eficiencia_evaluaciones", division_segura(
        c("curricular_units_1st_sem_approved") + c("curricular_units_2nd_sem_approved"),
        c("curricular_units_1st_sem_evaluations") + c("curricular_units_2nd_sem_evaluations")))
    return df


# ---------------------------------------------------------------------------
# (d) Predicción: datos del formulario -> DataFrame de Spark -> modelo
# ---------------------------------------------------------------------------
def predecir(modelo, labels, schema, valores):
    from pyspark.sql import types as T

    tipos_spark = {"int": T.IntegerType(), "bigint": T.LongType(), "double": T.DoubleType()}
    columnas = schema["columnas"]
    estructura = T.StructType([T.StructField(c["nombre"], tipos_spark.get(c["tipo"], T.DoubleType()), True)
                               for c in columnas])
    fila = tuple(int(valores[c["nombre"]]) if c["tipo"] in ("int", "bigint") else float(valores[c["nombre"]])
                 for c in columnas)

    spark = obtener_spark()
    df = spark.createDataFrame([fila], schema=estructura)   # tipos exactos del entrenamiento
    df = agregar_variables_derivadas(df)
    resultado = modelo.transform(df).select("prediction", "probability", *schema["variables_derivadas"]).first()

    clase = labels[int(resultado["prediction"])]            # índice -> nombre (orden del StringIndexer)
    probabilidades = dict(zip(labels, resultado["probability"].toArray().tolist()))
    derivadas = {v: resultado[v] for v in schema["variables_derivadas"]}
    return clase, probabilidades, derivadas


# ---------------------------------------------------------------------------
# Widgets del formulario
# ---------------------------------------------------------------------------
def describir_codigo(columna, codigo, version_v2):
    if columna == "course":
        return (CURSOS_V2 if version_v2 else CURSOS_V1).get(codigo, f"Código {codigo}")
    if columna == "application_mode":
        return (MODOS_V2 if version_v2 else MODOS_V1).get(codigo, f"Código {codigo}")
    return f"Código {codigo}"


def widget(columna, tipo, categorias, version_v2, contenedor):
    etiqueta = ETIQUETAS.get(columna, columna.replace("_", " ").capitalize())
    if columna in BINARIAS:
        defecto = VALORES_DEFECTO.get(columna, 0)
        opcion = contenedor.radio(etiqueta, ["Sí", "No"], index=0 if defecto == 1 else 1,
                                  horizontal=True, key=columna)
        return 1 if opcion == "Sí" else 0
    if columna in categorias:
        codigos = categorias[columna]
        return contenedor.selectbox(etiqueta, codigos, key=columna,
                                    format_func=lambda x: f"{describir_codigo(columna, x, version_v2)}"
                                    if columna in ("course", "application_mode") else f"Código {x}")
    if tipo in ("int", "bigint"):
        minimo, maximo = LIMITES.get(columna, (0, 60))
        return contenedor.number_input(etiqueta, min_value=int(minimo), max_value=int(maximo),
                                       value=int(VALORES_DEFECTO.get(columna, 0)), step=1, key=columna)
    minimo, maximo = LIMITES.get(columna, (0.0, 20.0))
    return contenedor.number_input(etiqueta, min_value=float(minimo), max_value=float(maximo),
                                   value=float(VALORES_DEFECTO.get(columna, 0.0)), step=0.1,
                                   format="%.2f", key=columna)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
if RUTA_BANNER.exists():
    st.image(str(RUTA_BANNER), width="stretch")
st.title("🎓 Alerta Temprana de Deserción Estudiantil")
st.markdown("Predice con **Spark ML** si un estudiante va a **desertar**, **seguir matriculado** "
            "o **graduarse**, a partir de su información de ingreso y de sus dos primeros semestres.")

with st.sidebar:
    st.header("📋 Instrucciones de uso")
    st.markdown(
        """
        1. Completa los datos del estudiante en las tres secciones del formulario:
           **Perfil**, **Familia** y **Rendimiento académico**.
        2. En cada semestre, las materias **aprobadas** no pueden superar a las **inscritas**
           ni a las **evaluaciones presentadas**.
        3. Presiona **Predecir**.
        4. Revisa la **clase predicha** (tarjeta de color) y la **probabilidad de cada clase**
           en la tabla y el gráfico.

        **Clases posibles**
        - 🟥 **Dropout:** abandona la carrera.
        - 🟨 **Enrolled:** sigue matriculado, pero no se gradúa a tiempo.
        - 🟩 **Graduate:** se gradúa en el tiempo normal.
        """
    )
    st.caption("Las variables derivadas (tasas de aprobación, variación de la nota y eficiencia) "
               "se calculan automáticamente dentro de la app.")

try:
    modelo, labels, schema, categorias, nombre_modelo = cargar_modelo()
except Exception as error:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo desde '{RUTA_MODELO.name}/'. Detalle: {error}")
    st.stop()

with st.sidebar:
    st.divider()
    st.markdown(f"**Modelo:** {nombre_modelo} (Spark ML {schema.get('pyspark_version', '')})")
    st.markdown(f"**Variables de entrada:** {len(schema['columnas'])}")

columnas_schema = {c["nombre"]: c["tipo"] for c in schema["columnas"]}
version_v2 = max(categorias.get("course", [0])) > 100
semestre1 = [c for c in columnas_schema if c.startswith("curricular_units_1st_sem")]
semestre2 = [c for c in columnas_schema if c.startswith("curricular_units_2nd_sem")]
familia = [c for c in columnas_schema if c.startswith(("mothers_", "fathers_"))]
macro = [c for c in columnas_schema if c in ("unemployment_rate", "inflation_rate", "gdp")]
perfil = [c for c in columnas_schema if c not in semestre1 + semestre2 + familia + macro]

# (b) Formulario con todas las variables que usa el modelo y botón Predecir
valores = {}
with st.form("formulario_estudiante"):
    tab_perfil, tab_familia, tab_academico = st.tabs(["👤 Perfil", "👨‍👩‍👧 Familia", "📚 Rendimiento académico"])
    with tab_perfil:
        cols = st.columns(2)
        for i, col in enumerate(perfil + macro):
            valores[col] = widget(col, columnas_schema[col], categorias, version_v2, cols[i % 2])
    with tab_familia:
        cols = st.columns(2)
        for i, col in enumerate(familia):
            valores[col] = widget(col, columnas_schema[col], categorias, version_v2, cols[i % 2])
        st.caption("Los códigos de formación y ocupación corresponden a la clasificación oficial del dataset.")
    with tab_academico:
        c1, c2 = st.columns(2)
        c1.subheader("1.er semestre")
        c2.subheader("2.º semestre")
        for col in semestre1:
            valores[col] = widget(col, columnas_schema[col], categorias, version_v2, c1)
        for col in semestre2:
            valores[col] = widget(col, columnas_schema[col], categorias, version_v2, c2)
    enviado = st.form_submit_button("🔮 Predecir", type="primary", width="stretch")

if enviado:
    errores = []
    for s, nombre in (("1st", "1.er"), ("2nd", "2.º")):
        aprobadas = valores.get(f"curricular_units_{s}_sem_approved", 0)
        if aprobadas > valores.get(f"curricular_units_{s}_sem_enrolled", aprobadas):
            errores.append(f"{nombre} semestre: las materias aprobadas superan a las inscritas.")
        if aprobadas > valores.get(f"curricular_units_{s}_sem_evaluations", aprobadas):
            errores.append(f"{nombre} semestre: las materias aprobadas superan a las evaluaciones presentadas.")
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    with st.spinner("Calculando la predicción con Spark..."):
        clase, probabilidades, derivadas = predecir(modelo, labels, schema, valores)

    # (e) Clase predicha destacada + probabilidad de cada clase (tabla y gráfico)
    st.header("Resultado de la predicción")
    izq, der = st.columns([1, 1.3])
    with izq:
        st.markdown(
            f"""
            <div class="tarjeta-resultado" style="background:{COLORES_CLASE.get(clase, '#264653')}">
                <div class="etiqueta">Clase predicha</div>
                <div class="clase">{NOMBRES_CLASE.get(clase, clase)}</div>
                <div class="prob">Probabilidad: <b>{probabilidades[clase] * 100:.1f} %</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(MENSAJES_CLASE.get(clase, ""))
    with der:
        tabla = pd.DataFrame({"Clase": list(probabilidades.keys()),
                              "Probabilidad": list(probabilidades.values())})
        tabla = tabla.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
        import altair as alt
        grafico = (alt.Chart(tabla)
                   .mark_bar(cornerRadiusEnd=6)
                   .encode(x=alt.X("Probabilidad:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                           y=alt.Y("Clase:N", sort="-x", title=None),
                           color=alt.Color("Clase:N", legend=None,
                                           scale=alt.Scale(domain=list(COLORES_CLASE.keys()),
                                                           range=list(COLORES_CLASE.values()))),
                           tooltip=["Clase", alt.Tooltip("Probabilidad:Q", format=".1%")])
                   .properties(height=180, title="Probabilidad por clase"))
        st.altair_chart(grafico, width="stretch")
        tabla_vista = tabla.copy()
        tabla_vista["Probabilidad"] = (tabla_vista["Probabilidad"] * 100).map("{:.2f} %".format)
        st.dataframe(tabla_vista, hide_index=True, width="stretch")

    with st.expander("Ver las variables derivadas calculadas por la app"):
        st.dataframe(pd.DataFrame({"Variable": list(derivadas.keys()),
                                   "Valor": [round(float(v), 4) for v in derivadas.values()]}),
                     hide_index=True, width="stretch")

st.markdown(
    f"<div class='pie-pagina'>Desarrollado por <b>{AUTOR}</b> · Examen de Big Data – Spark ML · "
    f"Universidad Autónoma de Bucaramanga (UNAB)</div>",
    unsafe_allow_html=True,
)
