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
RUTA_METRICAS = BASE_DIR / "assets" / "metricas.json"
URL_REPO = "https://github.com/Codemikey21/dropout-sparkml"
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
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dfe5ec; border-radius: 14px;
        padding: 0.8rem 1rem; box-shadow: 0 2px 8px rgba(29,53,87,0.06);
    }
    div[data-testid="stMetricValue"] { color: #1d3557; }
    .paso { background:#ffffff; border-left: 5px solid #2a9d8f; border-radius: 10px;
            padding: 0.7rem 1rem; margin-bottom: 0.6rem; box-shadow: 0 2px 6px rgba(29,53,87,0.05); }
    .paso b { color: #1d3557; }
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
# Perfiles de ejemplo (para probar la app con un clic)
# ---------------------------------------------------------------------------
PERFILES = {
    "🟥 Riesgo alto": {
        "debtor": 1, "tuition_fees_up_to_date": 0, "scholarship_holder": 0, "age_at_enrollment": 27,
        "curricular_units_1st_sem_enrolled": 6, "curricular_units_1st_sem_evaluations": 7,
        "curricular_units_1st_sem_approved": 1, "curricular_units_1st_sem_grade": 10.0,
        "curricular_units_2nd_sem_enrolled": 6, "curricular_units_2nd_sem_evaluations": 4,
        "curricular_units_2nd_sem_approved": 0, "curricular_units_2nd_sem_grade": 0.0,
    },
    "🟨 Riesgo intermedio": {
        "debtor": 0, "tuition_fees_up_to_date": 1, "scholarship_holder": 0, "age_at_enrollment": 20,
        "curricular_units_1st_sem_enrolled": 6, "curricular_units_1st_sem_evaluations": 11,
        "curricular_units_1st_sem_approved": 4, "curricular_units_1st_sem_grade": 11.5,
        "curricular_units_2nd_sem_enrolled": 6, "curricular_units_2nd_sem_evaluations": 12,
        "curricular_units_2nd_sem_approved": 3, "curricular_units_2nd_sem_grade": 11.2,
    },
    "🟩 Trayectoria favorable": {
        "debtor": 0, "tuition_fees_up_to_date": 1, "scholarship_holder": 1, "age_at_enrollment": 20,
        "curricular_units_1st_sem_enrolled": 6, "curricular_units_1st_sem_evaluations": 6,
        "curricular_units_1st_sem_approved": 6, "curricular_units_1st_sem_grade": 14.2,
        "curricular_units_2nd_sem_enrolled": 6, "curricular_units_2nd_sem_evaluations": 6,
        "curricular_units_2nd_sem_approved": 6, "curricular_units_2nd_sem_grade": 14.6,
    },
}


# ---------------------------------------------------------------------------
# Widgets del formulario (valores guardados en st.session_state)
# ---------------------------------------------------------------------------
def describir_codigo(columna, codigo, version_v2):
    if columna == "course":
        return (CURSOS_V2 if version_v2 else CURSOS_V1).get(codigo, f"Código {codigo}")
    if columna == "application_mode":
        return (MODOS_V2 if version_v2 else MODOS_V1).get(codigo, f"Código {codigo}")
    return f"Código {codigo}"


def valor_defecto(columna, tipo, categorias):
    if columna in BINARIAS:
        return "Sí" if VALORES_DEFECTO.get(columna, 0) == 1 else "No"
    if columna in categorias:
        codigos = categorias[columna]
        preferido = {"course": 11, "application_mode": 1}.get(columna)
        return preferido if preferido in codigos else codigos[0]
    if tipo in ("int", "bigint"):
        return int(VALORES_DEFECTO.get(columna, 0))
    return float(VALORES_DEFECTO.get(columna, 0.0))


def inicializar_estado(columnas_schema, categorias):
    for col, tipo in columnas_schema.items():
        if col not in st.session_state:
            st.session_state[col] = valor_defecto(col, tipo, categorias)


def aplicar_perfil():
    perfil = PERFILES.get(st.session_state.get("perfil_ejemplo"))
    if not perfil:
        return
    for col, valor in perfil.items():
        if col not in st.session_state:
            continue
        if col in BINARIAS:
            st.session_state[col] = "Sí" if valor == 1 else "No"
        elif isinstance(st.session_state[col], float):
            st.session_state[col] = float(valor)
        else:
            st.session_state[col] = int(valor)


def widget(columna, tipo, categorias, version_v2, contenedor):
    etiqueta = ETIQUETAS.get(columna, columna.replace("_", " ").capitalize())
    if columna in BINARIAS:
        opcion = contenedor.radio(etiqueta, ["Sí", "No"], horizontal=True, key=columna)
        return 1 if opcion == "Sí" else 0
    if columna in categorias:
        return contenedor.selectbox(
            etiqueta, categorias[columna], key=columna,
            format_func=lambda x: describir_codigo(columna, x, version_v2))
    if tipo in ("int", "bigint"):
        minimo, maximo = LIMITES.get(columna, (0, 60))
        return contenedor.number_input(etiqueta, min_value=int(minimo), max_value=int(maximo), step=1, key=columna)
    minimo, maximo = LIMITES.get(columna, (0.0, 20.0))
    return contenedor.number_input(etiqueta, min_value=float(minimo), max_value=float(maximo), step=0.1,
                                   format="%.2f", key=columna)


@st.cache_data
def cargar_metricas():
    if RUTA_METRICAS.exists():
        return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
if RUTA_BANNER.exists():
    st.image(str(RUTA_BANNER), width="stretch")
st.title("🎓 Alerta Temprana de Deserción Estudiantil")
st.markdown("Predice con **Spark ML** si un estudiante va a **desertar**, **seguir matriculado** "
            "o **graduarse**, a partir de su información de ingreso y de sus dos primeros semestres.")

try:
    modelo, labels, schema, categorias, nombre_modelo = cargar_modelo()
except Exception as error:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo desde '{RUTA_MODELO.name}/'. Detalle: {error}")
    st.stop()

metricas = cargar_metricas()
columnas_schema = {c["nombre"]: c["tipo"] for c in schema["columnas"]}
version_v2 = max(categorias.get("course", [0])) > 100
inicializar_estado(columnas_schema, categorias)

with st.sidebar:
    st.header("📋 Instrucciones de uso")
    st.markdown(
        """
        1. **Opcional:** elige un **perfil de ejemplo** para llenar el formulario con un clic.
        2. Completa o ajusta los datos del estudiante en las pestañas **Perfil**, **Familia**
           y **Rendimiento académico**.
        3. En cada semestre, las materias **aprobadas** no pueden superar a las **inscritas**
           ni a las **evaluaciones presentadas**.
        4. Presiona **🔮 Predecir**.
        5. Revisa la **clase predicha** (tarjeta de color) y la **probabilidad de cada clase**
           en el gráfico y la tabla.

        **Clases posibles**
        - 🟥 **Dropout:** abandona la carrera.
        - 🟨 **Enrolled:** sigue matriculado, pero no se gradúa a tiempo.
        - 🟩 **Graduate:** se gradúa en el tiempo normal.
        """
    )
    st.caption("Las variables derivadas (tasas de aprobación, variación de la nota y eficiencia) "
               "se calculan automáticamente dentro de la app.")
    st.divider()
    st.markdown(f"**Modelo:** {nombre_modelo} · Spark ML {schema.get('pyspark_version', '')}")
    st.markdown(f"**Variables de entrada:** {len(schema['columnas'])} + 5 derivadas")
    st.markdown(f"[📂 Repositorio en GitHub]({URL_REPO})")

if metricas:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Modelo seleccionado", metricas["mejor_modelo"])
    k2.metric("F1 ponderado (prueba)", f"{metricas['test']['f1']:.3f}")
    k3.metric("Accuracy (prueba)", f"{metricas['test']['accuracy'] * 100:.1f} %")
    k4.metric("Recall de Enrolled", f"{metricas['test']['recall_enrolled'] * 100:.1f} %",
              delta=f"+{(metricas['control_pesos']['recall_enrolled_con'] - metricas['control_pesos']['recall_enrolled_sin']) * 100:.1f} pts con pesos")

tab_prediccion, tab_modelo, tab_como = st.tabs(["🔮 Predicción", "📊 Sobre el modelo", "⚙️ Cómo funciona"])

# ---------------------------------------------------------------------------
# Pestaña 1: Predicción
# ---------------------------------------------------------------------------
with tab_prediccion:
    semestre1 = [c for c in columnas_schema if c.startswith("curricular_units_1st_sem")]
    semestre2 = [c for c in columnas_schema if c.startswith("curricular_units_2nd_sem")]
    familia = [c for c in columnas_schema if c.startswith(("mothers_", "fathers_"))]
    macro = [c for c in columnas_schema if c in ("unemployment_rate", "inflation_rate", "gdp")]
    perfil = [c for c in columnas_schema if c not in semestre1 + semestre2 + familia + macro]

    st.selectbox("⚡ Perfil de ejemplo (llena el formulario automáticamente)",
                 ["— Elegir —"] + list(PERFILES.keys()), key="perfil_ejemplo", on_change=aplicar_perfil)

    # (b) Formulario con todas las variables que usa el modelo y botón Predecir
    valores = {}
    with st.form("formulario_estudiante"):
        sub_perfil, sub_familia, sub_academico = st.tabs(["👤 Perfil", "👨‍👩‍👧 Familia", "📚 Rendimiento académico"])
        with sub_perfil:
            cols = st.columns(2)
            for i, col in enumerate(perfil + macro):
                valores[col] = widget(col, columnas_schema[col], categorias, version_v2, cols[i % 2])
        with sub_familia:
            cols = st.columns(2)
            for i, col in enumerate(familia):
                valores[col] = widget(col, columnas_schema[col], categorias, version_v2, cols[i % 2])
            st.caption("Los códigos de formación y ocupación corresponden a la clasificación oficial del dataset.")
        with sub_academico:
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
        else:
            with st.spinner("Calculando la predicción con Spark..."):
                clase, probabilidades, derivadas = predecir(modelo, labels, schema, valores)

            # (e) Clase predicha destacada + probabilidad de cada clase (gráfico y tabla)
            st.subheader("Resultado de la predicción")
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
                import altair as alt

                tabla = pd.DataFrame({"Clase": list(probabilidades.keys()),
                                      "Probabilidad": list(probabilidades.values())})
                tabla = tabla.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
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

            with st.expander("🧮 Variables derivadas calculadas por la app (punto 3 del notebook)"):
                nombres_derivadas = {
                    "tasa_aprob_1sem": "Tasa de aprobación 1.er semestre",
                    "tasa_aprob_2sem": "Tasa de aprobación 2.º semestre",
                    "variacion_nota": "Variación de la nota (2.º − 1.er)",
                    "variacion_tasa_aprob": "Variación de la tasa de aprobación",
                    "eficiencia_evaluaciones": "Eficiencia en evaluaciones",
                }
                st.dataframe(pd.DataFrame({
                    "Variable": [nombres_derivadas.get(k, k) for k in derivadas],
                    "Valor": [round(float(v), 4) for v in derivadas.values()],
                }), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Pestaña 2: Sobre el modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    if not metricas:
        st.info("No se encontró assets/metricas.json.")
    else:
        import altair as alt

        st.subheader("Comparación de los tres modelos (conjunto de prueba)")
        st.caption("Los tres modelos se entrenaron dentro de un Pipeline completo y se ajustaron con "
                   "ParamGridBuilder + CrossValidator (3 folds, F1 ponderado). El mejor se eligió por el F1 "
                   "ponderado y el recall de la clase minoritaria (Enrolled).")
        comp = pd.DataFrame(metricas["comparacion"])
        st.dataframe(comp, hide_index=True, width="stretch")

        izq, der = st.columns(2)
        with izq:
            st.subheader("Matriz de confusión")
            mc = metricas["matriz_confusion"]
            filas = [{"Real": r, "Predicha": p, "Estudiantes": mc["valores"][i][j]}
                     for i, r in enumerate(mc["clases"]) for j, p in enumerate(mc["clases"])]
            base = alt.Chart(pd.DataFrame(filas)).encode(
                x=alt.X("Predicha:N", sort=mc["clases"]), y=alt.Y("Real:N", sort=mc["clases"]))
            mapa = base.mark_rect(cornerRadius=4).encode(
                color=alt.Color("Estudiantes:Q", scale=alt.Scale(scheme="tealblues"), legend=None))
            texto = base.mark_text(fontSize=16, fontWeight="bold").encode(
                text="Estudiantes:Q",
                color=alt.condition(alt.datum.Estudiantes > 150, alt.value("white"), alt.value("#1d3557")))
            st.altair_chart((mapa + texto).properties(height=300), width="stretch")
        with der:
            st.subheader("Métricas por clase")
            st.dataframe(pd.DataFrame(metricas["por_clase"]), hide_index=True, width="stretch")
            st.markdown(
                f"""
                - Solo **{mc['valores'][1][0]} desertores** se confunden con *Graduate* (el error más costoso).
                - Con **pesos por clase**, el recall de *Enrolled* sube de
                  **{metricas['control_pesos']['recall_enrolled_sin'] * 100:.1f} %** a
                  **{metricas['control_pesos']['recall_enrolled_con'] * 100:.1f} %**.
                - Hiperparámetros: `numTrees={metricas['hiperparametros']['numTrees']}`,
                  `maxDepth={metricas['hiperparametros']['maxDepth']}` · F1 en validación cruzada:
                  **{metricas['f1_cv']:.4f}**.
                """
            )

        st.subheader("Variables más importantes")
        imp = pd.DataFrame(metricas["importancias"])
        barras = (alt.Chart(imp).mark_bar(cornerRadiusEnd=5)
                  .encode(x=alt.X("Importancia:Q", axis=alt.Axis(format="%")),
                          y=alt.Y("Variable:N", sort="-x", title=None),
                          color=alt.condition(alt.FieldOneOfPredicate(field="Variable", oneOf=[
                              v for v in imp["Variable"] if "derivada" in v]),
                              alt.value("#e76f51"), alt.value("#2a9d8f")),
                          tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".2%")])
                  .properties(height=320))
        st.altair_chart(barras, width="stretch")
        st.caption("En naranja, las variables derivadas creadas en la ingeniería de características: "
                   "3 de ellas están entre las 4 más importantes.")

# ---------------------------------------------------------------------------
# Pestaña 3: Cómo funciona
# ---------------------------------------------------------------------------
with tab_como:
    st.subheader("Del dato crudo a la predicción")
    pasos = [
        ("1. Formulario", "Se capturan las variables que usa el modelo con sus tipos exactos del entrenamiento (`schema_entrada.json`)."),
        ("2. Variables derivadas", "La app calcula las mismas 5 variables del notebook con control de división por cero: tasas de aprobación por semestre, variación de la nota, variación de la tasa y eficiencia en evaluaciones."),
        ("3. DataFrame de Spark", "Los datos se convierten en un DataFrame de Spark con `spark.createDataFrame` y el esquema guardado."),
        ("4. PipelineModel", "El modelo cargado con `PipelineModel.load` aplica StringIndexer (handleInvalid='keep') → OneHotEncoder → VectorAssembler → StandardScaler → Random Forest."),
        ("5. Resultado", "El índice de `prediction` y el vector `probability` se traducen a nombres de clase con el orden de `labels.json` (StringIndexer de Target)."),
    ]
    for titulo, texto in pasos:
        st.markdown(f"<div class='paso'><b>{titulo}</b><br>{texto}</div>", unsafe_allow_html=True)

    if metricas:
        d = metricas["datos"]
        st.subheader("Datos de entrenamiento")
        a, b, c = st.columns(3)
        a.metric("Estudiantes", f"{d['estudiantes']:,}".replace(",", "."))
        b.metric("Entrenamiento / prueba", f"{d['train']:,} / {d['test']:,}".replace(",", "."))
        c.metric("Variables del modelo", d["variables_modelo"])
    st.caption("Dataset: Predict Students' Dropout and Academic Success (Kaggle / UCI). "
               "Spark y el modelo se mantienen en memoria con @st.cache_resource.")

st.markdown(
    f"<div class='pie-pagina'>Desarrollado por <b>{AUTOR}</b> · Examen de Big Data – Spark ML · "
    f"Universidad Autónoma de Bucaramanga (UNAB) · <a href='{URL_REPO}' target='_blank'>GitHub</a></div>",
    unsafe_allow_html=True,
)
