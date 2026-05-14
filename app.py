import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="Riesgo actuarial", layout="centered")
st.title("Predicción de riesgo actuarial")


@st.cache_resource
def cargar_modelo():
    pkl = (
        "kmeans_riesgo_actuarial.pkl"
        if os.path.exists("kmeans_riesgo_actuarial.pkl")
        else "kmeans_riesgo_actuarial(2).pkl"
    )

    meta = (
        "model_metadata.json"
        if os.path.exists("model_metadata.json")
        else "model_metadata(2).json"
    )

    modelo = joblib.load(pkl)

    with open(meta, encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, metadata


@st.cache_data
def cargar_base():
    csv = "insurance.csv" if os.path.exists("insurance.csv") else "insurance(2).csv"
    return pd.read_csv(csv)


@st.cache_resource
def entrenar_regresion_logistica(_modelo, _df):
    """
    Entrena una Regresión Logística binaria (riesgo Alto = 1, resto = 0)
    usando el espacio PCA 2D del pipeline K-means existente.
    """
    df_work = _df.copy()
    for col in ["sex", "smoker", "region"]:
        df_work[col] = df_work[col].astype(str).str.strip().str.lower()
    df_work = df_work.drop_duplicates()

    numeric_features = ["age", "bmi", "children", "charges"]
    categorical_features = ["sex", "smoker", "region"]
    X = df_work[numeric_features + categorical_features]

    # Preprocesar con el pipeline existente
    preprocessor = _modelo.named_steps["preprocessor"]
    X_prep = preprocessor.transform(X)
    X_dense = X_prep.toarray() if hasattr(X_prep, "toarray") else X_prep

    # Reducir a PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)

    # Etiquetas del K-means
    clusters = _modelo.predict(X)
    mapa_local = {int(k): v for k, v in metadata["mapa_riesgo"].items()}
    riesgo_labels = pd.Series(clusters).map(mapa_local)

    # Variable binaria: 1 = Alto, 0 = No alto
    y_bin = (riesgo_labels == "Alto").astype(int)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_pca, y_bin)

    return lr, pca


modelo, metadata = cargar_modelo()
df = cargar_base()
mapa = {int(k): v for k, v in metadata["mapa_riesgo"].items()}
lr_model, pca_model = entrenar_regresion_logistica(modelo, df)

st.caption(metadata["nombre_modelo"])

with st.form("datos"):
    col1, col2 = st.columns(2)

    age = col1.number_input("Edad", 18, 100, 35)
    sex = col2.selectbox("Sexo", sorted(df["sex"].unique()))
    bmi = col1.number_input("BMI", 10.0, 60.0, 28.0)
    children = col2.number_input("Hijos", 0, 10, 1)
    smoker = col1.selectbox("Fumador", sorted(df["smoker"].unique()))
    region = col2.selectbox("Región", sorted(df["region"].unique()))
    charges = st.number_input(
        "Cargos médicos estimados",
        0.0,
        100000.0,
        12000.0
    )

    enviar = st.form_submit_button("Evaluar")


if enviar:
    cliente = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges,
        }
    ])

    # ── K-means: cluster y nivel de riesgo ──────────────────────────────────
    cluster = int(modelo.predict(cliente)[0])
    riesgo = mapa.get(cluster, "No definido")

    st.subheader(f"Riesgo actuarial: {riesgo}")
    st.write(f"Cluster asignado: {cluster}")

    # ── Regresión Logística: resultado binario y probabilidad ────────────────
    st.divider()
    st.subheader("Regresión Logística — Riesgo Alto")

    # Transformar cliente al espacio PCA
    X_cli_prep = modelo.named_steps["preprocessor"].transform(cliente)
    X_cli_dense = X_cli_prep.toarray() if hasattr(X_cli_prep, "toarray") else X_cli_prep
    X_cli_pca = pca_model.transform(X_cli_dense)

    lr_pred = int(lr_model.predict(X_cli_pca)[0])          # 0 o 1
    lr_proba = float(lr_model.predict_proba(X_cli_pca)[0][1])  # P(Alto)

    col_a, col_b = st.columns(2)
    col_a.metric(
        label="Resultado binario",
        value="1 — Riesgo Alto" if lr_pred == 1 else "0 — Sin riesgo alto",
    )
    col_b.metric(
        label="Probabilidad de riesgo alto",
        value=f"{lr_proba:.1%}",
    )

    st.progress(lr_proba, text=f"P(Alto) = {lr_proba:.1%}")

    # ── Groq: recomendaciones ────────────────────────────────────────────────
    st.divider()
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    if api_key:
        prompt = f"""
        Actúa como analista actuarial.

        Explica brevemente el resultado del modelo y brinda 3 recomendaciones prudentes,
        claras y profesionales para el usuario.

        Datos del cliente:
        - Edad: {age}
        - Sexo: {sex}
        - BMI: {bmi}
        - Hijos: {children}
        - Fumador: {smoker}
        - Región: {region}
        - Cargos médicos estimados: {charges}

        Resultado del modelo:
        - Cluster asignado: {cluster}
        - Nivel de riesgo actuarial (K-means): {riesgo}
        - Regresión Logística (riesgo alto): {lr_pred} ({"Sí" if lr_pred == 1 else "No"})
        - Probabilidad de riesgo alto: {lr_proba:.1%}
        """

        try:
            client = Groq(api_key=api_key)

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista actuarial prudente, claro y profesional.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.4,
                max_tokens=500,
            )

            respuesta = completion.choices[0].message.content
            st.info(respuesta)

        except Exception as e:
            st.warning(f"No se pudo generar recomendación con Groq: {e}")

    else:
        st.warning("Agregue GROQ_API_KEY en los secretos de Streamlit.")


st.divider()
st.write("Vista rápida de la base principal")
st.dataframe(df.head(20), use_container_width=True)
