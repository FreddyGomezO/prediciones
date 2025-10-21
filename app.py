"""
Reconocimiento de Personas — Streamlit + Teachable Machine (Keras)
Archivo: reconocimiento_personas_streamlit.py

Descripción:
Este archivo implementa la aplicación completa requerida para la Tarea de Exoneración:
- Sección "En vivo": cámara WebRTC + modo foto alternativo.
- Sección "Administración": CRUD de personas (label, name, email, role, threshold, notes) sobre SQLite.
- Sección "Analítica": 5 gráficas, guardado de PNG, exportación ZIP con PNGs.
- Persistencia: base de datos SQLite `predictions.db` con tablas `people` y `predictions`.
- Exportación CSV (DB) y CSV en memoria.

Instrucciones rápidas:
1) Coloca `keras_Model.h5` y `labels.txt` junto a este archivo.
2) Crear entorno e instalar dependencias (ver bloque REQUIREMENTS abajo).
3) Ejecutar: `streamlit run reconocimiento_personas_streamlit.py`

Archivos para entregar en GitHub (no subir .venv):
- reconocimiento_personas_streamlit.py
- requirements.txt
- labels.txt
- keras_Model.h5 (si pesa mucho, usar Git LFS o descargar en tiempo de inicio)
- carpeta `outputs/` (contendrá graphs y zip)

Informe (PDF/Word) debe incluir: descripción del modelo, ejemplos de imágenes, capturas de las 3 secciones, análisis breve, links a GitHub y Streamlit Cloud.

"""

import os
import io
import time
import zipfile
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# -----------------------------
# Config
# -----------------------------
# -----------------------------
# Directorio base del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(BASE_DIR, "keras_model.h5")  # asegúrate que coincida

print("Buscando modelo en:", MODEL_FILENAME)
print("Existe?", os.path.exists(MODEL_FILENAME))
LABELS_FILENAME = os.path.join(BASE_DIR, "labels.txt")
DB_FILENAME = os.path.join(BASE_DIR, "predictions.db")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
IMAGES_DIR = os.path.join(OUTPUTS_DIR, "images")
ZIP_PATH = os.path.join(OUTPUTS_DIR, "graphs.zip")

# Crear carpetas si no existen
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


INPUT_SIZE = (224, 224)  # Cambia si tu modelo usa otro tamaño

st.set_page_config(page_title="Reconocimiento Personas", layout="wide")
st.title("Reconocimiento de Personas — Streamlit + Teachable Machine")

# -----------------------------
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILENAME, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE,
            name TEXT,
            email TEXT,
            role TEXT,
            threshold REAL DEFAULT 0.5,
            notes TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            label TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()
    return conn

conn = init_db()

# -----------------------------
# Model & labels
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"No se encontró {MODEL_FILENAME}. Colócalo en la misma carpeta.")
        return None
    model = load_model(MODEL_FILENAME, compile=False)
    return model

@st.cache_data(show_spinner=False)
def load_labels_cached():
    if not os.path.exists(LABELS_FILENAME):
        st.error(f"No se encontró {LABELS_FILENAME}. Colócalo en la misma carpeta.")
        return None
    with open(LABELS_FILENAME, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    return labels

model = load_model_cached()
labels = load_labels_cached()

if model is None or labels is None:
    st.stop()

# -----------------------------
# Utils: preprocessing, predict, db log
# -----------------------------

def preprocess_pil(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img).astype(np.float32)
    # Normalización típica Teachable Machine: -1..1
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, 0)
    return arr


def predict_image_pil(img: Image.Image):
    x = preprocess_pil(img)
    preds = model.predict(x, verbose=0)
    if preds.ndim == 2:
        preds = preds[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    label = labels[idx] if idx < len(labels) else str(idx)
    return label, conf


def db_log_prediction(source: str, label: str, confidence: float):
    ts = datetime.utcnow().isoformat() + "Z"
    c = conn.cursor()
    c.execute("INSERT INTO predictions (timestamp, source, label, confidence) VALUES (?,?,?,?)",
              (ts, source, label, float(confidence)))
    conn.commit()


def get_threshold_for_label(label: str):
    c = conn.cursor()
    c.execute("SELECT threshold FROM people WHERE label=?", (label,))
    r = c.fetchone()
    return float(r[0]) if r else 0.5


def save_image(img: Image.Image, label: str):
    """Guarda la imagen en la carpeta outputs/images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    img.save(filepath)
    return filepath

# -----------------------------
# RTC config
# -----------------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Video transformer
# -----------------------------
class RecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest = {"label": None, "confidence": 0.0}

    def recv(self, frame):
        img = frame.to_image()  # PIL image
        # make prediction
        try:
            label, conf = predict_image_pil(img)
        except Exception:
            return frame
        self.latest = {"label": label, "confidence": conf}

        # Draw overlay
        draw = ImageDraw.Draw(img)
        text = f"{label} {conf*100:.1f}%"
        draw.rectangle([(0,0),(260,30)], fill=(0,0,0,180))
        draw.text((6,6), text, fill=(255,255,255))
        return frame.from_image(img)

# -----------------------------
# UI: sidebar
# -----------------------------
st.sidebar.header("Ajustes cámara")
facing = st.sidebar.selectbox("Tipo de cámara (facingMode)", ["auto (por defecto)", "user", "environment"], index=0)
quality = st.sidebar.selectbox("Calidad video", ["640x480", "1280x720"], index=1)
w, h = map(int, quality.split("x"))
media_constraints = {"video": {"width": w, "height": h, "facingMode": facing if facing != 'auto (por defecto)' else None}, "audio": False}

st.sidebar.header("Registro")
enable_log = st.sidebar.checkbox("Habilitar registro (persistente en SQLite)", value=True)
log_interval = st.sidebar.slider("Intervalo registro (s)", 0.2, 5.0, 1.0, 0.2)

# -----------------------------
# Main menu
# -----------------------------
menu = st.sidebar.radio("Sección", ["En vivo", "Administración", "Analítica", "Exportar / Entregar"]) 

# -----------------------------
# En vivo
# -----------------------------
if menu == "En vivo":
    st.header("En vivo — Cámara o Foto")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Streaming WebRTC")
        webrtc_ctx = webrtc_streamer(
            key="recog",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=media_constraints,
            video_transformer_factory=RecognitionTransformer,
            async_transform=True,
        )

        st.info("Si no se muestra la cámara usa la opción de Foto (modo alternativo) o revisa permisos.")

    with col2:
        st.subheader("Última predicción")
        placeholder_text = st.empty()
        placeholder_progress = st.empty()
        
        # Bucle de actualización continua
        if webrtc_ctx and webrtc_ctx.video_transformer:
            while webrtc_ctx.state.playing:
                vt = webrtc_ctx.video_transformer
                if vt.latest['label']:
                    placeholder_text.markdown(f"**Etiqueta:** `{vt.latest['label']}` — **Confianza:** `{vt.latest['confidence']*100:.2f}%`")
                    placeholder_progress.progress(vt.latest['confidence'])
                else:
                    placeholder_text.markdown("**Esperando detección...**")
                    placeholder_progress.progress(0.0)
                time.sleep(0.1)  # Actualizar cada 100ms
        else:
            placeholder_text.markdown("**Esperando detección...**")
            placeholder_progress.progress(0.0)

        st.markdown("---")
        st.subheader("SELECT DEVICE")
        uploaded_file = st.file_uploader("Seleccionar imagen", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            label, conf = predict_image_pil(img)
            st.image(img, caption=f"{label} — {conf*100:.2f}%")
            
            # Guardar imagen
            saved_path = save_image(img, label)
            
            # log if passes threshold
            thr = get_threshold_for_label(label)
            if conf >= thr and enable_log:
                db_log_prediction("imagen", label, conf)
                st.success(f"Registrado como 'imagen' (conf >= threshold {thr:.2f})")
                st.info(f"Imagen guardada en: {saved_path}")
            else:
                st.info(f"No registrado: conf {conf:.2f} < threshold {thr:.2f}")
        
        st.markdown("---")
        st.subheader("Prueba con foto (foto puntual)")
        snap = st.camera_input("Tomar una foto")
        if snap is not None:
            img = Image.open(snap)
            label, conf = predict_image_pil(img)
            st.image(img, caption=f"{label} — {conf*100:.2f}%")
            
            # Guardar imagen
            saved_path = save_image(img, label)
            
            # log if passes threshold
            thr = get_threshold_for_label(label)
            if conf >= thr and enable_log:
                db_log_prediction("camara", label, conf)
                st.success(f"Registrado como 'camara' (conf >= threshold {thr:.2f})")
                st.info(f"Imagen guardada en: {saved_path}")
            else:
                st.info(f"No registrado: conf {conf:.2f} < threshold {thr:.2f}")

# -----------------------------
# Administración
# -----------------------------
if menu == "Administración":
    st.header("Administración de personas (CRUD)")
    with st.expander("Agregar persona"):
        with st.form("form_add"):
            label = st.text_input("Etiqueta (label que devuelve el modelo)")
            name = st.text_input("Nombre completo")
            email = st.text_input("Correo")
            role = st.text_input("Rol")
            threshold = st.slider("Umbral (confianza mínima)", 0.0, 1.0, 0.5)
            notes = st.text_area("Notas")
            ok = st.form_submit_button("Agregar")
            if ok:
                if not label:
                    st.error("Etiqueta requerida")
                else:
                    c = conn.cursor()
                    try:
                        c.execute("INSERT INTO people (label,name,email,role,threshold,notes) VALUES (?,?,?,?,?,?)",
                                  (label, name, email, role, threshold, notes))
                        conn.commit()
                        st.success("Persona agregada")
                    except sqlite3.IntegrityError:
                        st.error("La etiqueta ya existe. Usa editar.")

    st.markdown("---")
    st.subheader("Editar / Eliminar")
    df_people = pd.read_sql_query("SELECT * FROM people ORDER BY label", conn)
    if df_people.empty:
        st.info("No hay personas registradas aún.")
    else:
        sel = st.selectbox("Selecciona etiqueta", df_people['label'].tolist())
        row = df_people[df_people['label'] == sel].iloc[0]
        with st.form("form_edit"):
            name = st.text_input("Nombre", value=row['name'] or "")
            email = st.text_input("Correo", value=row['email'] or "")
            role = st.text_input("Rol", value=row['role'] or "")
            threshold = st.slider("Umbral", 0.0, 1.0, float(row['threshold'] or 0.5))
            notes = st.text_area("Notas", value=row['notes'] or "")
            save = st.form_submit_button("Guardar")
            delete = st.form_submit_button("Eliminar")
            if save:
                c = conn.cursor()
                c.execute("UPDATE people SET name=?,email=?,role=?,threshold=?,notes=? WHERE label=?",
                          (name, email, role, threshold, notes, sel))
                conn.commit()
                st.success("Actualizado")
            if delete:
                c = conn.cursor()
                c.execute("DELETE FROM people WHERE label=?", (sel,))
                conn.commit()
                st.success("Eliminado")

# -----------------------------
# Analítica
# -----------------------------
if menu == "Analítica":
    st.header("Analítica — Gráficas y estadísticas")
    df_pred = pd.read_sql_query("SELECT * FROM predictions", conn)
    if df_pred.empty:
        st.info("No hay predicciones registradas aún. Usa En vivo para generar datos.")
    else:
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])

        # 1) Conteo por etiqueta
        fig1, ax1 = plt.subplots()
        df_pred['label'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Detecciones por etiqueta')
        fig1.tight_layout()
        fig1_path = os.path.join(GRAPHS_DIR, 'detecciones_por_etiqueta.png')
        fig1.savefig(fig1_path)
        st.pyplot(fig1)

        # 2) Confianza promedio por etiqueta
        fig2, ax2 = plt.subplots()
        df_pred.groupby('label')['confidence'].mean().sort_values(ascending=False).plot(kind='bar', ax=ax2)
        ax2.set_title('Confianza promedio por etiqueta')
        fig2.tight_layout()
        fig2_path = os.path.join(GRAPHS_DIR, 'confianza_promedio.png')
        fig2.savefig(fig2_path)
        st.pyplot(fig2)

        # 3) Serie temporal (detecciones por día)
        fig3, ax3 = plt.subplots()
        df_pred.set_index('timestamp').resample('D').size().plot(ax=ax3)
        ax3.set_title('Detecciones por día')
        fig3.tight_layout()
        fig3_path = os.path.join(GRAPHS_DIR, 'detecciones_por_dia.png')
        fig3.savefig(fig3_path)
        st.pyplot(fig3)

        # 4) Distribución de confianza
        fig4, ax4 = plt.subplots()
        df_pred['confidence'].plot(kind='hist', bins=20, ax=ax4)
        ax4.set_title('Distribución de confianza')
        fig4.tight_layout()
        fig4_path = os.path.join(GRAPHS_DIR, 'distribucion_confianza.png')
        fig4.savefig(fig4_path)
        st.pyplot(fig4)

        # 5) Fuente vs cantidad
        fig5, ax5 = plt.subplots()
        df_pred['source'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax5)
        ax5.set_ylabel('')
        ax5.set_title('Fuente de las predicciones')
        fig5.tight_layout()
        fig5_path = os.path.join(GRAPHS_DIR, 'fuente_predicciones.png')
        fig5.savefig(fig5_path)
        st.pyplot(fig5)

        st.markdown("---")
        st.subheader("Tabla de registros")
        st.dataframe(df_pred.sort_values('timestamp', ascending=False).reset_index(drop=True))

# -----------------------------
# Exportar / Entregar
# -----------------------------
if menu == "Exportar / Entregar":
    st.header("Exportación y archivos para entrega")
    st.write("Descargar CSV con registros de predicción (DB):")
    df_all = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    if not df_all.empty:
        csv_bytes = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar CSV (DB)", data=csv_bytes, file_name='predictions_db.csv', mime='text/csv')
    else:
        st.info("No hay datos para exportar")

    st.write('Generar ZIP con gráficas (PNG)')
    if st.button('Generar ZIP con PNG'):
        if not os.path.exists(GRAPHS_DIR) or not os.listdir(GRAPHS_DIR):
            st.warning('Aún no hay gráficas generadas. Ve a Analítica y espera a que aparezcan.')
        else:
            with zipfile.ZipFile(ZIP_PATH, 'w') as zf:
                for fname in os.listdir(GRAPHS_DIR):
                    zf.write(os.path.join(GRAPHS_DIR, fname), arcname=fname)
            with open(ZIP_PATH, 'rb') as f:
                st.download_button('Descargar ZIP con gráficas', data=f, file_name='graphs.zip', mime='application/zip')

    st.markdown('---')
    st.subheader('Instrucciones para entrega')
    st.markdown(
        """
        - Sube tu repositorio a GitHub **sin** el entorno virtual (.venv).
        - Incluye `requirements.txt`, `reconocimiento_personas_streamlit.py`, `keras_Model.h5`, `labels.txt` y una carpeta `outputs` vacía.
        - Si el modelo pesa mucho, usa GitHub LFS o descarga el modelo desde una URL pública en el arranque de la app.
        - En el informe: captura En vivo, Administración y Analítica, describe el modelo y pega los links a GitHub y Streamlit Cloud.
        """
    )

# -----------------------------
# REQUIREMENTS (guardar en requirements.txt)
# -----------------------------
REQUIREMENTS = '''
streamlit
streamlit-webrtc
tensorflow>=2.9
pillow
numpy
pandas
matplotlib
'''