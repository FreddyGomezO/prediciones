"""
Reconocimiento de Personas — Streamlit + Teachable Machine (Keras) - MEJORADO
Archivo: reconocimiento_personas_mejorado.py

Mejoras implementadas:
- Diseño visual mejorado con métricas y cards
- 7 gráficas analíticas profesionales
- Mejor organización de la interfaz
- Sistema de notificaciones mejorado
- Exportación optimizada
"""

import os
import io
import time
import zipfile
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# -----------------------------
# Configuración de estilo
# -----------------------------
plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------
# Config de directorios
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_FILENAME = os.path.join(BASE_DIR, "labels.txt")
DB_FILENAME = os.path.join(BASE_DIR, "predictions.db")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
IMAGES_DIR = os.path.join(OUTPUTS_DIR, "images")
ZIP_PATH = os.path.join(OUTPUTS_DIR, "graphs.zip")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

INPUT_SIZE = (224, 224)

# -----------------------------
# Configuración de página
# -----------------------------
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Sistema de Reconocimiento Facial</div>', unsafe_allow_html=True)

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
# Carga del modelo y etiquetas
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"⚠️ No se encontró {MODEL_FILENAME}. Colócalo en la misma carpeta.")
        return None
    model = load_model(MODEL_FILENAME, compile=False)
    return model

@st.cache_data(show_spinner=False)
def load_labels_cached():
    if not os.path.exists(LABELS_FILENAME):
        st.error(f"⚠️ No se encontró {LABELS_FILENAME}. Colócalo en la misma carpeta.")
        return None
    with open(LABELS_FILENAME, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    return labels

with st.spinner("Cargando modelo de reconocimiento..."):
    model = load_model_cached()
    labels = load_labels_cached()

if model is None or labels is None:
    st.stop()

# -----------------------------
# Funciones auxiliares
# -----------------------------
def preprocess_pil(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img).astype(np.float32)
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
    return label, conf, preds

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    img.save(filepath)
    return filepath

# -----------------------------
# Video transformer
# -----------------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class RecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest = {"label": None, "confidence": 0.0}

    def recv(self, frame):
        img = frame.to_image()
        try:
            label, conf, _ = predict_image_pil(img)
        except Exception:
            return frame
        self.latest = {"label": label, "confidence": conf}

        draw = ImageDraw.Draw(img)
        text = f"{label} {conf*100:.1f}%"
        draw.rectangle([(0,0),(280,40)], fill=(0,120,215,200))
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10,10), text, fill=(255,255,255), font=font)
        return frame.from_image(img)

# -----------------------------
# Sidebar mejorado
# -----------------------------
st.sidebar.image("https://via.placeholder.com/250x80/1f77b4/ffffff?text=Face+Recognition", use_container_width=True)
st.sidebar.title("⚙️ Configuración")

st.sidebar.markdown("### 📹 Ajustes de Cámara")
facing = st.sidebar.selectbox("Tipo de cámara", ["auto (por defecto)", "user", "environment"], index=0)
quality = st.sidebar.selectbox("Calidad de video", ["640x480", "1280x720", "1920x1080"], index=1)
w, h = map(int, quality.split("x"))
media_constraints = {
    "video": {"width": w, "height": h, "facingMode": facing if facing != 'auto (por defecto)' else None},
    "audio": False
}

st.sidebar.markdown("### 📊 Registro de Datos")
enable_log = st.sidebar.checkbox("Habilitar registro automático", value=True)
log_interval = st.sidebar.slider("Intervalo de registro (segundos)", 0.2, 5.0, 1.0, 0.2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navegación")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    ["🎥 En vivo", "👥 Administración", "📈 Analítica", "📦 Exportar"],
    label_visibility="collapsed"
)

# -----------------------------
# SECCIÓN: En vivo
# -----------------------------
if menu == "🎥 En vivo":
    st.header("🎥 Reconocimiento en Tiempo Real")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📡 Transmisión en Vivo")
        webrtc_ctx = webrtc_streamer(
            key="recog",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=media_constraints,
            video_transformer_factory=RecognitionTransformer,
            async_transform=True,
        )
        st.info("💡 Si no se muestra la cámara, usa las opciones de captura alternativa abajo.")
    
    with col2:
        st.subheader("📊 Detección Actual")
        placeholder_card = st.empty()
        
        if webrtc_ctx and webrtc_ctx.video_transformer:
            while webrtc_ctx.state.playing:
                vt = webrtc_ctx.video_transformer
                if vt.latest['label']:
                    conf_pct = vt.latest['confidence'] * 100
                    color = "🟢" if conf_pct >= 70 else "🟡" if conf_pct >= 50 else "🔴"
                    placeholder_card.markdown(f"""
                    <div class="metric-card">
                        <h3>{color} {vt.latest['label']}</h3>
                        <p><strong>Confianza:</strong> {conf_pct:.2f}%</p>
                        <div style="background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="width: {conf_pct}%; background-color: #1f77b4; height: 20px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    placeholder_card.info("⏳ Esperando detección...")
                time.sleep(0.1)
        else:
            placeholder_card.info("⏳ Esperando detección...")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📤 Subir Imagen")
        uploaded_file = st.file_uploader("Selecciona una imagen", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            label, conf, preds = predict_image_pil(img)
            
            st.image(img, caption=f"Predicción: {label}", use_container_width=True)
            
            conf_pct = conf * 100
            st.metric("Confianza", f"{conf_pct:.2f}%", delta=f"{conf_pct-50:.1f}% vs umbral medio")
            
            saved_path = save_image(img, label)
            thr = get_threshold_for_label(label)
            
            if conf >= thr and enable_log:
                db_log_prediction("imagen", label, conf)
                st.success(f"✅ Registrado correctamente (umbral: {thr:.2f})")
                st.caption(f"📁 Guardado en: {saved_path}")
            else:
                st.warning(f"⚠️ No registrado: confianza {conf:.2f} < umbral {thr:.2f}")
    
    with col4:
        st.subheader("📸 Captura Instantánea")
        snap = st.camera_input("Toma una foto")
        if snap is not None:
            img = Image.open(snap)
            label, conf, preds = predict_image_pil(img)
            
            st.image(img, caption=f"Predicción: {label}", use_container_width=True)
            
            conf_pct = conf * 100
            st.metric("Confianza", f"{conf_pct:.2f}%", delta=f"{conf_pct-50:.1f}% vs umbral medio")
            
            saved_path = save_image(img, label)
            thr = get_threshold_for_label(label)
            
            if conf >= thr and enable_log:
                db_log_prediction("camara", label, conf)
                st.success(f"✅ Registrado correctamente (umbral: {thr:.2f})")
                st.caption(f"📁 Guardado en: {saved_path}")
            else:
                st.warning(f"⚠️ No registrado: confianza {conf:.2f} < umbral {thr:.2f}")

# -----------------------------
# SECCIÓN: Administración
# -----------------------------
elif menu == "👥 Administración":
    st.header("👥 Gestión de Personas")
    
    tab1, tab2 = st.tabs(["➕ Agregar Persona", "✏️ Editar/Eliminar"])
    
    with tab1:
        st.subheader("Registrar nueva persona")
        with st.form("form_add", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                label = st.text_input("🏷️ Etiqueta (del modelo)", placeholder="Ej: Class 1")
                name = st.text_input("👤 Nombre completo", placeholder="Ej: Juan Pérez")
                email = st.text_input("📧 Correo electrónico", placeholder="juan@example.com")
            with col2:
                role = st.text_input("💼 Rol/Cargo", placeholder="Ej: Estudiante")
                threshold = st.slider("🎯 Umbral de confianza", 0.0, 1.0, 0.5, 0.05)
                notes = st.text_area("📝 Notas adicionales", placeholder="Información relevante...")
            
            submitted = st.form_submit_button("✅ Agregar Persona", use_container_width=True)
            
            if submitted:
                if not label:
                    st.error("⚠️ La etiqueta es obligatoria")
                else:
                    c = conn.cursor()
                    try:
                        c.execute(
                            "INSERT INTO people (label,name,email,role,threshold,notes) VALUES (?,?,?,?,?,?)",
                            (label, name, email, role, threshold, notes)
                        )
                        conn.commit()
                        st.success(f"✅ Persona '{name or label}' agregada exitosamente")
                    except sqlite3.IntegrityError:
                        st.error("❌ Esta etiqueta ya existe. Usa la pestaña Editar.")
    
    with tab2:
        st.subheader("Modificar o eliminar persona")
        df_people = pd.read_sql_query("SELECT * FROM people ORDER BY label", conn)
        
        if df_people.empty:
            st.info("ℹ️ No hay personas registradas. Agrega una en la pestaña anterior.")
        else:
            st.dataframe(df_people[['label', 'name', 'email', 'role', 'threshold']], use_container_width=True)
            
            sel = st.selectbox("Selecciona una persona para editar:", df_people['label'].tolist())
            row = df_people[df_people['label'] == sel].iloc[0]
            
            with st.form("form_edit"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("👤 Nombre completo", value=row['name'] or "")
                    email = st.text_input("📧 Correo electrónico", value=row['email'] or "")
                    role = st.text_input("💼 Rol/Cargo", value=row['role'] or "")
                with col2:
                    threshold = st.slider("🎯 Umbral de confianza", 0.0, 1.0, float(row['threshold'] or 0.5), 0.05)
                    notes = st.text_area("📝 Notas adicionales", value=row['notes'] or "")
                
                col_save, col_delete = st.columns(2)
                with col_save:
                    save = st.form_submit_button("💾 Guardar Cambios", use_container_width=True)
                with col_delete:
                    delete = st.form_submit_button("🗑️ Eliminar", use_container_width=True, type="secondary")
                
                if save:
                    c = conn.cursor()
                    c.execute(
                        "UPDATE people SET name=?,email=?,role=?,threshold=?,notes=? WHERE label=?",
                        (name, email, role, threshold, notes, sel)
                    )
                    conn.commit()
                    st.success("✅ Cambios guardados exitosamente")
                    st.rerun()
                
                if delete:
                    if st.session_state.get('confirm_delete') == sel:
                        c = conn.cursor()
                        c.execute("DELETE FROM people WHERE label=?", (sel,))
                        conn.commit()
                        st.success("✅ Persona eliminada")
                        st.session_state.confirm_delete = None
                        st.rerun()
                    else:
                        st.session_state.confirm_delete = sel
                        st.warning("⚠️ Haz clic en Eliminar nuevamente para confirmar")

# -----------------------------
# SECCIÓN: Analítica (7 gráficas)
# -----------------------------
elif menu == "📈 Analítica":
    st.header("📈 Panel de Análisis y Estadísticas")
    
    df_pred = pd.read_sql_query("SELECT * FROM predictions", conn)
    
    if df_pred.empty:
        st.warning("⚠️ No hay predicciones registradas. Usa la sección 'En vivo' para generar datos.")
    else:
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predicciones", len(df_pred))
        with col2:
            st.metric("Personas Únicas", df_pred['label'].nunique())
        with col3:
            avg_conf = df_pred['confidence'].mean() * 100
            st.metric("Confianza Promedio", f"{avg_conf:.1f}%")
        with col4:
            high_conf = (df_pred['confidence'] >= 0.8).sum()
            st.metric("Alta Confianza (≥80%)", high_conf)
        
        st.markdown("---")
        
        # GRÁFICA 1: Detecciones por etiqueta
        st.subheader("1️⃣ Detecciones por Persona")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        counts = df_pred['label'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        counts.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
        ax1.set_title('Número de Detecciones por Persona', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Persona', fontsize=12)
        ax1.set_ylabel('Cantidad de Detecciones', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        fig1.tight_layout()
        fig1_path = os.path.join(GRAPHS_DIR, '01_detecciones_por_persona.png')
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig1)
        
        # GRÁFICA 2: Confianza promedio por etiqueta
        st.subheader("2️⃣ Nivel de Confianza por Persona")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        avg_conf_by_label = df_pred.groupby('label')['confidence'].mean().sort_values(ascending=False) * 100
        colors2 = ['#2ecc71' if x >= 70 else '#f39c12' if x >= 50 else '#e74c3c' for x in avg_conf_by_label]
        avg_conf_by_label.plot(kind='barh', ax=ax2, color=colors2, edgecolor='black')
        ax2.set_title('Confianza Promedio por Persona (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confianza Promedio (%)', fontsize=12)
        ax2.set_ylabel('Persona', fontsize=12)
        ax2.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Umbral Alto (70%)')
        ax2.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Umbral Medio (50%)')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        fig2.tight_layout()
        fig2_path = os.path.join(GRAPHS_DIR, '02_confianza_promedio.png')
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig2)
        
        # GRÁFICA 3: Serie temporal por día
        st.subheader("3️⃣ Evolución Temporal de Detecciones")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        daily_counts = df_pred.set_index('timestamp').resample('D').size()
        ax3.plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax3.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='#3498db')
        ax3.set_title('Detecciones por Día', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Fecha', fontsize=12)
        ax3.set_ylabel('Número de Detecciones', fontsize=12)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3_path = os.path.join(GRAPHS_DIR, '03_serie_temporal.png')
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig3)
        
        # GRÁFICA 4: Distribución de confianza (histograma mejorado)
        st.subheader("4️⃣ Distribución del Nivel de Confianza")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        confidence_pct = df_pred['confidence'] * 100
        n, bins, patches = ax4.hist(confidence_pct, bins=20, edgecolor='black', alpha=0.7)
        
        # Colorear por rangos
        for i, patch in enumerate(patches):
            if bins[i] >= 70:
                patch.set_facecolor('#2ecc71')
            elif bins[i] >= 50:
                patch.set_facecolor('#f39c12')
            else:
                patch.set_facecolor('#e74c3c')
        
        ax4.axvline(x=confidence_pct.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {confidence_pct.mean():.1f}%')
        ax4.axvline(x=confidence_pct.median(), color='blue', linestyle='--', linewidth=2, label=f'Mediana: {confidence_pct.median():.1f}%')
        ax4.set_title('Distribución de Niveles de Confianza', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Confianza (%)', fontsize=12)
        ax4.set_ylabel('Frecuencia', fontsize=12)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        fig4.tight_layout()
        fig4_path = os.path.join(GRAPHS_DIR, '04_distribucion_confianza.png')
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig4)
        
        # GRÁFICA 5: Fuente de predicciones (mejorado)
        st.subheader("5️⃣ Fuente de las Predicciones")
        fig5, ax5 = plt.subplots(figsize=(8, 8))
        source_counts = df_pred['source'].value_counts()
        colors5 = ['#3498db', '#e74c3c', '#f39c12'][:len(source_counts)]
        wedges, texts, autotexts = ax5.pie(
            source_counts,
            labels=source_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors5,
            explode=[0.05] * len(source_counts),
            shadow=True
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        ax5.set_title('Distribución por Fuente', fontsize=14, fontweight='bold')
        fig5.tight_layout()
        fig5_path = os.path.join(GRAPHS_DIR, '05_fuente_predicciones.png')
        fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig5)
        
        # GRÁFICA 6: Detecciones por hora del día
        st.subheader("6️⃣ Distribución Horaria de Detecciones")
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        df_pred['hour'] = df_pred['timestamp'].dt.hour
        hourly_counts = df_pred['hour'].value_counts().sort_index()
        ax6.bar(hourly_counts.index, hourly_counts.values, color='#9b59b6', edgecolor='black', alpha=0.8)
        ax6.set_title('Detecciones por Hora del Día', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Hora (0-23)', fontsize=12)
        ax6.set_ylabel('Cantidad de Detecciones', fontsize=12)
        ax6.set_xticks(range(24))
        ax6.grid(axis='y', alpha=0.3)
        fig6.tight_layout()
        fig6_path = os.path.join(GRAPHS_DIR, '06_detecciones_por_hora.png')
        fig6.savefig(fig6_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig6)
        
        # GRÁFICA 7: Top personas con detecciones de alta confianza
        st.subheader("7️⃣ Personas con Mayor Precisión (≥80% confianza)")
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        high_conf_df = df_pred[df_pred['confidence'] >= 0.8]
        if not high_conf_df.empty:
            top_high_conf = high_conf_df['label'].value_counts().head(10)
            ax7.barh(range(len(top_high_conf)), top_high_conf.values, color='#27ae60', edgecolor='black')
            ax7.set_yticks(range(len(top_high_conf)))
            ax7.set_yticklabels(top_high_conf.index)
            ax7.set_xlabel('Cantidad de Detecciones de Alta Confianza', fontsize=12)
            ax7.set_ylabel('Persona', fontsize=12)
            ax7.set_title('Top Personas con Detecciones de Alta Precisión', fontsize=14, fontweight='bold')
            ax7.grid(axis='x', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No hay detecciones con confianza ≥80%', 
                    ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        fig7.tight_layout()
        fig7_path = os.path.join(GRAPHS_DIR, '07_top_alta_confianza.png')
        fig7.savefig(fig7_path, dpi=300, bbox_inches='tight')
        st.pyplot(fig7)
        
        # Tabla de registros detallada
        st.markdown("---")
        st.subheader("📋 Registro Detallado de Predicciones")
        
        # Filtros
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            filter_label = st.multiselect("Filtrar por persona:", options=['Todos'] + df_pred['label'].unique().tolist(), default=['Todos'])
        with col_filter2:
            filter_source = st.multiselect("Filtrar por fuente:", options=['Todos'] + df_pred['source'].unique().tolist(), default=['Todos'])
        with col_filter3:
            min_confidence = st.slider("Confianza mínima:", 0.0, 1.0, 0.0, 0.05)
        
        # Aplicar filtros
        df_filtered = df_pred.copy()
        if 'Todos' not in filter_label:
            df_filtered = df_filtered[df_filtered['label'].isin(filter_label)]
        if 'Todos' not in filter_source:
            df_filtered = df_filtered[df_filtered['source'].isin(filter_source)]
        df_filtered = df_filtered[df_filtered['confidence'] >= min_confidence]
        
        # Formatear para mostrar
        df_display = df_filtered.copy()
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x*100:.2f}%")
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            df_display[['timestamp', 'label', 'source', 'confidence']].sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=400
        )
        
        st.caption(f"Mostrando {len(df_filtered)} de {len(df_pred)} registros")

# -----------------------------
# SECCIÓN: Exportar
# -----------------------------
elif menu == "📦 Exportar":
    st.header("📦 Exportación y Descarga de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Exportar Base de Datos")
        
        # CSV de predicciones
        df_all = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        if not df_all.empty:
            csv_bytes = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Predicciones (CSV)",
                data=csv_bytes,
                file_name=f'predicciones_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
            st.success(f"✅ {len(df_all)} registros listos para exportar")
        else:
            st.info("ℹ️ No hay predicciones para exportar")
        
        # CSV de personas
        df_people = pd.read_sql_query("SELECT * FROM people ORDER BY label", conn)
        if not df_people.empty:
            csv_people = df_people.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Personas (CSV)",
                data=csv_people,
                file_name=f'personas_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Exportar toda la base de datos
        if st.button("💾 Exportar Base de Datos Completa (SQLite)", use_container_width=True):
            with open(DB_FILENAME, 'rb') as f:
                st.download_button(
                    label="⬇️ Descargar predictions.db",
                    data=f,
                    file_name=f'predictions_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db',
                    mime='application/x-sqlite3',
                    use_container_width=True
                )
    
    with col2:
        st.subheader("📊 Exportar Gráficas")
        
        if os.path.exists(GRAPHS_DIR) and os.listdir(GRAPHS_DIR):
            num_graphs = len([f for f in os.listdir(GRAPHS_DIR) if f.endswith('.png')])
            st.info(f"📈 {num_graphs} gráficas disponibles")
            
            if st.button("📦 Generar ZIP con todas las gráficas", use_container_width=True):
                with st.spinner("Generando archivo ZIP..."):
                    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fname in os.listdir(GRAPHS_DIR):
                            if fname.endswith('.png'):
                                file_path = os.path.join(GRAPHS_DIR, fname)
                                zf.write(file_path, arcname=fname)
                    
                    with open(ZIP_PATH, 'rb') as f:
                        st.download_button(
                            label="⬇️ Descargar graphs.zip",
                            data=f,
                            file_name=f'graficas_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                            mime='application/zip',
                            use_container_width=True
                        )
                    st.success("✅ ZIP generado exitosamente")
        else:
            st.warning("⚠️ No hay gráficas generadas. Ve a la sección 'Analítica' primero.")
        
        st.markdown("---")
        
        # Exportar imágenes capturadas
        if os.path.exists(IMAGES_DIR) and os.listdir(IMAGES_DIR):
            num_images = len([f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
            st.info(f"📸 {num_images} imágenes capturadas")
            
            if st.button("📦 Generar ZIP con imágenes capturadas", use_container_width=True):
                images_zip = os.path.join(OUTPUTS_DIR, "images.zip")
                with st.spinner("Generando archivo ZIP de imágenes..."):
                    with zipfile.ZipFile(images_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fname in os.listdir(IMAGES_DIR):
                            if fname.endswith('.png'):
                                file_path = os.path.join(IMAGES_DIR, fname)
                                zf.write(file_path, arcname=fname)
                    
                    with open(images_zip, 'rb') as f:
                        st.download_button(
                            label="⬇️ Descargar images.zip",
                            data=f,
                            file_name=f'imagenes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                            mime='application/zip',
                            use_container_width=True
                        )
                    st.success("✅ ZIP de imágenes generado")
    
    # Estadísticas generales
    st.markdown("---")
    st.subheader("📊 Resumen del Sistema")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    df_stats = pd.read_sql_query("SELECT COUNT(*) as total FROM predictions", conn)
    df_people_count = pd.read_sql_query("SELECT COUNT(*) as total FROM people", conn)
    
    with col_stat1:
        st.metric("Total Predicciones", df_stats['total'].iloc[0])
    with col_stat2:
        st.metric("Personas Registradas", df_people_count['total'].iloc[0])
    with col_stat3:
        db_size = os.path.getsize(DB_FILENAME) / 1024  # KB
        st.metric("Tamaño BD", f"{db_size:.1f} KB")
    with col_stat4:
        if os.path.exists(GRAPHS_DIR):
            num_files = len(os.listdir(GRAPHS_DIR))
            st.metric("Archivos Generados", num_files)
        else:
            st.metric("Archivos Generados", 0)
    
    # Instrucciones de entrega
    st.markdown("---")
    st.subheader("📝 Instrucciones para la Entrega")
    
    with st.expander("📋 Ver checklist completo", expanded=False):
        st.markdown("""
        ### ✅ Checklist de Entrega
        
        #### 1. Repositorio GitHub
        - [ ] Crear repositorio público
        - [ ] Subir `reconocimiento_personas_mejorado.py`
        - [ ] Incluir `requirements.txt` con todas las dependencias
        - [ ] Agregar `keras_model.h5` (o usar Git LFS si es muy grande)
        - [ ] Incluir `labels.txt`
        - [ ] Crear carpeta `outputs/` vacía (con `.gitkeep`)
        - [ ] Agregar `.gitignore` (incluir `.venv/`, `*.db`, `outputs/graphs/`, `outputs/images/`)
        - [ ] Crear `README.md` con instrucciones de instalación
        
        #### 2. Deployment en Streamlit Cloud
        - [ ] Conectar repositorio GitHub con Streamlit Cloud
        - [ ] Verificar que `requirements.txt` esté correcto
        - [ ] Probar que la aplicación funcione online
        - [ ] Copiar el link público de la aplicación
        
        #### 3. Informe (PDF o Word)
        - [ ] **Descripción del modelo**: Explicar cómo entrenaste el modelo en Teachable Machine
        - [ ] **Ejemplos de imágenes**: Screenshots de las clases/personas entrenadas
        - [ ] **Capturas de pantalla**: 
            - Sección "En vivo" funcionando
            - Sección "Administración" con personas registradas
            - Sección "Analítica" con las 7 gráficas
        - [ ] **Análisis de resultados**: Breve interpretación de las gráficas
        - [ ] **Link de GitHub**: URL del repositorio
        - [ ] **Link de Streamlit Cloud**: URL de la app desplegada
        
        #### 4. Verificación Final
        - [ ] Todas las 7 gráficas se generan correctamente
        - [ ] El sistema CRUD funciona (agregar, editar, eliminar personas)
        - [ ] Las predicciones se guardan en SQLite
        - [ ] La exportación CSV funciona
        - [ ] El ZIP con gráficas se genera
        - [ ] La cámara o captura de fotos funciona
        """)
    
    st.info("""
    💡 **Tip**: Asegúrate de incluir en tu `.gitignore`:
    ```
    .venv/
    *.db
    outputs/graphs/*.png
    outputs/images/*.png
    __pycache__/
    *.pyc
    ```
    """)
    
    st.success("""
    ✅ **Tu aplicación incluye**:
    - ✔️ Modelo Teachable Machine funcional
    - ✔️ Interfaz Streamlit con 4 secciones
    - ✔️ Base de datos SQLite con 2 tablas
    - ✔️ CRUD completo de personas
    - ✔️ **7 gráficas analíticas** (cumple requisito de mínimo 5)
    - ✔️ Exportación CSV y ZIP
    - ✔️ Guardado automático de imágenes
    - ✔️ Diseño mejorado y profesional
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p><strong>Sistema de Reconocimiento Facial v2.0</strong></p>
    <p>Desarrollado con Streamlit + TensorFlow + Teachable Machine</p>
</div>
""", unsafe_allow_html=True)