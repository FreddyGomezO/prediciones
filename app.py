"""
Sistema de Reconocimiento de Personas
Aplicación Streamlit con Teachable Machine (Keras)
Autor: [Tu Nombre]
Fecha: Octubre 2025

Descripción:
Sistema integral para reconocimiento facial que incluye:
- Detección en tiempo real via webcam
- Gestión completa de personas (CRUD)
- Analytics con múltiples visualizaciones
- Exportación de datos y gráficas
"""

import os
import io
import time
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model

# ==================== CONFIGURACIÓN ====================
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DB_PATH = "recognition_system.db"
EXPORT_FOLDER = Path("exports")
GRAPHS_FOLDER = EXPORT_FOLDER / "graphs"
IMAGES_FOLDER = EXPORT_FOLDER / "images"

# Crear carpetas necesarias
EXPORT_FOLDER.mkdir(exist_ok=True)
GRAPHS_FOLDER.mkdir(exist_ok=True)
IMAGES_FOLDER.mkdir(exist_ok=True)

INPUT_SHAPE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7

# Configuración de página
st.set_page_config(
    page_title="Sistema Reconocimiento Personas",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== BASE DE DATOS ====================
def initialize_database():
    """Inicializa la base de datos SQLite con las tablas necesarias"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # Tabla de personas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etiqueta TEXT UNIQUE NOT NULL,
            nombre_completo TEXT NOT NULL,
            correo_electronico TEXT,
            rol TEXT,
            umbral_confianza REAL DEFAULT 0.7,
            notas TEXT,
            fecha_registro TEXT,
            activo INTEGER DEFAULT 1
        )
    """)
    
    # Tabla de predicciones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT NOT NULL,
            fuente TEXT NOT NULL,
            etiqueta_detectada TEXT NOT NULL,
            nivel_confianza REAL NOT NULL,
            umbral_aplicado REAL,
            aprobada INTEGER,
            imagen_guardada TEXT
        )
    """)
    
    # Tabla de sesiones (para tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sesiones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_inicio TEXT,
            fecha_fin TEXT,
            total_detecciones INTEGER DEFAULT 0,
            personas_unicas INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    return conn

# ==================== CARGA DE MODELO ====================
@st.cache_resource(show_spinner="Cargando modelo de reconocimiento...")
def cargar_modelo_keras():
    """Carga el modelo de Keras entrenado"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ No se encontró el archivo del modelo: {MODEL_PATH}")
        st.stop()
    
    try:
        modelo = load_model(MODEL_PATH, compile=False)
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

@st.cache_data(show_spinner="Cargando etiquetas...")
def cargar_etiquetas():
    """Carga las etiquetas del archivo de texto"""
    if not os.path.exists(LABELS_PATH):
        st.error(f"❌ No se encontró el archivo de etiquetas: {LABELS_PATH}")
        st.stop()
    
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        etiquetas = [line.strip() for line in f if line.strip()]
    return etiquetas

# ==================== FUNCIONES DE PROCESAMIENTO ====================
def preprocesar_imagen(imagen_pil: Image.Image) -> np.ndarray:
    """Preprocesa una imagen PIL para el modelo"""
    img = imagen_pil.convert("RGB")
    img = img.resize(INPUT_SHAPE)
    arr = np.array(img, dtype=np.float32)
    # Normalización Teachable Machine: -1 a 1
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def realizar_prediccion(imagen_pil: Image.Image, modelo, etiquetas) -> Tuple[str, float]:
    """Realiza predicción sobre una imagen"""
    try:
        entrada = preprocesar_imagen(imagen_pil)
        predicciones = modelo.predict(entrada, verbose=0)[0]
        idx_max = np.argmax(predicciones)
        confianza = float(predicciones[idx_max])
        etiqueta = etiquetas[idx_max] if idx_max < len(etiquetas) else f"Clase_{idx_max}"
        return etiqueta, confianza
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return "Error", 0.0

def guardar_imagen_detectada(imagen: Image.Image, etiqueta: str, confianza: float) -> str:
    """Guarda imagen detectada en disco"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_archivo = f"{etiqueta}_{confianza:.2f}_{timestamp}.png"
    ruta_completa = IMAGES_FOLDER / nombre_archivo
    
    # Agregar overlay de información
    draw = ImageDraw.Draw(imagen)
    texto = f"{etiqueta} - {confianza*100:.1f}%"
    draw.rectangle([(0, 0), (300, 40)], fill=(0, 0, 0, 128))
    draw.text((10, 10), texto, fill=(255, 255, 255))
    
    imagen.save(ruta_completa)
    return str(ruta_completa)

def obtener_umbral_persona(etiqueta: str, conn) -> float:
    """Obtiene el umbral configurado para una persona"""
    cursor = conn.cursor()
    cursor.execute("SELECT umbral_confianza FROM personas WHERE etiqueta = ? AND activo = 1", (etiqueta,))
    resultado = cursor.fetchone()
    return resultado[0] if resultado else CONFIDENCE_THRESHOLD

def registrar_prediccion(fecha_hora: str, fuente: str, etiqueta: str, 
                        confianza: float, umbral: float, aprobada: bool, 
                        imagen_ruta: Optional[str], conn):
    """Registra una predicción en la base de datos"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predicciones 
        (fecha_hora, fuente, etiqueta_detectada, nivel_confianza, umbral_aplicado, aprobada, imagen_guardada)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (fecha_hora, fuente, etiqueta, confianza, umbral, int(aprobada), imagen_ruta))
    conn.commit()

# ==================== TRANSFORMADOR DE VIDEO (ACTUALIZADO) ====================
class TransformadorReconocimiento(VideoTransformerBase):
    """Clase para procesamiento de video en tiempo real - Versión actualizada"""
    
    def __init__(self):
        self.modelo = None
        self.etiquetas = None
        self.conn = None
        self.ultima_deteccion = {"etiqueta": None, "confianza": 0.0, "timestamp": None}
        self.contador_frames = 0
        self.log_interval = 30
        
    def set_components(self, modelo, etiquetas, conn):
        """Configurar componentes después de la inicialización"""
        self.modelo = modelo
        self.etiquetas = etiquetas
        self.conn = conn
        
    def recv(self, frame):
        """Procesa cada frame del video - Método actualizado"""
        if self.modelo is None or self.etiquetas is None:
            return frame
            
        self.contador_frames += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Convertir a PIL para predicción
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Realizar predicción
        etiqueta, confianza = realizar_prediccion(img_pil, self.modelo, self.etiquetas)
        self.ultima_deteccion = {
            "etiqueta": etiqueta,
            "confianza": confianza,
            "timestamp": datetime.now()
        }
        
        # Registrar en DB cada N frames si supera umbral
        if self.contador_frames % self.log_interval == 0:
            umbral = obtener_umbral_persona(etiqueta, self.conn)
            if confianza >= umbral:
                registrar_prediccion(
                    datetime.now().isoformat(),
                    "webcam_streaming",
                    etiqueta,
                    confianza,
                    umbral,
                    True,
                    None,
                    self.conn
                )
        
        # Dibujar overlay
        color = (0, 255, 0) if confianza >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        texto = f"{etiqueta}: {confianza*100:.1f}%"
        cv2.rectangle(img, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(img, texto, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return frame.from_ndarray(img, format="bgr24")

# ==================== INTERFAZ PRINCIPAL ====================
def main():
    # Inicializar recursos
    conn = initialize_database()
    modelo = cargar_modelo_keras()
    etiquetas = cargar_etiquetas()
    
    # Header
    st.markdown('<p class="main-header">👤 Sistema de Reconocimiento de Personas</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Navegación
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Recognition+AI", width='stretch')
        st.markdown("### 📋 Navegación")
        
        seccion = st.radio(
            "Selecciona una sección:",
            ["🎥 Detección en Vivo", "👥 Administración", "📊 Analítica", "📦 Exportación"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuración Global")
        guardar_imagenes = st.checkbox("💾 Guardar imágenes detectadas", value=False)
        modo_debug = st.checkbox("🐛 Modo debug", value=False)
        
        if modo_debug:
            st.info(f"Total etiquetas: {len(etiquetas)}")
            st.info(f"DB: {DB_PATH}")
    
    # ==================== SECCIÓN: DETECCIÓN EN VIVO ====================
    if seccion == "🎥 Detección en Vivo":
        st.header("🎥 Detección en Tiempo Real")
        
        tab1, tab2, tab3 = st.tabs(["📹 Webcam Streaming", "📸 Captura de Foto", "🖼️ Subir Imagen"])
        
        with tab1:
            st.subheader("Streaming de Webcam")
            
            col_config1, col_config2 = st.columns(2)
            with col_config1:
                calidad = st.selectbox("Calidad de video", ["640x480", "1280x720", "1920x1080"], index=1)
                w, h = map(int, calidad.split("x"))
            with col_config2:
                tipo_camara = st.selectbox("Tipo de cámara", ["Automático", "Frontal", "Trasera"])
                facing_mode = {"Automático": None, "Frontal": "user", "Trasera": "environment"}[tipo_camara]
            
            media_constraints = {
                "video": {"width": w, "height": h, "facingMode": facing_mode} if facing_mode else {"width": w, "height": h},
                "audio": False
            }
            
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            
            col_video, col_info = st.columns([2, 1])
            
            with col_video:
                ctx = webrtc_streamer(
                    key="reconocimiento_facial",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_config,
                    media_stream_constraints=media_constraints,
                    video_processor_factory=TransformadorReconocimiento,
                    async_processing=True
                )
                
                # Configurar componentes después de crear el contexto
                if ctx.video_processor:
                    ctx.video_processor.set_components(modelo, etiquetas, conn)
            
            with col_info:
                st.markdown("### 📊 Información en Tiempo Real")
                placeholder_deteccion = st.empty()
                placeholder_confianza = st.empty()
                placeholder_progreso = st.empty()
                
                if ctx and ctx.video_processor:
                    while ctx.state.playing:
                        procesador = ctx.video_processor
                        det = procesador.ultima_deteccion
                        
                        if det["etiqueta"]:
                            placeholder_deteccion.metric("Persona Detectada", det["etiqueta"])
                            placeholder_confianza.metric("Nivel de Confianza", f"{det['confianza']*100:.2f}%")
                            placeholder_progreso.progress(det["confianza"])
                        else:
                            placeholder_deteccion.info("Esperando detección...")
                        
                        time.sleep(0.3)
                else:
                    placeholder_deteccion.warning("Inicia la cámara para ver detecciones")
        
        with tab2:
            st.subheader("Captura de Foto desde Cámara")
            foto_capturada = st.camera_input("Toma una foto")
            
            if foto_capturada:
                img_pil = Image.open(foto_capturada)
                etiqueta, confianza = realizar_prediccion(img_pil, modelo, etiquetas)
                umbral = obtener_umbral_persona(etiqueta, conn)
                aprobada = confianza >= umbral
                
                col_img, col_result = st.columns(2)
                with col_img:
                    st.image(img_pil, caption="Imagen capturada", width='stretch')
                
                with col_result:
                    st.metric("Persona Detectada", etiqueta)
                    st.metric("Confianza", f"{confianza*100:.2f}%")
                    st.progress(confianza)
                    
                    if aprobada:
                        st.success(f"✅ Detección aprobada (Umbral: {umbral*100:.1f}%)")
                    else:
                        st.warning(f"⚠️ Confianza por debajo del umbral ({umbral*100:.1f}%)")
                
                # Guardar
                imagen_ruta = None
                if guardar_imagenes:
                    imagen_ruta = guardar_imagen_detectada(img_pil, etiqueta, confianza)
                    st.info(f"💾 Imagen guardada: {imagen_ruta}")
                
                registrar_prediccion(
                    datetime.now().isoformat(),
                    "captura_foto",
                    etiqueta,
                    confianza,
                    umbral,
                    aprobada,
                    imagen_ruta,
                    conn
                )
        
        with tab3:
            st.subheader("Subir Imagen para Análisis")
            archivo_subido = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])
            
            if archivo_subido:
                img_pil = Image.open(archivo_subido)
                etiqueta, confianza = realizar_prediccion(img_pil, modelo, etiquetas)
                umbral = obtener_umbral_persona(etiqueta, conn)
                aprobada = confianza >= umbral
                
                col_img, col_result = st.columns(2)
                with col_img:
                    st.image(img_pil, caption="Imagen subida", width='stretch')
                
                with col_result:
                    st.metric("Persona Detectada", etiqueta)
                    st.metric("Confianza", f"{confianza*100:.2f}%")
                    st.progress(confianza)
                    
                    if aprobada:
                        st.success(f"✅ Detección aprobada (Umbral: {umbral*100:.1f}%)")
                    else:
                        st.warning(f"⚠️ Confianza por debajo del umbral ({umbral*100:.1f}%)")
                
                # Guardar
                imagen_ruta = None
                if guardar_imagenes:
                    imagen_ruta = guardar_imagen_detectada(img_pil, etiqueta, confianza)
                    st.info(f"💾 Imagen guardada: {imagen_ruta}")
                
                registrar_prediccion(
                    datetime.now().isoformat(),
                    "imagen_subida",
                    etiqueta,
                    confianza,
                    umbral,
                    aprobada,
                    imagen_ruta,
                    conn
                )
    
    # ==================== SECCIÓN: ADMINISTRACIÓN ====================
    elif seccion == "👥 Administración":
        st.header("👥 Gestión de Personas")
        
        tab_add, tab_manage, tab_view = st.tabs(["➕ Agregar Persona", "✏️ Editar/Eliminar", "📋 Ver Todas"])
        
        with tab_add:
            st.subheader("Registrar Nueva Persona")
            with st.form("form_agregar_persona", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    etiqueta = st.text_input("Etiqueta del Modelo*", help="Debe coincidir con la etiqueta del modelo")
                    nombre = st.text_input("Nombre Completo*")
                    correo = st.text_input("Correo Electrónico")
                
                with col2:
                    rol = st.text_input("Rol/Cargo")
                    umbral = st.slider("Umbral de Confianza", 0.0, 1.0, 0.7, 0.05)
                    activo = st.checkbox("Activo", value=True)
                
                notas = st.text_area("Notas adicionales")
                
                submitted = st.form_submit_button("💾 Registrar Persona", width='stretch')
                
                if submitted:
                    if not etiqueta or not nombre:
                        st.error("⚠️ La etiqueta y el nombre son obligatorios")
                    else:
                        try:
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO personas 
                                (etiqueta, nombre_completo, correo_electronico, rol, umbral_confianza, notas, fecha_registro, activo)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (etiqueta, nombre, correo, rol, umbral, notas, datetime.now().isoformat(), int(activo)))
                            conn.commit()
                            st.success(f"✅ Persona '{nombre}' registrada exitosamente")
                        except sqlite3.IntegrityError:
                            st.error(f"❌ La etiqueta '{etiqueta}' ya existe en el sistema")
        
        with tab_manage:
            st.subheader("Editar o Eliminar Persona")
            
            df_personas = pd.read_sql_query("SELECT * FROM personas ORDER BY nombre_completo", conn)
            
            if df_personas.empty:
                st.info("No hay personas registradas aún")
            else:
                persona_seleccionada = st.selectbox(
                    "Selecciona una persona",
                    df_personas['nombre_completo'].tolist()
                )
                
                persona_data = df_personas[df_personas['nombre_completo'] == persona_seleccionada].iloc[0]
                
                with st.form("form_editar"):
                    st.markdown(f"**Etiqueta:** `{persona_data['etiqueta']}`")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        nombre_edit = st.text_input("Nombre Completo", value=persona_data['nombre_completo'])
                        correo_edit = st.text_input("Correo", value=persona_data['correo_electronico'] or "")
                        rol_edit = st.text_input("Rol", value=persona_data['rol'] or "")
                    
                    with col2:
                        umbral_edit = st.slider("Umbral", 0.0, 1.0, float(persona_data['umbral_confianza']), 0.05)
                        activo_edit = st.checkbox("Activo", value=bool(persona_data['activo']))
                    
                    notas_edit = st.text_area("Notas", value=persona_data['notas'] or "")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        guardar = st.form_submit_button("💾 Guardar Cambios", width='stretch')
                    with col_btn2:
                        eliminar = st.form_submit_button("🗑️ Eliminar", type="secondary", width='stretch')
                    
                    if guardar:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE personas 
                            SET nombre_completo=?, correo_electronico=?, rol=?, 
                                umbral_confianza=?, notas=?, activo=?
                            WHERE etiqueta=?
                        """, (nombre_edit, correo_edit, rol_edit, umbral_edit, notas_edit, int(activo_edit), persona_data['etiqueta']))
                        conn.commit()
                        st.success("✅ Cambios guardados correctamente")
                        st.rerun()
                    
                    if eliminar:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM personas WHERE etiqueta=?", (persona_data['etiqueta'],))
                        conn.commit()
                        st.success("✅ Persona eliminada")
                        st.rerun()
        
        with tab_view:
            st.subheader("Listado Completo de Personas")
            
            df_todas = pd.read_sql_query("""
                SELECT 
                    etiqueta, nombre_completo, correo_electronico, rol, 
                    umbral_confianza, fecha_registro, activo
                FROM personas 
                ORDER BY nombre_completo
            """, conn)
            
            if df_todas.empty:
                st.info("No hay personas registradas")
            else:
                st.dataframe(
                    df_todas,
                    width='stretch',
                    column_config={
                        "umbral_confianza": st.column_config.ProgressColumn("Umbral", format="%.2f", min_value=0, max_value=1),
                        "activo": st.column_config.CheckboxColumn("Activo")
                    }
                )
    
    # ==================== SECCIÓN: ANALÍTICA ====================
    elif seccion == "📊 Analítica":
        st.header("📊 Panel de Analítica y Estadísticas")
        
        df_predicciones = pd.read_sql_query("""
            SELECT * FROM predicciones 
            ORDER BY fecha_hora DESC
        """, conn)
        
        if df_predicciones.empty:
            st.warning("⚠️ No hay datos de predicciones aún. Usa la sección 'Detección en Vivo' para generar datos.")
            st.stop()
        
        df_predicciones['fecha_hora'] = pd.to_datetime(df_predicciones['fecha_hora'])
        
        # Métricas generales
        st.subheader("📈 Métricas Generales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predicciones = len(df_predicciones)
            st.metric("Total Predicciones", total_predicciones)
        
        with col2:
            personas_detectadas = df_predicciones['etiqueta_detectada'].nunique()
            st.metric("Personas Únicas", personas_detectadas)
        
        with col3:
            aprobadas = df_predicciones['aprobada'].sum()
            st.metric("Detecciones Aprobadas", aprobadas)
        
        with col4:
            confianza_promedio = df_predicciones['nivel_confianza'].mean()
            st.metric("Confianza Promedio", f"{confianza_promedio*100:.1f}%")
        
        st.markdown("---")
        
        # Filtros
        st.subheader("🔍 Filtros")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            fecha_inicio = st.date_input("Desde", value=df_predicciones['fecha_hora'].min())
        with col_f2:
            fecha_fin = st.date_input("Hasta", value=df_predicciones['fecha_hora'].max())
        with col_f3:
            fuente_filtro = st.multiselect("Fuente", df_predicciones['fuente'].unique(), default=df_predicciones['fuente'].unique())
        
        # Aplicar filtros
        df_filtrado = df_predicciones[
            (df_predicciones['fecha_hora'].dt.date >= fecha_inicio) &
            (df_predicciones['fecha_hora'].dt.date <= fecha_fin) &
            (df_predicciones['fuente'].isin(fuente_filtro))
        ]
        
        st.markdown("---")
        
        # Gráficas
        st.subheader("📊 Visualizaciones")
        
        # Gráfica 1: Detecciones por Persona
        st.markdown("#### 1️⃣ Detecciones por Persona")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        conteo_personas = df_filtrado['etiqueta_detectada'].value_counts()
        conteo_personas.plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel('Número de Detecciones')
        ax1.set_ylabel('Persona')
        ax1.set_title('Total de Detecciones por Persona')
        ax1.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        fig1_path = GRAPHS_FOLDER / "1_detecciones_por_persona.png"
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        st.pyplot(fig1)
        plt.close()
        
        # Gráfica 2: Confianza Promedio por Persona
        st.markdown("#### 2️⃣ Nivel de Confianza Promedio por Persona")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        confianza_promedio_persona = df_filtrado.groupby('etiqueta_detectada')['nivel_confianza'].mean().sort_values(ascending=False)
        confianza_promedio_persona.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_ylabel('Confianza Promedio')
        ax2.set_xlabel('Persona')
        ax2.set_title('Confianza Promedio por Persona')
        ax2.axhline(y=CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'Umbral Global ({CONFIDENCE_THRESHOLD})')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig2_path = GRAPHS_FOLDER / "2_confianza_promedio_persona.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        st.pyplot(fig2)
        plt.close()
        
        # Gráfica 3: Serie Temporal de Detecciones
        st.markdown("#### 3️⃣ Detecciones a lo Largo del Tiempo")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        df_temporal = df_filtrado.set_index('fecha_hora').resample('H').size()
        df_temporal.plot(ax=ax3, color='green', marker='o', linestyle='-', linewidth=2)
        ax3.set_xlabel('Fecha y Hora')
        ax3.set_ylabel('Número de Detecciones')
        ax3.set_title('Detecciones por Hora')
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig3_path = GRAPHS_FOLDER / "3_serie_temporal_detecciones.png"
        fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
        st.pyplot(fig3)
        plt.close()
        
        # Gráfica 4: Distribución de Confianza
        st.markdown("#### 4️⃣ Distribución de Niveles de Confianza")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.hist(df_filtrado['nivel_confianza'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(x=CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Umbral ({CONFIDENCE_THRESHOLD})')
        ax4.set_xlabel('Nivel de Confianza')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de Niveles de Confianza')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig4_path = GRAPHS_FOLDER / "4_distribucion_confianza.png"
        fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
        st.pyplot(fig4)
        plt.close()
        
        # Gráfica 5: Fuentes de Detección
        st.markdown("#### 5️⃣ Distribución por Fuente de Detección")
        fig5, ax5 = plt.subplots(figsize=(8, 8))
        fuentes_count = df_filtrado['fuente'].value_counts()
        colores = plt.cm.Set3(range(len(fuentes_count)))
        ax5.pie(fuentes_count, labels=fuentes_count.index, autopct='%1.1f%%', startangle=90, colors=colores)
        ax5.set_title('Distribución de Predicciones por Fuente')
        plt.tight_layout()
        fig5_path = GRAPHS_FOLDER / "5_fuentes_deteccion.png"
        fig5.savefig(fig5_path, dpi=150, bbox_inches='tight')
        st.pyplot(fig5)
        plt.close()
        
        # Gráfica 6: Tasa de Aprobación por Persona
        st.markdown("#### 6️⃣ Tasa de Aprobación por Persona")
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        tasa_aprobacion = df_filtrado.groupby('etiqueta_detectada').apply(
            lambda x: (x['aprobada'].sum() / len(x)) * 100
        ).sort_values(ascending=False)
        tasa_aprobacion.plot(kind='bar', ax=ax6, color='teal')
        ax6.set_ylabel('Tasa de Aprobación (%)')
        ax6.set_xlabel('Persona')
        ax6.set_title('Tasa de Aprobación por Persona')
        ax6.axhline(y=80, color='orange', linestyle='--', label='Meta 80%')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig6_path = GRAPHS_FOLDER / "6_tasa_aprobacion.png"
        fig6.savefig(fig6_path, dpi=150, bbox_inches='tight')
        st.pyplot(fig6)
        plt.close()
        
        st.markdown("---")
        
        # Tabla de datos detallados
        st.subheader("📋 Datos Detallados")
        st.dataframe(
            df_filtrado[['fecha_hora', 'fuente', 'etiqueta_detectada', 'nivel_confianza', 'aprobada']].sort_values('fecha_hora', ascending=False),
            width='stretch',
            column_config={
                "fecha_hora": "Fecha/Hora",
                "fuente": "Fuente",
                "etiqueta_detectada": "Persona",
                "nivel_confianza": st.column_config.ProgressColumn("Confianza", format="%.2f%%", min_value=0, max_value=1),
                "aprobada": st.column_config.CheckboxColumn("Aprobada")
            }
        )
    
    # ==================== SECCIÓN: EXPORTACIÓN ====================
    elif seccion == "📦 Exportación":
        st.header("📦 Exportación de Datos y Gráficas")
        
        st.markdown("""
        Esta sección te permite exportar todos los datos y visualizaciones generadas por el sistema.
        """)
        
        # Exportar CSV de predicciones
        st.subheader("📄 Exportar Datos de Predicciones")
        
        df_export_pred = pd.read_sql_query("""
            SELECT 
                p.fecha_hora,
                p.fuente,
                p.etiqueta_detectada,
                per.nombre_completo,
                per.rol,
                p.nivel_confianza,
                p.umbral_aplicado,
                p.aprobada,
                p.imagen_guardada
            FROM predicciones p
            LEFT JOIN personas per ON p.etiqueta_detectada = per.etiqueta
            ORDER BY p.fecha_hora DESC
        """, conn)
        
        if not df_export_pred.empty:
            col_csv1, col_csv2 = st.columns(2)
            
            with col_csv1:
                csv_predicciones = df_export_pred.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV de Predicciones",
                    data=csv_predicciones,
                    file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            with col_csv2:
                st.metric("Total de Registros", len(df_export_pred))
            
            with st.expander("👁️ Vista Previa de Datos"):
                st.dataframe(df_export_pred.head(50), width='stretch')
        else:
            st.info("No hay predicciones para exportar")
        
        st.markdown("---")
        
        # Exportar CSV de personas
        st.subheader("👥 Exportar Datos de Personas")
        
        df_export_personas = pd.read_sql_query("SELECT * FROM personas", conn)
        
        if not df_export_personas.empty:
            csv_personas = df_export_personas.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar CSV de Personas",
                data=csv_personas,
                file_name=f"personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        else:
            st.info("No hay personas registradas para exportar")
        
        st.markdown("---")
        
        # Exportar gráficas como ZIP
        st.subheader("📊 Exportar Gráficas (ZIP)")
        
        st.info("💡 Las gráficas se generan en la sección 'Analítica'. Visita esa sección primero para crear las visualizaciones.")
        
        archivos_graficas = list(GRAPHS_FOLDER.glob("*.png"))
        
        if archivos_graficas:
            st.success(f"✅ Se encontraron {len(archivos_graficas)} gráficas disponibles")
            
            if st.button("🗜️ Generar archivo ZIP con gráficas", width='stretch'):
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for archivo in archivos_graficas:
                        zip_file.write(archivo, arcname=archivo.name)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Descargar ZIP de Gráficas",
                    data=zip_buffer,
                    file_name=f"graficas_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width='stretch'
                )
                
                st.success("✅ Archivo ZIP generado exitosamente")
                
                with st.expander("📂 Archivos incluidos en el ZIP"):
                    for archivo in archivos_graficas:
                        st.text(f"📊 {archivo.name}")
        else:
            st.warning("⚠️ No hay gráficas disponibles. Ve a la sección 'Analítica' para generar visualizaciones.")
        
        st.markdown("---")
        
        # Exportar imágenes detectadas
        st.subheader("🖼️ Exportar Imágenes Guardadas")
        
        archivos_imagenes = list(IMAGES_FOLDER.glob("*.png"))
        
        if archivos_imagenes:
            st.success(f"✅ Se encontraron {len(archivos_imagenes)} imágenes guardadas")
            
            if st.button("🗜️ Generar archivo ZIP con imágenes", width='stretch'):
                zip_img_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_img_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for archivo in archivos_imagenes:
                        zip_file.write(archivo, arcname=archivo.name)
                
                zip_img_buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Descargar ZIP de Imágenes",
                    data=zip_img_buffer,
                    file_name=f"imagenes_detectadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width='stretch'
                )
                
                st.success("✅ Archivo ZIP de imágenes generado exitosamente")
        else:
            st.info("ℹ️ No hay imágenes guardadas. Activa 'Guardar imágenes detectadas' en la configuración global.")
        
        st.markdown("---")
        
        # Estadísticas del sistema
        st.subheader("📈 Estadísticas del Sistema")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predicciones")
            total_preds = cursor.fetchone()[0]
            st.metric("Total Predicciones", total_preds)
        
        with col_stat2:
            cursor.execute("SELECT COUNT(*) FROM personas")
            total_personas = cursor.fetchone()[0]
            st.metric("Personas Registradas", total_personas)
        
        with col_stat3:
            tamano_db = os.path.getsize(DB_PATH) / 1024  # KB
            st.metric("Tamaño Base de Datos", f"{tamano_db:.2f} KB")
        
        st.markdown("---")
        
        # Instrucciones para entrega
        st.subheader("📝 Instrucciones para Entrega del Proyecto")
        
        st.markdown("""
        ### 📋 Checklist de Entrega
        
        #### 1. Archivos requeridos en GitHub:
        - ✅ `reconocimiento_personas_streamlit.py` (este archivo)
        - ✅ `requirements.txt` (dependencias del proyecto)
        - ✅ `keras_model.h5` (modelo entrenado - usar Git LFS si es muy grande)
        - ✅ `labels.txt` (etiquetas del modelo)
        - ✅ `README.md` (documentación del proyecto)
        - ✅ `.gitignore` (excluir .venv, __pycache__, *.db, exports/)
        
        #### 2. Informe (PDF/Word) debe incluir:
        - 📄 Descripción del modelo entrenado en Teachable Machine
        - 🖼️ Ejemplos de imágenes utilizadas para entrenar cada clase
        - 📸 Capturas de pantalla de las 3 secciones principales:
          - Detección en Vivo
          - Administración de Personas
          - Analítica con las 5+ gráficas
        - 📊 Análisis breve de los resultados obtenidos
        - 🔗 Link del repositorio de GitHub
        - 🌐 Link de la aplicación en Streamlit Cloud
        
        #### 3. Despliegue en Streamlit Cloud:
        1. Crea una cuenta en [streamlit.io](https://streamlit.io)
        2. Conecta tu repositorio de GitHub
        3. Configura el archivo principal como `reconocimiento_personas_streamlit.py`
        4. Asegúrate de que `requirements.txt` esté correcto
        5. Despliega y copia el link público
        
        #### 4. Consideraciones importantes:
        - ⚠️ NO subir el entorno virtual (.venv) a GitHub
        - ⚠️ NO subir archivos .db (base de datos) a GitHub
        - ⚠️ Si el modelo es muy grande (>100MB), usar Git LFS o descargarlo desde URL
        - ✅ Documentar bien el README con instrucciones de instalación
        - ✅ Incluir capturas de pantalla en el README
        """)
        
        st.success("✅ Sistema listo para exportación y entrega")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Sistema de Reconocimiento de Personas v1.0 | Desarrollado con Streamlit + TensorFlow</p>
        <p>© 2025 - Proyecto de Exoneración</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()