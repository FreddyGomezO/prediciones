"""
Sistema de Reconocimiento de Personas
Aplicación Streamlit con Teachable Machine (Keras)
Autor: [Tu Nombre]
Fecha: Octubre 2025

Descripción:
Sistema integral para reconocimiento facial que incluye:
- Detección en tiempo real via cámara del navegador
- Gestión completa de personas (CRUD)
- Analytics con múltiples visualizaciones
- Exportación de datos y gráficas
"""

import os
import io
import time
import sqlite3
import zipfile
import base64
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
    .camera-container {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
        background: #f8f9fa;
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
        from tensorflow.keras.models import load_model
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
        etiquetas = [line.strip() for line in f.readlines()]
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
    
    # Crear un fondo para el texto
    bbox = draw.textbbox((0, 0), texto)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    draw.rectangle([(10, 10), (20 + text_width, 20 + text_height)], fill=(0, 0, 0, 128))
    draw.text((15, 15), texto, fill=(255, 255, 255))
    
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

# ==================== COMPONENTE DE CÁMARA EN VIVO ====================
def camera_component():
    """Componente de cámara en vivo usando JavaScript"""
    st.markdown("""
    <div class="camera-container">
        <h4>📹 Cámara en Vivo</h4>
        <video id="video" width="100%" autoplay></video>
        <br>
        <button id="capture" style="padding: 10px 20px; background: #1f77b4; color: white; border: none; border-radius: 5px; cursor: pointer;">
            📸 Capturar Foto
        </button>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>
    
    <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');
    
    // Acceder a la cámara
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Error accessing camera:", err);
            alert("No se pudo acceder a la cámara. Asegúrate de permitir el acceso.");
        });
    
    captureButton.addEventListener('click', () => {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convertir a base64 y enviar a Streamlit
        const imageData = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = 'captura.png';
        link.href = imageData;
        
        // Crear un input file simulado para Streamlit
        const input = document.createElement('input');
        input.type = 'file';
        input.name = 'camera_capture';
        input.style.display = 'none';
        document.body.appendChild(input);
        
        // Convertir base64 a blob y crear File object
        fetch(imageData)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], 'captura.png', { type: 'image/png' });
                const dt = new DataTransfer();
                dt.items.add(file);
                input.files = dt.files;
                
                // Disparar evento change
                input.dispatchEvent(new Event('change', { bubbles: true }));
            });
    });
    </script>
    """, unsafe_allow_html=True)

# ==================== INTERFAZ PRINCIPAL ====================
def main():
    # Inicializar recursos
    conn = initialize_database()
    
    # Cargar modelo y etiquetas con manejo de errores
    try:
        modelo = cargar_modelo_keras()
        etiquetas = cargar_etiquetas()
        modelo_cargado = True
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        modelo_cargado = False
        modelo = None
        etiquetas = []
    
    # Header
    st.markdown('<p class="main-header">👤 Sistema de Reconocimiento de Personas</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Navegación
    with st.sidebar:
        st.markdown("### 📋 Navegación")
        
        seccion = st.radio(
            "Selecciona una sección:",
            ["🎥 Detección en Vivo", "👥 Administración", "📊 Analítica", "📦 Exportación"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuración Global")
        guardar_imagenes = st.checkbox("💾 Guardar imágenes detectadas", value=False)
        
        if st.checkbox("🐛 Modo debug", value=False):
            st.info(f"Modelo cargado: {modelo_cargado}")
            st.info(f"Total etiquetas: {len(etiquetas)}")
            if etiquetas:
                st.info(f"Etiquetas: {', '.join(etiquetas)}")
    
    if not modelo_cargado:
        st.warning("⚠️ El modelo no está cargado. Algunas funciones no estarán disponibles.")
    
    # ==================== SECCIÓN: DETECCIÓN EN VIVO ====================
    if seccion == "🎥 Detección en Vivo":
        st.header("🎥 Detección en Tiempo Real")
        
        tab1, tab2, tab3 = st.tabs(["📹 Cámara en Vivo", "📸 Captura de Foto", "🖼️ Subir Imagen"])
        
        with tab1:
            st.subheader("Cámara en Vivo del Navegador")
            
            if not modelo_cargado:
                st.error("❌ El modelo no está cargado. No se puede realizar reconocimiento.")
            else:
                st.info("""
                **Instrucciones:**
                1. Permite el acceso a la cámara cuando el navegador lo solicite
                2. Posiciona tu rostro en el cuadro de la cámara
                3. Haz clic en **📸 Capturar Foto** para tomar una foto
                4. El sistema analizará la imagen automáticamente
                """)
                
                # Componente de cámara
                camera_component()
                
                # Manejar capturas de cámara
                camara_capturada = st.file_uploader("Captura de cámara", type=["png", "jpg", "jpeg"], 
                                                   key="camera_upload", label_visibility="collapsed")
                
                if camara_capturada:
                    procesar_imagen(camara_capturada, modelo, etiquetas, conn, guardar_imagenes, "camara_vivo")
        
        with tab2:
            st.subheader("Captura de Foto desde Cámara")
            
            if not modelo_cargado:
                st.error("❌ El modelo no está cargado. No se puede realizar reconocimiento.")
            else:
                foto_capturada = st.camera_input("Toma una foto con tu cámara")
                
                if foto_capturada:
                    procesar_imagen(foto_capturada, modelo, etiquetas, conn, guardar_imagenes, "captura_foto")
        
        with tab3:
            st.subheader("Subir Imagen para Análisis")
            
            if not modelo_cargado:
                st.error("❌ El modelo no está cargado. No se puede realizar reconocimiento.")
            else:
                archivo_subido = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])
                
                if archivo_subido:
                    procesar_imagen(archivo_subido, modelo, etiquetas, conn, guardar_imagenes, "imagen_subida")
    
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
                
                submitted = st.form_submit_button("💾 Registrar Persona", use_container_width=True)
                
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
                        guardar = st.form_submit_button("💾 Guardar Cambios", use_container_width=True)
                    with col_btn2:
                        eliminar = st.form_submit_button("🗑️ Eliminar", type="secondary", use_container_width=True)
                    
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
                    use_container_width=True,
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
        else:
            mostrar_analitica(df_predicciones, conn)
    
    # ==================== SECCIÓN: EXPORTACIÓN ====================
    elif seccion == "📦 Exportación":
        st.header("📦 Exportación de Datos y Gráficas")
        mostrar_exportacion(conn)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Sistema de Reconocimiento de Personas v1.0 | Desarrollado con Streamlit + TensorFlow</p>
        <p>© 2025 - Proyecto de Exoneración</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FUNCIONES AUXILIARES ====================
def procesar_imagen(archivo_imagen, modelo, etiquetas, conn, guardar_imagenes, fuente):
    """Procesa una imagen y muestra resultados"""
    img_pil = Image.open(archivo_imagen)
    etiqueta, confianza = realizar_prediccion(img_pil, modelo, etiquetas)
    umbral = obtener_umbral_persona(etiqueta, conn)
    aprobada = confianza >= umbral
    
    col_img, col_result = st.columns(2)
    with col_img:
        st.image(img_pil, caption="Imagen analizada", use_container_width=True)
    
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
        fuente,
        etiqueta,
        confianza,
        umbral,
        aprobada,
        imagen_ruta,
        conn
    )

def mostrar_analitica(df_predicciones, conn):
    """Muestra la sección de analítica"""
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
    
    # Gráficas (simplificadas para el ejemplo)
    st.subheader("📊 Visualizaciones")
    
    # Gráfica 1: Detecciones por Persona
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    conteo_personas = df_predicciones['etiqueta_detectada'].value_counts().head(10)
    conteo_personas.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Número de Detecciones')
    ax1.set_ylabel('Persona')
    ax1.set_title('Top 10 - Detecciones por Persona')
    ax1.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

def mostrar_exportacion(conn):
    """Muestra la sección de exportación"""
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
        csv_predicciones = df_export_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV de Predicciones",
            data=csv_predicciones,
            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No hay predicciones para exportar")

if __name__ == "__main__":
    main()