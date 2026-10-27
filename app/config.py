# ============================================================================
# Módulo: config.py
# Descripción: Configuración centralizada de la aplicación Agroasis.
#   Contiene rutas de archivos, listas de features del suelo,
#   texturas validas y parámetros de Flask.
# ============================================================================
"""
Configuración centralizada para la aplicación Agroasis
"""
from pathlib import Path
import os

# Directorios base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "trained_models"
UPLOADS_DIR = BASE_DIR / "uploads"

# Archivos de datos
CSV_CULTIVOS = BASE_DIR / "cultivos_mexico_v3_final.csv"

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Rutas de modelos entrenados
CROP_CLASSIFIER_PATH = MODELS_DIR / "crop_classifier.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_TEXTURE_PATH = MODELS_DIR / "encoder_texture.pkl"
YIELD_REGRESSOR_PATH = MODELS_DIR / "yield_regressor.pkl"
CROP_NUTRIENT_PROFILE_PATH = MODELS_DIR / "crop_nutrient_profile.pkl"

# Configuración de aplicación Flask
FLASK_CONFIG = {
    "DEBUG": True,
    "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,  # 16MB máximo para upload
    "UPLOAD_FOLDER": str(UPLOADS_DIR),
    "JSON_SORT_KEYS": False
}

# Parámetros del modelo
MODEL_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "rf_estimators": 200,
    "gb_estimators": 200
}

# Variables de análisis quémico del suelo
SOIL_FEATURES = [
    "ph", "mo_pct", "N_mgkg", "P_mgkg", "K_mgkg",
    "Ca_mgkg", "Mg_mgkg", "S_mgkg",
    "Fe_mgkg", "Zn_mgkg", "Mn_mgkg", "Cu_mgkg", "B_mgkg",
    "ce_ds_m", "precipitacion_mm", "temp_media_c", "altitud_msnm"
]

# Límites de nutrientes por cultivo (referencia agrícola)
NUTRIENT_LIMITS = {
    "N_mgkg": {"bajo": 20, "medio": 50, "alto": 100},
    "P_mgkg": {"bajo": 10, "medio": 25, "alto": 50},
    "K_mgkg": {"bajo": 100, "medio": 250, "alto": 500},
    "Ca_mgkg": {"bajo": 1000, "medio": 2500, "alto": 5000},
    "Mg_mgkg": {"bajo": 100, "medio": 250, "alto": 500},
    "S_mgkg": {"bajo": 10, "medio": 25, "alto": 50},
}

# Ejemplos de fertilizantes mexicanos
FERTILIZERS_MEXICO = {
    "Nitrato de Amonio (N 33.5%)": {
        "nombre_comercial": "Nitrato de Amonio",
        "N": 33.5, "P": 0, "K": 0,
        "precio_kg": 12.5,
        "proveedor": "PEMEX Fertilizantes"
    },
    "Superfosfato Triple (P 46%)": {
        "nombre_comercial": "Superfosfato Triple",
        "N": 0, "P": 46, "K": 0,
        "precio_kg": 15.0,
        "proveedor": "Profert"
    },
    "Cloruro de Potasio (K 60%)": {
        "nombre_comercial": "Cloruro de Potasio",
        "N": 0, "P": 0, "K": 60,
        "precio_kg": 18.0,
        "proveedor": "Mosaic"
    },
    "Urea (N 46%)": {
        "nombre_comercial": "Urea",
        "N": 46, "P": 0, "K": 0,
        "precio_kg": 11.5,
        "proveedor": "PEMEX Fertilizantes"
    },
    "Fosfato Diamónico DAP (N 18%, P 46%)": {
        "nombre_comercial": "DAP",
        "N": 18, "P": 46, "K": 0,
        "precio_kg": 14.5,
        "proveedor": "Profert"
    }
}

# Estados de salud del suelo
HEALTH_STATES = {
    "excelente": {"color": "#27ae60", "score": (65, 100)},  # Verde
    "bueno":     {"color": "#9acd32", "score": (50, 65)},   # Amarillo verdoso (yellowgreen)
    "aceptable": {"color": "#f39c12", "score": (40, 50)},   # Amarillo naranja
    "pobre":     {"color": "#e67e22", "score": (30, 40)},   # Naranja
    "muy_pobre": {"color": "#e74c3c", "score": (0, 30)}     # Rojo
}

# Texturas de suelo disponibles
SOIL_TEXTURES = [
    "arenoso",
    "franco_arenoso",
    "franco",
    "franco_limoso",
    "limoso",
    "franco_arcilloso",
    "arcillo_arenoso",
    "arcillo_limoso",
    "arcilloso"
]
