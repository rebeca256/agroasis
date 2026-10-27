# ============================================================================
# Módulo: data_loader.py
# Descripción: Carga y preprocesamiento del dataset principal de cultivos
#   mexicanos (cultivos_mexico_v3_final.csv). Limpia datos, normaliza
#   texturas y valida rangos antes de alimentar al clasificador.
# ============================================================================
"""
Carga y preparación de datos del CSV de cultivos
Responsable de procesar y validar los datos de entrada
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import CSV_CULTIVOS, SOIL_FEATURES, SOIL_TEXTURES

class DataLoader:
    """Cargador de datos y preparador de características"""
    
    def __init__(self):
        self.df = None
        self.original_df = None
        
    def cargar_csv_cultivos(self):
        """Carga el CSV principal de cultivos"""
        try:
            self.df = pd.read_csv(CSV_CULTIVOS)
            self.original_df = self.df.copy()
            print(f"CSV de cultivos cargado: {len(self.df)} registros")
            return True
        except Exception as e:
            print(f"Error al cargar CSV: {e}")
            return False
    
    def cargar_csv_usuario(self, filepath):
        """Carga un CSV subido por el usuario"""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error al cargar CSV de usuario: {e}")
            return None
    
    def validar_columnas_usuario(self, df):
        """Valida que el CSV del usuario tenga las columnas necesarias"""
        columnas_requeridas = SOIL_FEATURES + ["textura_suelo"]
        columnas_presentes = set(df.columns)
        columnas_faltantes = set(columnas_requeridas) - columnas_presentes
        
        if columnas_faltantes:
            return False, list(columnas_faltantes)
        return True, []
    
    def limpiar_datos(self, df):
        """Limpia y prepara los datos"""
        df_clean = df.copy()
        
        # Eliminar filas con valores nulos en características críticas
        df_clean = df_clean.dropna(subset=SOIL_FEATURES + ["textura_suelo"])
        
        # Validar texturas
        df_clean = df_clean[df_clean["textura_suelo"].isin(SOIL_TEXTURES)]
        
        # Convertir a float
        for col in SOIL_FEATURES:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        
        # Eliminar NaN resultantes de conversión
        df_clean = df_clean.dropna(subset=SOIL_FEATURES)
        
        return df_clean
    
    def obtener_estadisticas(self, df):
        """Obtiene estadísticas descriptivas de un dataframe"""
        stats = {}
        for col in SOIL_FEATURES:
            if col in df.columns:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
        return stats
    
    def preparar_features_x(self, df):
        """Prepara la matriz X de features para predicción"""
        features_presentes = [f for f in SOIL_FEATURES if f in df.columns]
        X = df[features_presentes].values
        return X, features_presentes
    
    def obtener_cultivos_unicos(self):
        """Obtiene lista de cultivos únicos en el dataset"""
        if self.df is not None and "cultivo" in self.df.columns:
            return sorted(self.df["cultivo"].unique().tolist())
        return []


def cargar_y_preparar_datos_principales():
    """Función de conveniencia para cargar y preparar datos principales"""
    loader = DataLoader()
    if loader.cargar_csv_cultivos():
        return loader
    return None
