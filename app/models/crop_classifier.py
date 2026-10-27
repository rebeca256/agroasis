# ============================================================================
# Módulo: crop_classifier.py
# Materia: Machine Learning / Estadística
# Descripción: Clasificador de cultivos compatibles basado en Random Forest.
#   Utiliza un bosque de 200 árboles de decisión entrenados con datos
#   agronómicos de INIFAP para predecir qué cultivos son compatibles
#   con las propiedades fisicoquímicas del suelo del agricultor.
# ============================================================================
"""
Clasificador de cultivos basado en propiedades del suelo
Entrena y predice cultivos recomendados
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import (
    CROP_CLASSIFIER_PATH, SCALER_PATH, ENCODER_TEXTURE_PATH,
    SOIL_FEATURES, MODEL_PARAMS
)
from utils.data_loader import DataLoader

class CropClassifier:
    """Clasificador de cultivos recomendados"""
    
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.encoder_textura = None
        self.features_usadas = None
        self.cultivos_entrenados = None
        self.modelos_guardados = False
        self.intentar_cargar_modelos()
    
    def intentar_cargar_modelos(self):
        """Intenta cargar modelos previamente entrenados"""
        try:
            with open(CROP_CLASSIFIER_PATH, 'rb') as f:
                self.modelo = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(ENCODER_TEXTURE_PATH, 'rb') as f:
                self.encoder_textura = pickle.load(f)
            
            self.modelos_guardados = True
            print("[OK] Modelos entrenados cargados correctamente")
        except:
            self.modelos_guardados = False
    
    def entrenar(self, df):
        """
        Entrena el clasificador con los datos del CSV
        """

        
        # Preparar datos
        self.encoder_textura = LabelEncoder()
        df_train = df.copy()
        
        # Codificar textura
        df_train["textura_cod"] = self.encoder_textura.fit_transform(df_train["textura_suelo"])
        
        # Features
        self.features_usadas = SOIL_FEATURES + ["textura_cod"]
        X = df_train[self.features_usadas].values
        y = df_train["cultivo"]
        self.cultivos_entrenados = y.unique().tolist()
        
        # Escalar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=MODEL_PARAMS["test_size"],
            random_state=MODEL_PARAMS["random_state"]
        )
        
        # Entrenar
        self.modelo = RandomForestClassifier(
            n_estimators=MODEL_PARAMS["rf_estimators"],
            random_state=MODEL_PARAMS["random_state"],
            n_jobs=-1,
            verbose=0
        )
        self.modelo.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Guardar modelos
        self.guardar_modelos()
        
        return {
            "accuracy": float(accuracy),
            "n_samples": len(df_train),
            "n_cultivos": len(self.cultivos_entrenados),
            "n_features": len(self.features_usadas)
        }
    
    def guardar_modelos(self):
        """Guarda los modelos entrenados"""
        try:
            with open(CROP_CLASSIFIER_PATH, 'wb') as f:
                pickle.dump(self.modelo, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(ENCODER_TEXTURE_PATH, 'wb') as f:
                pickle.dump(self.encoder_textura, f)
            
            self.modelos_guardados = True
            print("[OK] Modelos guardados exitosamente")
        except Exception as e:
            print(f"[ERROR] Error al guardar modelos: {e}")
    
    def preparar_entrada(self, datos_dict):
        """
        Prepara los datos del usuario para predicción
        datos_dict debe tener todas las features + textura_suelo
        """
        try:
            # Codificar textura
            textura_cod = self.encoder_textura.transform([datos_dict["textura_suelo"]])[0]
            
            # Construir vector X y convertir N, P, K de mg/kg a kg/ha (*2.0)
            X = []
            for feature in SOIL_FEATURES:
                val = float(datos_dict.get(feature, 0))
                # Los parámetros N_mgkg, P_mgkg, K_mgkg se multiplican por 2 para kg/ha
                if feature in ["N_mgkg", "P_mgkg", "K_mgkg"]:
                    val = val * 2.0
                X.append(val)
            X.append(textura_cod)
            
            X = np.array(X).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            return X_scaled, True
        except Exception as e:
            return None, False
    
    def predecir_cultivos(self, datos_dict, n_mejores=5):
        """
        Predice los cultivos recomendados
        Retorna top N cultivos con probabilidades
        """
        if not self.modelos_guardados and self.modelo is None:
            return None, "Modelo no entrenado"
        
        X_scaled, es_valido = self.preparar_entrada(datos_dict)
        if not es_valido:
            return None, "Datos inválidos"
        
        # Obtener probabilidades
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        cultivos = self.modelo.classes_
        
        # Top N
        indices_top = np.argsort(probabilidades)[-n_mejores:][::-1]
        
        resultados = []
        for idx in indices_top:
            resultados.append({
                "cultivo": cultivos[idx],
                "confianza": float(probabilidades[idx]),
                "porcentaje": float(probabilidades[idx] * 100)
            })
        
        return resultados, None
    
    def obtener_importancia_features(self):
        """Retorna la importancia de cada feature en el modelo"""
        if self.modelo is None:
            return None
        
        importancias = self.modelo.feature_importances_
        features = self.features_usadas
        
        resultados = []
        for feature, importancia in sorted(zip(features, importancias), key=lambda x: x[1], reverse=True):
            resultados.append({
                "feature": feature,
                "importancia": float(importancia),
                "porcentaje": float(importancia * 100)
            })
        
        return resultados


def crear_y_entrenar_clasificador():
    """Función de conveniencia para crear y entrenar el clasificador"""
    loader = DataLoader()
    if not loader.cargar_csv_cultivos():
        return None
    
    df_limpio = loader.limpiar_datos(loader.df)
    
    clasificador = CropClassifier()
    clasificador.entrenar(df_limpio)
    
    return clasificador
