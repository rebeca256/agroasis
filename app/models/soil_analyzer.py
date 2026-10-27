# ============================================================================
# Módulo: soil_analyzer.py
# Materia: Estadística
# Descripción: Analiza la salud del suelo clasificando cada nutriente
#   contra rangos de referencia agronómica (INIFAP). Calcula un score
#   de salud general (0-100) mediante promedio ponderado de nutrientes.
# ============================================================================
"""
Analizador de suelo: vectorización y compatibilidad con cultivos.
Calcula el vector 5D del suelo y lo compara con demandas de cultivos.
"""
import numpy as np
from config import SOIL_FEATURES, NUTRIENT_LIMITS
from models.crop_classifier import CropClassifier


class SoilAnalyzer:
    """Analizador de propiedades del suelo y compatibilidad con cultivos"""

    def __init__(self):
        self.clasificador = CropClassifier()

    def analizar_suelo(self, datos_dict):
        """
        Realiza análisis completo del suelo.
        Retorna: perfil nutricional, comparativa con límites, cultivos compatibles.
        """
        # 1. Perfil nutricional básico
        perfil_nutricional = self._obtener_perfil_nutricional(datos_dict)

        # 2. Comparativa con límites de referencia
        comparativa = self._comparar_con_limites(datos_dict)

        # 3. Vectorización normalizada (5D para graficas)
        vector = self._vectorizar_suelo(datos_dict)

        # 4. Predicciones de cultivos compatibles desde el clasificador
        cultivos_compatibles, _ = self.clasificador.predecir_cultivos(datos_dict, n_mejores=10)

        # 5. Requerimientos nutricionales por cultivo (desde el modelo, no hardcodeado)
        analisis_nutrientes = self._analizar_requerimientos_cultivos(
            cultivos_compatibles, datos_dict
        )

        return {
            "perfil_nutricional": perfil_nutricional,
            "comparativa_limites": comparativa,
            "vector_normalizado": vector.tolist() if vector is not None else None,
            "cultivos_compatibles": cultivos_compatibles,
            "requerimientos_nutricionales": analisis_nutrientes,
            "salud_general": self._calcular_salud_suelo(comparativa),
        }

    # -----------------------------------------------------------------------
    # Métodos internos
    # -----------------------------------------------------------------------

    def _obtener_perfil_nutricional(self, datos_dict):
        """Extrae perfil de nutrientes principales del suelo."""
        nutrientes_principales = ["N_mgkg", "P_mgkg", "K_mgkg", "Ca_mgkg", "Mg_mgkg"]
        perfil = {}
        for nutriente in nutrientes_principales:
            if nutriente in datos_dict:
                valor = float(datos_dict[nutriente])
                limites = NUTRIENT_LIMITS.get(nutriente, {})
                perfil[nutriente] = {
                    "valor": valor,
                    "nivel": self._clasificar_nutriente(valor, limites),
                    "limites": limites,
                }
        return perfil

    def _clasificar_nutriente(self, valor, limites):
        """Clasifica el nivel de un nutriente según límites de referencia."""
        bajo  = limites.get("bajo",  valor / 2)
        medio = limites.get("medio", valor * 1.5)
        alto  = limites.get("alto",  valor * 3)
        if valor < bajo:
            return "deficiente"
        elif valor < medio:
            return "bajo"
        elif valor < alto:
            return "medio"
        else:
            return "alto"

    def _comparar_con_limites(self, datos_dict):
        """Compara cada nutriente con límites de referencia usando numpy."""
        comparativa = {}
        for nutriente, limites in NUTRIENT_LIMITS.items():
            if nutriente in datos_dict:
                valor = float(datos_dict[nutriente])
                # Clasificación simple sin scipy
                if valor < limites["bajo"]:
                    clasificacion = "bajo"
                elif valor < limites["medio"]:
                    clasificacion = "medio"
                elif valor < limites["alto"]:
                    clasificacion = "alto"
                else:
                    clasificacion = "muy_alto"

                if clasificacion == "bajo":
                    recomendacion = f"Aumentar a {limites['medio']} mg/kg"
                elif clasificacion == "alto" or clasificacion == "muy_alto":
                    recomendacion = f"Mantener o reducir a {limites['medio']} mg/kg"
                else:
                    recomendacion = "Nivel adecuado, mantener"

                comparativa[nutriente] = {
                    "valor": valor,
                    "clasificacion": clasificacion,
                    "recomendacion": recomendacion,
                }
        return comparativa

    def _vectorizar_suelo(self, datos_dict):
        """Vectoriza los datos del suelo (normalización min-max simple)."""
        try:
            valores = [float(datos_dict[f]) for f in SOIL_FEATURES if f in datos_dict]
            if not valores:
                return None
            arr = np.array(valores)
            rango = arr.max() - arr.min()
            if rango == 0:
                return np.zeros_like(arr)
            return (arr - arr.min()) / rango
        except Exception:
            return None

    def _analizar_requerimientos_cultivos(self, cultivos_compatibles, datos_suelo):
        """
        Analiza requerimientos de nutrientes para los cultivos recomendados.
        Los valores de demanda provienen directamente del dataset (via el recomendador),
        no de valores hardcodeados.
        """
        if not cultivos_compatibles:
            return {}

        from models.fertilizer_recommender_v2 import obtener_recomendador
        recommender = obtener_recomendador()
        requerimientos = {}

        for cultivo_info in cultivos_compatibles[:3]:  # Top 3
            cultivo = cultivo_info["cultivo"]
            req_base = recommender.demandas_cultivo.get(
                cultivo.lower().replace(" ", "_"),
                list(recommender.demandas_cultivo.values())[0] if recommender.demandas_cultivo else {"n": 150, "p": 50, "k": 100},
            )

            analisis = {}
            for nut, key_base in [("N", "n"), ("P", "p"), ("K", "k")]:
                requerido   = float(req_base.get(key_base, 0))
                actual_kgha = float(datos_suelo.get(f"{nut}_mgkg", 0)) * 2.0
                analisis[nut] = {
                    "requerido":    requerido,
                    "disponible":   actual_kgha,
                    "deficit":      max(0.0, requerido - actual_kgha),
                    "es_suficiente": actual_kgha >= requerido,
                }

            requerimientos[cultivo] = {
                "confianza": cultivo_info["porcentaje"],
                "nutrientes": analisis,
            }

        return requerimientos

    def _calcular_salud_suelo(self, comparativa):
        """Calcula un score general de salud del suelo (0-100)."""
        mapa_score = {"deficiente": 10, "bajo": 30, "medio": 75, "alto": 85, "muy_alto": 60}
        scores = [mapa_score.get(info["clasificacion"], 50) for info in comparativa.values()]
        salud_score = float(np.mean(scores)) if scores else 50.0

        if salud_score >= 80:   estado = "excelente"
        elif salud_score >= 65: estado = "bueno"
        elif salud_score >= 50: estado = "aceptable"
        elif salud_score >= 30: estado = "pobre"
        else:                   estado = "muy_pobre"

        return {"score": salud_score, "estado": estado}
