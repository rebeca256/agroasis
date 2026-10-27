# ============================================================================
# Módulo: fertilizer_recommender_v2.py
# Materia: Álgebra Lineal
# Descripción: Motor de vectorización 5D y recomendación de fertilizantes.
#   Representa suelo y cultivo como vectores en R^5 = [N, P, K, pH, MO],
#   calcula el déficit mediante resta de vectores (DeltaV = V_demanda - V_suelo),
#   y selecciona fertilizantes óptimos con un algoritmo greedy.
#   Conversión de unidades: mg/kg -> kg/ha (factor 2.0)
# ============================================================================
"""
Recomendador de Fertilizantes basado en Álgebra Lineal Extendida.

Vector de análisis de 5 dimensiones:
    V[0] = Nitrógeno   (N)   en kg/ha
    V[1] = Fósforo     (P)   en kg/ha
    V[2] = Potasio     (K)   en kg/ha
    V[3] = pH          (adimensional, convertido a escala de corrección)
    V[4] = Materia Orgánica (MO%) en porcentaje

La ecuación central es: delta_V = V_demanda - V_suelo
Los valores positivos indican déficit (se debe aplicar),
los negativos indican exceso (no se debe aplicar).
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constantes de clasificación de estado nutricional
# ---------------------------------------------------------------------------

# Umbrales de balance (delta_V) para cada parámetro que determinan el estado.
# El orden importa: se evaluan de mayor a menor severidad.
UMBRALES_ESTADO = {
    # Nitrógeno, Fósforo, Potasio: en kg/ha
    "N": {
        "exceso_toxico": 200,   # exceso mayor a 200 kg/ha es tóxico
        "exceso":        30,    # exceso entre 30-200 kg/ha
        "optimo_max":    10,    # margen de óptimo: -10 a +10 kg/ha
        "optimo_min":   -10,
        "critico":      -50,    # déficit mayor a 50 kg/ha es crítico
    },
    "P": {
        "exceso_toxico": 100,
        "exceso":        20,
        "optimo_max":    10,
        "optimo_min":   -10,
        "critico":      -40,
    },
    "K": {
        "exceso_toxico": 300,
        "exceso":        50,
        "optimo_max":    20,
        "optimo_min":   -20,
        "critico":      -80,
    },
    # pH: el balance es pH_optimo - pH_actual
    "pH": {
        "exceso_toxico": 1.5,   # suelo demasiado alcalino
        "exceso":        0.5,
        "optimo_max":    0.3,
        "optimo_min":   -0.3,
        "critico":      -1.0,   # suelo muy ácido
    },
    # Materia orgánica: en porcentaje
    "MO": {
        "exceso_toxico": 3.0,
        "exceso":        1.0,
        "optimo_max":    0.5,
        "optimo_min":   -0.5,
        "critico":      -1.5,
    },
}


def clasificar_estado(parametro: str, balance: float) -> str:
    """
    Determina el estado nutricional de un parámetro dado su balance.

    El balance se define como: demanda_cultivo - aporte_suelo.
    Valores negativos indican exceso del suelo sobre la demanda.
    Valores positivos indican déficit.

    Complejidad: O(1) — evaluación por umbrales fijos.
    """
    u = UMBRALES_ESTADO.get(parametro, UMBRALES_ESTADO["N"])
    balance_neg = -balance  # Invertir para interpretar exceso correctamente

    if balance_neg >= u["exceso_toxico"]:
        return "Exceso Toxico"
    if balance_neg >= u["exceso"]:
        return "Exceso"
    if u["optimo_min"] <= balance <= u["optimo_max"]:
        return "Optimo"
    if balance <= u["critico"]:
        return "Critico"
    return "Deficit"


def formatear_dosis_kg_ha(kg_ha: float) -> str:
    """
    Convierte una dosis en kg/ha al formato estandar de presentación.

    Fórmula: costales = kg_ha / 50
    Salida: "150 kg/ha (Equivalente a 3.0 costales de 50 kg)"

    Complejidad: O(1)
    """
    costales = kg_ha / 50.0
    return f"{kg_ha:.1f} kg/ha (Equivalente a {costales:.1f} costales de 50 kg)"


# ---------------------------------------------------------------------------
# Clase principal del recomendador
# ---------------------------------------------------------------------------

class FertilizerRecommender:
    """
    Sistema de recomendación vectorial de fertilizantes.

    Utiliza un vector de 5 dimensiones para el análisis de déficit:
        V = [N (kg/ha), P (kg/ha), K (kg/ha), pH, MO (%)]

    La selección de fertilizantes aplica programación lineal simple
    (greedy por nutriente con mayor déficit) priorizando menor costo.
    """

    def __init__(self, csv_path=None):
        """Carga el catálogo de fertilizantes y define perfiles de demanda por cultivo."""
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "fertilizantes.csv"

        self.df_fertilizantes = pd.read_csv(csv_path)
        # Precio por kg calculado una sola vez al cargar
        self.df_fertilizantes["precio_por_kg"] = (
            self.df_fertilizantes["precio_mxn"] / self.df_fertilizantes["peso_costal_kg"]
        ).round(2)

        # Perfil de demanda por cultivo: [N, P, K, pH_optimo, MO_optimo]
        # Obtenido dinámicamente del dataset de entrenamiento (promedios por cultivo)
        ruta_cultivos = Path(__file__).parent.parent / "cultivos_mexico_v3_final.csv"
        self.demandas_cultivo = {}
        if ruta_cultivos.exists():
            df_c = pd.read_csv(ruta_cultivos)
            grupos = df_c.groupby("cultivo")
            for cultivo, df_g in grupos:
                # N, P, K en mg/kg se multiplican por 2.0 para llevarlos a kg/ha
                self.demandas_cultivo[cultivo] = {
                    "n": float(df_g["N_mgkg"].mean() * 2.0),
                    "p": float(df_g["P_mgkg"].mean() * 2.0),
                    "k": float(df_g["K_mgkg"].mean() * 2.0),
                    "ph": float(df_g["ph"].mean()),
                    "mo": float(df_g["mo_pct"].mean())
                }
        else:
            # Fallback en caso de no encontrar el csv
            self.demandas_cultivo = {
                "maiz_grano_blanco": {"n": 180, "p": 80,  "k": 120, "ph": 6.5, "mo": 3.0}
            }

    # ---------------------------------------------------------------------------
    # Método principal de cálculo de déficit (5D)
    # ---------------------------------------------------------------------------

    def calcular_deficit(self, cultivo: str, propiedades_suelo: dict) -> dict:
        """
        Calcula el vector déficit de 5 dimensiones: delta_V = V_demanda - V_suelo.

        La conversión de mg/kg a kg/ha usa el factor estandar 2.0
        (densidad aparente típica de suelo agrícola: 1 g/cm3 en 20 cm profundidad).

        Retorna el déficit real (con signo) y el aplicable (solo valores positivos).
        Complejidad: O(n_fertilizantes) para la selección.
        """
        demanda = self.demandas_cultivo.get(
            cultivo.lower().replace(" ", "_")
        )
        if demanda is None:
            demanda = self.demandas_cultivo.get(
                "maiz_grano_blanco", 
                list(self.demandas_cultivo.values())[0] if self.demandas_cultivo else {"n": 180, "p": 80, "k": 120, "ph": 6.5, "mo": 3.0}
            )

        # Vector demanda del cultivo [N, P, K, pH_optimo, MO_optimo]
        v_c = np.array([
            demanda["n"],
            demanda["p"],
            demanda["k"],
            demanda["ph"],
            demanda["mo"]
        ], dtype=float)

        # Vector aporte del suelo — conversión mg/kg -> kg/ha para macronutrientes
        n_suelo  = float(propiedades_suelo.get("N_mgkg",  0)) * 2.0
        p_suelo  = float(propiedades_suelo.get("P_mgkg",  0)) * 2.0
        k_suelo  = float(propiedades_suelo.get("K_mgkg",  0)) * 2.0
        ph_suelo = float(propiedades_suelo.get("ph",      7.0))
        mo_suelo = float(propiedades_suelo.get("mo_pct",  2.0))

        v_s = np.array([n_suelo, p_suelo, k_suelo, ph_suelo, mo_suelo], dtype=float)

        # Delta: positivo = déficit, negativo = exceso
        delta_v = v_c - v_s

        # Solo los primeros 3 elementos (N, P, K) son aplicables como fertilizante
        # pH y MO no se corrigen directamente con fertilizantes NPK
        delta_aplicable = np.array([
            max(0.0, delta_v[0]),
            max(0.0, delta_v[1]),
            max(0.0, delta_v[2]),
            0.0,  # pH: requiere encalado/acidificacion, no NPK
            0.0   # MO: requiere enmiendas orgánicas
        ], dtype=float)

        # Clasificación de estado para cada parámetro
        params = ["N", "P", "K", "pH", "MO"]
        balances_redondeados = delta_v.tolist()

        # Tabla de análisis para presentación en UI
        tabla_analisis = []
        unidades = {
            "N":  "kg/ha",
            "P":  "kg/ha",
            "K":  "kg/ha",
            "pH": "pH",
            "MO": "%"
        }
        for i, param in enumerate(params):
            balance = delta_v[i]
            estado  = clasificar_estado(param, balance)
            tabla_analisis.append({
                "parametro":      param,
                "demanda_cultivo": round(float(v_c[i]), 2),
                "aporte_suelo":   round(float(v_s[i]), 2),
                "balance":        round(float(balance),  2),
                "estado":         estado,
                "unidad":         unidades[param]
            })

        return {
            "v_demanda":       v_c.tolist(),
            "v_suelo":         v_s.tolist(),
            "deficit_real":    delta_v.tolist(),
            "deficit_aplicable": delta_aplicable.tolist(),
            "cultivo":         cultivo,
            "tabla_analisis":  tabla_analisis,
            # Excesos individuales para alertas en UI
            "exceso": {
                "n":  max(0.0, v_s[0] - v_c[0]),
                "p":  max(0.0, v_s[1] - v_c[1]),
                "k":  max(0.0, v_s[2] - v_c[2]),
                "ph": max(0.0, v_s[3] - v_c[3]),
                "mo": max(0.0, v_s[4] - v_c[4]),
            }
        }

    # ---------------------------------------------------------------------------
    # Método principal de recomendación
    # ---------------------------------------------------------------------------

    def recomendar_fertilizantes(self, cultivo: str, propiedades_suelo: dict, hectareas: float = 1.0) -> dict:
        """
        Orquesta el cálculo de déficit y la selección de fertilizantes óptimos.

        Retorna el plan completo con análisis de déficit (5D), fertilizantes
        recomendados con formato kg/ha + costales, y costo total.
        """
        analisis_deficit = self.calcular_deficit(cultivo, propiedades_suelo)
        delta_v_aplicable = np.array(analisis_deficit["deficit_aplicable"])

        # Construir lista de fertilizantes con sus vectores NPK
        catalogo = self._construir_catalogo()

        # Selección greedy por nutriente de mayor déficit
        recomendaciones = self._seleccionar_fertilizantes_optimos(
            delta_v_aplicable[:3],  # Solo N, P, K son corregibles con fertilizantes
            catalogo,
            hectareas
        )

        return {
            "analisis_deficit": analisis_deficit,
            "recomendaciones":  recomendaciones,
            "cultivo":          cultivo,
            "hectareas":        hectareas
        }

    # ---------------------------------------------------------------------------
    # Métodos internos (privados por convencion _)
    # ---------------------------------------------------------------------------

    def _construir_catalogo(self) -> list:
        """
        Convierte el DataFrame de fertilizantes en una lista de dicts
        optimizada para iteración repetida.

        Complejidad: O(f) donde f = número de fertilizantes en el CSV.
        """
        return [
            {
                "id":          row["id_fertilizante"],
                "nombre":      row["nombre_comercial"],
                "vector_npk":  np.array([row["n"], row["p"], row["k"]], dtype=float),
                "precio_kg":   row["precio_por_kg"],
                "precio_costal": row["precio_mxn"],
                "peso_costal": row["peso_costal_kg"],
                "clasificacion": row["clasificacion_objetivo"],
                "funcion":     row["funcion_algoritmica"]
            }
            for _, row in self.df_fertilizantes.iterrows()
        ]

    def _seleccionar_fertilizantes_optimos(self, deficit_npk: np.ndarray, catalogo: list, hectareas: float) -> dict:
        """
        Algoritmo greedy de selección de fertilizantes:
        Para cada nutriente con déficit > umbral mínimo, busca el fertilizante
        con mejor relación (menor costo / mayor concentración del nutriente objetivo)
        que no introduzca un nutriente en exceso.

        Complejidad: O(n * f) donde n = nutrientes deficitarios, f = fertilizantes.
        """
        nutrientes      = ["n", "p", "k"]
        umbral_minimo   = 10.0  # Déficit menor a 10 kg/ha no justifica aplicación
        recomendaciones = []

        # Identificar cuales nutrientes requieren corrección
        nutrientes_a_corregir = [
            (i, nutriente, float(valor))
            for i, (nutriente, valor) in enumerate(zip(nutrientes, deficit_npk))
            if valor > umbral_minimo
        ]

        if not nutrientes_a_corregir:
            return {"mensaje": "Suelo equilibrado, sin necesidad de fertilización adicional.", "fertilizantes": [], "costo_total": 0.0}

        for idx_nut, nutriente, valor_deficit in nutrientes_a_corregir:
            mejor_opcion = None
            mejor_score  = float("inf")

            for fert in catalogo:
                concentracion = fert["vector_npk"][idx_nut]
                if concentracion <= 0:
                    continue  # Este fertilizante no aporta el nutriente objetivo

                dosis_necesaria = valor_deficit / concentracion
                costo_total_fert = dosis_necesaria * fert["precio_kg"] * hectareas

                # Penalizar si el fertilizante introduce nutrientes que ya estan en exceso
                tiene_conflicto = any(
                    nut_idx != idx_nut
                    and fert["vector_npk"][nut_idx] > 0
                    and deficit_npk[nut_idx] <= 0
                    for nut_idx in range(len(nutrientes))
                )
                if tiene_conflicto:
                    continue

                # Score: menor es mejor (costo ponderado por eficiencia del nutriente)
                score = costo_total_fert / (concentracion + 1)
                if score < mejor_score:
                    mejor_score  = score
                    mejor_opcion = {
                        "fertilizante":       fert,
                        "dosis_kg_ha":        dosis_necesaria,
                        "costo_total":        costo_total_fert,
                        "nutriente_objetivo": nutriente,
                    }

            if mejor_opcion:
                recomendaciones.append(mejor_opcion)

        # Formatear respuesta con unidades y costales
        return {
            "fertilizantes": [
                {
                    "id":                   r["fertilizante"]["id"],
                    "nombre_comercial":     r["fertilizante"]["nombre"],
                    "composicion": {
                        "n": r["fertilizante"]["vector_npk"][0],
                        "p": r["fertilizante"]["vector_npk"][1],
                        "k": r["fertilizante"]["vector_npk"][2],
                    },
                    "nutriente_objetivo":   r["nutriente_objetivo"],
                    "dosis_kg_ha":          round(r["dosis_kg_ha"], 2),
                    "dosis_formateada":     formatear_dosis_kg_ha(r["dosis_kg_ha"]),
                    "dosis_total_kg":       round(r["dosis_kg_ha"] * 1, 2),
                    "precio_kg":            r["fertilizante"]["precio_kg"],
                    "precio_total":         round(r["costo_total"], 2),
                    "numero_costales_50kg": round(r["dosis_kg_ha"] / 50, 2),
                    "costo_costales":       round((r["dosis_kg_ha"] / 50) * r["fertilizante"]["precio_costal"], 2),
                    "clasificacion":        r["fertilizante"]["clasificacion"],
                    "funcion":              r["fertilizante"]["funcion"],
                }
                for r in recomendaciones
            ],
            "costo_total": sum(r["costo_total"] for r in recomendaciones)
        }

    def obtener_fertilizante_por_id(self, fert_id: int) -> dict | None:
        """Búsqueda por ID en el catálogo. Retorna None si no existe."""
        row = self.df_fertilizantes[self.df_fertilizantes["id_fertilizante"] == fert_id]
        if row.empty:
            return None
        row = row.iloc[0]
        return {
            "id":          row["id_fertilizante"],
            "nombre":      row["nombre_comercial"],
            "composicion": {"n": row["n"], "p": row["p"], "k": row["k"]},
            "precio_kg":   row["precio_mxn"] / row["peso_costal_kg"],
            "precio_costal": row["precio_mxn"],
            "peso_costal": row["peso_costal_kg"]
        }


# ---------------------------------------------------------------------------
# Instancia singleton para evitar recargar el CSV en cada request
# ---------------------------------------------------------------------------

_recomendador: FertilizerRecommender | None = None


def obtener_recomendador() -> FertilizerRecommender:
    """Devuelve la instancia global del recomendador (patrón singleton)."""
    global _recomendador
    if _recomendador is None:
        _recomendador = FertilizerRecommender()
    return _recomendador
