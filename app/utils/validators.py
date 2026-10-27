# ============================================================================
# Módulo: validators.py
# Descripción: Validación y sanitización de datos de entrada del usuario.
#   Verifica rangos físicos de cada nutriente, limpia textos y asegura
#   que los datos sean seguros antes de procesarlos.
# ============================================================================
"""
Validadores y sanitizadores para datos de entrada
"""
from config import SOIL_FEATURES, SOIL_TEXTURES

class DataValidator:
    """Validador de datos de entrada"""
    
    @staticmethod
    def validar_valor_numerico(valor, nombre, rango=None):
        """Valida que un valor sea numérico y dentro del rango"""
        try:
            val_float = float(valor)
            if rango and (val_float < rango[0] or val_float > rango[1]):
                return False, f"{nombre} debe estar entre {rango[0]} y {rango[1]}"
            return True, val_float
        except:
            return False, f"{nombre} debe ser un número válido"
    
    @staticmethod
    def validar_ph(valor):
        """Valida pH entre 3 y 10"""
        return DataValidator.validar_valor_numerico(valor, "pH", (3, 10))
    
    @staticmethod
    def validar_porcentaje(valor):
        """Valida que sea porcentaje (0-100)"""
        return DataValidator.validar_valor_numerico(valor, "Porcentaje", (0, 100))
    
    @staticmethod
    def validar_nutriente(valor, nombre="Nutriente"):
        """Valida valores de nutrientes (mg/kg) - rango ampliado para Ca"""
        # Ca puede alcanzar 3000-4000 mg/kg en suelos calcáreos
        rango_max = 5000 if "Ca" in nombre or "ca_" in nombre.lower() else 2000
        return DataValidator.validar_valor_numerico(valor, nombre, (0, rango_max))
    
    @staticmethod
    def validar_textura(textura):
        """Valida que sea una textura válida"""
        """Valida que sea una textura válida"""
        if textura not in SOIL_TEXTURES:
            return False, f"Textura inválida. Opciones: {', '.join(SOIL_TEXTURES)}"
        return True, textura
    
    @staticmethod
    def validar_temperatura(valor):
        """Valida temperatura (-20 a 50C)"""
        return DataValidator.validar_valor_numerico(valor, "Temperatura", (-20, 50))
    
    @staticmethod
    def validar_precipitacion(valor):
        """Valida precipitación (0-5000 mm)"""
        return DataValidator.validar_valor_numerico(valor, "Precipitación", (0, 5000))
    
    @staticmethod
    def validar_altitud(valor):
        """Valida altitud (0-5000 msnm)"""
        return DataValidator.validar_valor_numerico(valor, "Altitud", (0, 5000))
    
    @staticmethod
    def sanitizar_diccionario_suelo(datos_dict):
        """
        Sanitiza un diccionario de datos de suelo
        Retorna: (es_valido, datos_limpios, errores)
        """
        errores = []
        datos_limpios = {}
        
        validadores_especiales = {
            "ph": DataValidator.validar_ph,
            "mo_pct": DataValidator.validar_porcentaje,
            "temp_media_c": DataValidator.validar_temperatura,
            "precipitacion_mm": DataValidator.validar_precipitacion,
            "altitud_msnm": DataValidator.validar_altitud,
            "textura_suelo": DataValidator.validar_textura
        }
        
        # Los campos requeridos son SOIL_FEATURES + textura_suelo
        campos_requeridos = SOIL_FEATURES + ["textura_suelo"]
        
        for feature in campos_requeridos:
            if feature not in datos_dict:
                errores.append(f"Falta el campo: {feature}")
                continue
            
            valor = datos_dict[feature]
            
            # Usar validador específico si existe
            if feature in validadores_especiales:
                es_valido, resultado = validadores_especiales[feature](valor)
                if es_valido:
                    datos_limpios[feature] = resultado
                else:
                    errores.append(resultado)
            else:
                # Validar como nutriente genérico (rango 0-1000 por defecto)
                es_valido, resultado = DataValidator.validar_nutriente(valor, feature)
                if es_valido:
                    datos_limpios[feature] = resultado
                else:
                    errores.append(resultado)
        
        return len(errores) == 0, datos_limpios, errores


class InputSanitizer:
    """Sanitizador de entrada para formularios"""
    
    @staticmethod
    def limpiar_string(valor, max_length=100):
        """Limpia strings peligrosos"""
        if isinstance(valor, str):
            return valor.strip()[:max_length]
        return str(valor)[:max_length]
    
    @staticmethod
    def limpiar_archivo_nombre(filename):
        """Limpia nombre de archivo"""
        import os
        filename = os.path.basename(filename)
        # Permitir solo caracteres seguros
        import re
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        return filename
