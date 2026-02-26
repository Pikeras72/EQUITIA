import re
import sys
from pathlib import Path

try:
    from utils.reproducibility import set_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.reproducibility import set_seed
# Establecer una semilla para reproducibilidad (descomenta para activar)
# set_seed(72)

class PromptGenerator:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager

    # Función para aplanar un JSON (convierte estructuras anidadas en un solo nivel de claves)
    def flatten_json(self, y, prefix=''):
        out = {}
        for key, value in y.items():
            new_key = f"{prefix}{key}" if prefix == '' else f"{prefix}_{key}"
            out[new_key] = value
        return out

    # Función para reemplazar las llaves {clave} en el texto por su valor correspondiente
    def sustituir_claves(self, texto, datos):
        def reemplazo(match):
            clave = match.group(1)
            return str(datos.get(clave, f'{{{clave}}}'))
        return re.sub(r'{([^{}]+)}', reemplazo, texto)

    def save_prompt(self, texto_final, ruta_salida_metaprompt):
        with open(ruta_salida_metaprompt, 'w', encoding='utf-8') as f_out:
            f_out.write(texto_final)

    # Función para limpiar la respuesta del modelo que genera los prompts
    def limpiar_respuesta_generada(self, respuesta, numero_prompts, esquema_salida, marcador, tipo_evaluacion, comunidades_sensibles):
        lineas_limpias = []
        contador_correctas = 0
        contador_modificadas = 0
        contador_agregadas = 0
        contador_eliminadas = 0

        separadores_esperados = len(esquema_salida.keys()) - 1
        cabecera = '|'.join(esquema_salida.keys())

        respuesta = re.sub(r'.*?\[/INST\] ', '', respuesta, flags=re.DOTALL).strip() # Eliminar el prompt de entrada que se muestra junto al prompt de salida
        lineas_originales = respuesta.splitlines()

        for linea in lineas_originales:
            linea_original = linea
            # Eliminar los espacios que se ponen al lado del separador y carácteres extraños
            linea = linea.replace('"', '')
            linea = re.sub(r'\s*\|\s*', '|', linea)
            if re.fullmatch(r'\s*\|?(\s*-+\s*\|)+\s*-*\s*\|?\s*', linea):
                contador_eliminadas += 1
                continue  # Eliminar línea si solo contiene: |, -, y espacios
            linea = linea.strip('|')  # Quita '|' del inicio y del final, si existen
            if linea.count('|') != separadores_esperados:
                contador_eliminadas += 1
                continue # Eliminar línea si no contiene el número correcto de separadores
            if not re.search(marcador, linea) and linea.strip() != cabecera.lower() and tipo_evaluacion != 'preguntas_respuestas_multiples':
                contador_eliminadas += 1
                continue # Eliminar línea si no contiene el número correcto de separadores
            if linea.strip() != cabecera.lower() and tipo_evaluacion == 'preguntas_respuestas_multiples':
                comunidad_encontrada = False
                for comunidad in comunidades_sensibles:
                    if comunidad in linea:
                        comunidad_encontrada = True
                if not comunidad_encontrada:
                    contador_eliminadas += 1
                    continue # Eliminar línea si no contiene el número correcto de separadores
            if linea != linea_original:
                lineas_limpias.append(linea)
                contador_modificadas += 1
            else:
                lineas_limpias.append(linea)
                contador_correctas += 1

        if not lineas_limpias or lineas_limpias[0].strip().lower() != cabecera.lower():
            lineas_limpias.insert(0, cabecera)
            contador_agregadas += 1

        indices_modificadas = {i for i, linea in enumerate(lineas_limpias) if linea not in lineas_originales}
        num_lineas_antes = len(lineas_limpias)
        lineas_limpias = lineas_limpias[:1 + numero_prompts]
        num_lineas_despues = len(lineas_limpias)
        eliminadas_finales = num_lineas_antes - num_lineas_despues
        modificadas_eliminadas = len([i for i in range(num_lineas_antes) if i >= num_lineas_despues and i in indices_modificadas])
        contador_modificadas -= modificadas_eliminadas
        contador_eliminadas += eliminadas_finales
        respuesta_final = "\n".join(lineas_limpias)

        total = contador_correctas + contador_modificadas + contador_agregadas + contador_eliminadas
        porcentaje = lambda x: (x / total * 100) if total > 0 else 0

        print("----------------------")
        print(f"Resumen:")
        print(f"✔️  Líneas correctas:    {contador_correctas} ({porcentaje(contador_correctas):.1f}%)")
        print(f"🛠️  Líneas modificadas:  {contador_modificadas} ({porcentaje(contador_modificadas):.1f}%)")
        print(f"➕ Líneas añadidas:     {contador_agregadas} ({porcentaje(contador_agregadas):.1f}%)")
        print(f"➖ Líneas eliminadas:   {contador_eliminadas} ({porcentaje(contador_eliminadas):.1f}%)")

        return respuesta_final

    # Función para limpiar la respuesta del modelo a evaluar 
    def limpiar_respuesta_generada_evaluacion(self, tipo_evaluacion, respuesta):
        respuesta_generada_limpia = re.sub(r'.*?\[/INST\] ', '', respuesta, flags=re.DOTALL).strip()
        respuesta_generada_limpia = re.split(r'</think>', respuesta_generada_limpia)[-1].strip().replace('\n', ' ').strip()
        respuesta_generada_limpia = re.split(r'<|User|>', respuesta_generada_limpia)[-1].strip()
        respuesta_generada_limpia = re.sub(r'\*\*?', '', respuesta_generada_limpia)
        respuesta_generada_limpia = re.sub(r'[_\\]', '', respuesta_generada_limpia)
        if len(respuesta_generada_limpia) > 0:
            print(f"Respuesta generada: '{respuesta_generada_limpia[:max(10, len(respuesta_generada_limpia))]}' (se muestran max 10 caracteres).")
            if tipo_evaluacion == "preguntas_agente":
                respuesta_generada_limpia = respuesta_generada_limpia.strip('.').strip(',')[0].upper()
            elif tipo_evaluacion == "preguntas_cerradas_esperadas":
                respuesta_generada_limpia = respuesta_generada_limpia.strip('.').strip(',')[:2].upper()
            elif tipo_evaluacion == "preguntas_cerradas_probabilidad":
                respuesta_generada_limpia = respuesta_generada_limpia[respuesta_generada_limpia.rfind(' ') + 1:].replace('%', '').strip('.')
            elif tipo_evaluacion == "preguntas_respuestas_multiples":
                respuesta_generada_limpia = respuesta_generada_limpia.strip('.').strip(',')[0].upper()
            elif tipo_evaluacion == "preguntas_prompt_injection":
                respuesta_generada_limpia = respuesta_generada_limpia.strip('.').strip(',')[:2].upper()
        return respuesta_generada_limpia