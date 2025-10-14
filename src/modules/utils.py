import torch
import os
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from config.seed_utils import set_seed
# Establecer una semilla para reproducibilidad (descomenta para activar)
# set_seed(72)

# Función para formatear el tiempo en pantalla
def formatear_tiempo(segundos):
    minutos, segundos = divmod(int(segundos), 60)
    horas, minutos = divmod(minutos, 60)
    return f"{horas}h {minutos}m {segundos}s"

# Función para invocar modelo en local
# Usa ThreadPoolExecutor para aplicar un timeout por llamada y liberar memoria CUDA al finalizar.
def invocar_modelo(prompt, modelo, tokenizer, max_tokens=5000, contexto="", timeout=180):
    print("------------------------------------------------------------------")
    print(f"⏳ Iniciando generación de respuesta con timeout de {timeout}s...", flush=True)
    def generar_respuesta_thread(prompt, contexto, modelo, tokenizer, max_tokens, result_holder):
        try:
            mensajes = [
                {"role": "system", "content": contexto},
                {"role": "user", "content": prompt}
            ]
            mensajes_tokenizados = tokenizer.apply_chat_template(mensajes, return_tensors="pt").to("cuda")
            with torch.no_grad():
                respuesta_generada = modelo.generate(mensajes_tokenizados, max_new_tokens=max_tokens, do_sample=True)
            respuesta = tokenizer.batch_decode(respuesta_generada, skip_special_tokens=True)
            result_holder.append(respuesta[0])
        except Exception as e:
            print(f"⚠️ Error generando respuesta: {e}")
            result_holder.append("")
        finally:
            del mensajes_tokenizados, respuesta_generada
            torch.cuda.empty_cache()
    def tarea():
        result_holder = []
        generar_respuesta_thread(prompt, contexto, modelo, tokenizer, max_tokens, result_holder)
        return result_holder[0] if result_holder else ""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tarea)
        try:
            resultado = future.result(timeout=timeout)
            return resultado
        except FuturesTimeoutError:
            print("⏰ Tiempo de espera agotado. Terminando generación...", flush=True)
            return ""

# Calcula el tamaño de bloques por comunidad sensible para evaluar por grupos
def calcular_array_comunidades(nombre_archivo, ruta_prompts_csv, plantillas_json, carpeta_plantillas, tipo_json):
    array_resultado = []
    with open(ruta_prompts_csv, 'r', encoding='utf-8') as f:
        num_lineas = sum(1 for _ in f) - 1
    for archivo_json in plantillas_json:
        if archivo_json.endswith('.json') and tipo_json in archivo_json:
            ruta_json = os.path.join(carpeta_plantillas, archivo_json)
            with open(ruta_json, 'r', encoding='utf-8') as fjson:
                datos = json.load(fjson)
            for sesgo in datos['sesgos_a_analizar']:
                if sesgo.get('preocupacion_etica', '').upper() in nombre_archivo:
                    n_comunidades = len(sesgo.get('comunidades_sensibles', []))
                    if n_comunidades > 0:
                        repeticiones = num_lineas // n_comunidades
                        resto = num_lineas % n_comunidades
                        array_resultado.extend([n_comunidades] * repeticiones)
                        if resto > 0:
                            array_resultado.append(resto)
    return array_resultado

# Obtiene el contexto y el tipo de evaluación a partir de la plantilla JSON asociada al CSV
def obtener_contexto_y_tipo_evaluacion(nombre_archivo, plantillas_json, carpeta_plantillas):
    contexto = ''
    tipo_evaluacion = ''
    for archivo_json in plantillas_json:
        if archivo_json.endswith('.json'):
            ruta_json = os.path.join(carpeta_plantillas, archivo_json)
            with open(ruta_json, 'r', encoding='utf-8') as fjson:
                datos = json.load(fjson)
            if datos.get('tipo_evaluacion', '').replace(' ', '_').upper() in nombre_archivo.upper():
                contexto = datos.get('config_prompt', {}).get('respuesta_esperada', '')
                tipo_evaluacion = datos.get('tipo_evaluacion', '')
                break
    return contexto, tipo_evaluacion

# Función para guardar el fichero csv con los datos generados
def procesar_y_guardar_respuesta(respuesta, ruta_csv):
    with open(ruta_csv, 'w', encoding='utf-8', newline='') as f_csv:
        f_csv.write(respuesta)
    print("----------------------")
    print(f"📄 Respuesta guardada como: {os.path.basename(ruta_csv)}")