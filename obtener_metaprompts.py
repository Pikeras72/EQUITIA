import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
from datetime import datetime


# Rutas de carpetas
carpeta_plantillas_json = 'Plantillas evaluacion JSON'           # Carpeta donde están las plantillas JSON para cada tipo de evaluación
carpeta_metaprompts_salida = 'Plantillas metaprompts TXT'        # Carpeta donde se guardarán los archivos txt con los metaprompts como salida
carpeta_prompts_salida = 'Prompts Generados CSV'                 # Carpeta donde se guardarán los archivos csv con los prompts generados por un modelo
os.makedirs(carpeta_metaprompts_salida, exist_ok=True)           # Crear la carpeta de metaprompts de salida si no existe
os.makedirs(carpeta_prompts_salida, exist_ok=True)               # Crear la carpeta de prompts de salida si no existe

# Cargar configuración del modelo para generar los prompts
with open('config_general.json', 'r', encoding='utf-8') as f:
    config_general = json.load(f)

modelo_id = config_general['modelo_generador']['id_modelo']
modo = config_general['modelo_generador']['modo_interaccion']

# Configuración para inicializar el modelo según el modo elegido
if modo == 'API':
    print("Aún por definir")
elif modo == 'local':
    print(f"Se está intentando cargar el modelo: {modelo_id}, en modo: {modo}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(modelo_id, use_fast=True)
    modelo = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="auto")
print(f"Usando actualmente modelo: {modelo_id}, en modo: {modo}")

# Cargar el texto base con llaves a reemplazar
with open('meta_prompt.txt', 'r', encoding='utf-8') as f:
    texto_base = f.read()

# Función para aplanar un JSON (convierte estructuras anidadas en un solo nivel de claves)
def flatten_json(y, prefix=''):
    out = {}
    for key, value in y.items():
        new_key = f"{prefix}{key}" if prefix == '' else f"{prefix}_{key}"
        if isinstance(value, dict):
            out.update(flatten_json(value, new_key))  # Recursivo si hay diccionarios anidados
        elif isinstance(value, list):
            out[new_key] = ', '.join(map(str, value))  # Convertir listas en strings separados por comas
        else:
            out[new_key] = value
    return out

# Función para reemplazar las llaves {clave} en el texto por su valor correspondiente
def sustituir_claves(texto, datos):
    def reemplazo(match):
        clave = match.group(1)
        return str(datos.get(clave, f'{{{clave}}}'))  # Si no se encuentra, se deja como estaba
    return re.sub(r'{([^{}]+)}', reemplazo, texto)

# Función para invocar modelo en local
def invocar_modelo(prompt, modelo, tokenizer, max_tokens):
    mensajes = [{"role": "user", "content": prompt}]

    mensajes_tokenizados = tokenizer.apply_chat_template(mensajes, return_tensors="pt")
    model_inputs = mensajes_tokenizados.to("cuda")

    with torch.no_grad(): # Con esto se evita guardar información para retropropagación o entrenamiento
        respuesta_generada = modelo.generate(model_inputs, max_new_tokens=max_tokens, do_sample=True)
    
    respuesta = tokenizer.batch_decode(respuesta_generada, skip_special_tokens=True)

    torch.cuda.empty_cache()  # Libera memoria GPU

    return respuesta[0]

# Función para invocar modelo via API
def invocar_modelo_api(texto_final, modelo_id, max_tokens):
    print("Aún por definir")
    return 0

# Función para guardar el fichero csv con los prompts generados
def procesar_y_guardar_respuesta(respuesta, ruta_csv):
    with open(ruta_csv, 'w', encoding='utf-8', newline='') as f_csv:
        f_csv.write(respuesta)
    print(f"📄 Respuesta guardada como: {os.path.basename(ruta_csv)}")

# Mostrar por pantalla el momento exacto en el que comienza el análisis de las plantillas JSON
inicio = time.time()
fecha_inicio = datetime.now()
print(f"🕒 Inicio del proceso: {fecha_inicio.strftime('%Y-%m-%d %H:%M:%S')}")

# Recorrer todos los archivos JSON dentro de la carpeta de plantillas de evaluación
for archivo_json in os.listdir(carpeta_plantillas_json):
    if archivo_json.endswith('.json'):
        ruta_json = os.path.join(carpeta_plantillas_json, archivo_json)
        
        # Cargar datos desde el archivo JSON
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        # Aplanar los datos generales
        datos_globales = flatten_json({
            k: v for k, v in datos.items() if k not in ['sesgos_a_analizar', 'config_prompt']
        })

        # Agregar las claves de config_prompt directamente (sin el prefijo 'config_prompt')
        datos_globales.update(datos.get('config_prompt', {}))

        # Recorrer cada sesgo que se debe analizar
        for sesgo in datos['sesgos_a_analizar']:
            # Extraer los datos específicos del sesgo
            sesgo_base = {
                'preocupacion_etica': sesgo.get('preocupacion_etica', ''),
                'comunidades_sensibles': ', '.join(map(str, sesgo.get('comunidades_sensibles', []))),
                'marcador': sesgo.get('marcador', '')
            }

            # Recorrer los contextos asociados a ese sesgo
            for contexto_obj in sesgo.get('contextos', []):
                contexto_nombre = contexto_obj.get('contexto', '')
                escenarios = ', '.join(contexto_obj.get('escenarios', []))
                ejemplo_salida = ''.join(contexto_obj.get('ejemplo_salida', []))

                # Combinar toda la información relevante
                datos_combinados = {
                    **datos_globales,
                    **sesgo_base,
                    'contexto': contexto_nombre,
                    'escenarios': escenarios,
                    'ejemplo_salida' : ejemplo_salida
                }

                # Realizar doble pasada para el reemplazo de claves para cubrir llaves internas
                texto_intermedio = sustituir_claves(texto_base, datos_combinados)
                texto_final = sustituir_claves(texto_intermedio, datos_combinados)

                # Construir el nombre del archivo con las partes variables en MAYÚSCULAS
                tipo_eval = datos_combinados['tipo_evaluacion'].replace(' ', '_').upper()
                preocupacion = sesgo_base['preocupacion_etica'].replace(' ', '_').upper()
                contexto_slug = contexto_nombre.replace(' ', '_').upper()
                nombre_archivo = f"meta_prompt_{tipo_eval}_sesgo_{preocupacion}_contexto_{contexto_slug}.txt"

                # Guardar el archivo con el metaprompt completo en la carpeta de salida
                ruta_salida = os.path.join(carpeta_metaprompts_salida, nombre_archivo)
                with open(ruta_salida, 'w', encoding='utf-8') as f_out:
                    f_out.write(texto_final)

                print(f"✔ Plantilla guardada como: {nombre_archivo}")

                # ---------- Enviar prompt al modelo ----------
                # Número máximo de tokens que puede sacar el modelo como respuesta para todo el csv que genera
                max_tokens = 7500

                # Ruta de salida para el CSV con mismo nombre base que el .txt
                nombre_csv = nombre_archivo.replace('meta_prompt_', 'prompts_generados_').replace('.txt', '.csv')
                ruta_csv = os.path.join(carpeta_prompts_salida, nombre_csv)

                if modo == "API":
                    print(f"🌐 Enviando prompt a modelo ({modelo_id})...")
                    try:
                        respuesta = invocar_modelo_api(texto_final, modelo_id, max_tokens)
                        
                        procesar_y_guardar_respuesta(respuesta, ruta_csv)

                    except Exception as e:
                        print(f"❌ Error al invocar modelo API para {nombre_archivo}: {e}")

                elif modo == "local":
                    print(f"💻 Ejecutando modelo local ({modelo_id})...")
                    try:
                        respuesta = invocar_modelo(texto_final, modelo, tokenizer, max_tokens)
                        respuesta_limpia = re.sub(r'.*?\[/INST\]', '', respuesta, flags=re.DOTALL).strip()

                        procesar_y_guardar_respuesta(respuesta_limpia, ruta_csv)
                        
                    except Exception as e:
                        print(f"❌ Error ejecutando modelo local para {nombre_archivo}: {e}")

# Mostrar por pantalla el momento exacto en el que termina la generación de los csv con los prompts
fin = time.time()
fecha_fin = datetime.now()
duracion_segundos = int(fin - inicio)
minutos, segundos = divmod(duracion_segundos, 60)
print(f"🕒 Fin del proceso: {fecha_fin.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Duración total: {minutos} minutos y {segundos} segundos")
