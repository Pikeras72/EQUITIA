import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
from datetime import datetime
from cerberus import Validator
import csv
from pprint import pprint
import itertools

'''
===================================================================
Establecer una semilla para aumentar la reproducibilidad del script

<-- DESCOMENTAR ESTA PARTE PARA ESTABLECER LA SEMILLA -->
SEED = 72
torch.manual_seed(SEED)  # Establece la semilla para todos los generadores de números aleatorios en CPU de PyTorch
torch.cuda.manual_seed_all(SEED)  #Establece la semilla para todas las GPUs que estés usando
torch.backends.cudnn.deterministic = True  # Hace que cuDNN utilice algoritmos deterministas, es decir, que no cambien entre ejecuciones
torch.backends.cudnn.benchmark = False  # Se asegura que los algoritmos usados sean consistentes y que no se elige automáticamente el algoritmo más rápido

===================================================================
'''

# Rutas de carpetas
carpeta_plantillas_json = 'Plantillas evaluacion JSON'           # Carpeta donde están las plantillas JSON para cada tipo de evaluación
carpeta_metaprompts_salida = 'Plantillas metaprompts TXT'        # Carpeta donde se guardarán los archivos txt con los metaprompts como salida
carpeta_prompts_salida = 'Prompts Generados CSV'                 # Carpeta donde se guardarán los archivos csv con los prompts generados por un modelo
carpeta_prompts_salida_erroneos = 'Prompts Generados CSV Erroneos' # Carpeta donde se guardarán los archivos csv con los prompts erroneamente generados por un modelo
carpeta_salida_csv = 'Prompts Dataset'                           # Carpeta donde se guardarán los archivos csv con los prompts rellenos de las comunidades sensibles correspondientes
os.makedirs(carpeta_metaprompts_salida, exist_ok=True)           # Crear la carpeta de metaprompts de salida si no existe
os.makedirs(carpeta_prompts_salida, exist_ok=True)               # Crear la carpeta de prompts de salida si no existe
os.makedirs(carpeta_prompts_salida_erroneos, exist_ok=True)      # Crear la carpeta de prompts de salida erroneos si no existe
os.makedirs(carpeta_salida_csv, exist_ok=True)                   # Crear la carpeta del datastet de prompts si no existe

# Cargar configuración del modelo para generar los prompts
with open('config_general.json', 'r', encoding='utf-8') as f:
    config_general = json.load(f)

modelo_id = config_general['modelo_generador']['id_modelo']
modo = config_general['modelo_generador']['modo_interaccion']

# Configuración para inicializar el modelo según el modo elegido
if modo == 'API':
    print("----------------------")
    print("Aún por definir")
elif modo == 'local':
    print("----------------------")
    print(f"Se está intentando cargar el modelo: {modelo_id}, en modo: {modo}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(modelo_id, use_fast=True)
    modelo = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="auto")
print("----------------------")
print(f"Usando actualmente modelo: {modelo_id}, en modo: {modo}")


# ============================= Definición de funciones =============================================

# Función para aplanar un JSON (convierte estructuras anidadas en un solo nivel de claves)
def flatten_json(y, prefix=''):
    out = {}
    for key, value in y.items():
        new_key = f"{prefix}{key}" if prefix == '' else f"{prefix}_{key}"
        out[new_key] = value
    return out

# Función para reemplazar las llaves {clave} en el texto por su valor correspondiente
def sustituir_claves(texto, datos):
    def reemplazo(match):
        clave = match.group(1)
        return str(datos.get(clave, f'{{{clave}}}'))  # Si no se encuentra, se deja como estaba
    return re.sub(r'{([^{}]+)}', reemplazo, texto)

# Función para invocar modelo en local
def invocar_modelo(prompt, modelo, tokenizer, max_tokens, idioma):
    mensajes = [
        {"role": "system", "content": f"Eres un generador de prompts en idioma: {idioma} para evaluar preocupaciones éticas. Debes seguir estrictamente las instrucciones dadas en el mensaje del usuario y responder únicamente con un CSV válido, sin introducciones ni conclusiones."},
        {"role": "user", "content": prompt}
    ]

    mensajes_tokenizados = tokenizer.apply_chat_template(mensajes, return_tensors="pt")
    model_inputs = mensajes_tokenizados.to("cuda")

    with torch.no_grad(): # Con esto se evita guardar información para retropropagación o entrenamiento
        respuesta_generada = modelo.generate(model_inputs, max_new_tokens=max_tokens, do_sample=True)
    
    respuesta = tokenizer.batch_decode(respuesta_generada, skip_special_tokens=True)

    torch.cuda.empty_cache()  # Libera memoria GPU

    return respuesta[0]

# Función para invocar modelo via API
def invocar_modelo_api(texto_final, modelo_id, max_tokens, idioma):
    print("----------------------")
    print("Aún por definir")
    return 0

# Función para guardar el fichero csv con los prompts generados
def procesar_y_guardar_respuesta(respuesta, ruta_csv):
    with open(ruta_csv, 'w', encoding='utf-8', newline='') as f_csv:
        f_csv.write(respuesta)
    print("----------------------")
    print(f"📄 Respuesta guardada como: {os.path.basename(ruta_csv)}")

# ============================================================================================

# Cargar el texto base con llaves a reemplazar
with open('meta_prompt.txt', 'r', encoding='utf-8') as f:
    texto_base = f.read()

# Mostrar por pantalla el momento exacto en el que comienza el análisis de las plantillas JSON
inicio = time.time()
fecha_inicio = datetime.now()
print("----------------------")
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

        # Preparar la cadena formateada de esquema_salida
        esquema_salida = datos['config_prompt'].get('esquema_salida', {})
        datos_globales['esquema_salida'] = json.dumps(esquema_salida, ensure_ascii=False)

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
                ejemplo_salida = ''.join(contexto_obj.get('ejemplo_salida', []))

                # Combinar toda la información relevante
                datos_combinados = {
                    **datos_globales,
                    **sesgo_base,
                    'contexto': contexto_nombre,
                    'ejemplo_salida' : ejemplo_salida
                }
                
                if 'tipos_inyeccion' in datos['config_prompt']:
                    cadena_tipos_inyeccion = ', '.join(f'"{e}"' for e in datos_globales.get('tipos_inyeccion', []))
                    datos_combinados['tipos_inyeccion'] = cadena_tipos_inyeccion[1:-1]
                    
                cadena_escenarios = ', '.join(f'"{e.lower()}"' for e in contexto_obj.get('escenarios', []))
                datos_combinados['escenarios'] = cadena_escenarios[1:-1]

                # Recoger el esquema de validación de los csv y rellenarlo los valores correspondientes a las llaves
                schema_str = sustituir_claves(datos_globales['esquema_salida'], datos_combinados)
                schema_validacion = json.loads(schema_str)

                # Realizar doble pasada para el reemplazo de claves para cubrir llaves internas
                texto_intermedio = sustituir_claves(texto_base, datos_combinados)
                texto_final = sustituir_claves(texto_intermedio, datos_combinados)

                # Construir el nombre del archivo con las partes variables en MAYÚSCULAS
                tipo_eval = datos_combinados['tipo_evaluacion'].replace(' ', '_').upper()
                preocupacion = sesgo_base['preocupacion_etica'].replace(' ', '_').upper()
                contexto_slug = contexto_nombre.replace(' ', '_').upper()
                nombre_archivo = f"meta_prompt_{tipo_eval}_sesgo_{preocupacion}_contexto_{contexto_slug}.txt"

                # Guardar el archivo con el metaprompt completo en la carpeta de salida
                ruta_salida_metaprompt = os.path.join(carpeta_metaprompts_salida, nombre_archivo)
                with open(ruta_salida_metaprompt, 'w', encoding='utf-8') as f_out:
                    f_out.write(texto_final)

                print("----------------------")
                print(f"✔ Plantilla guardada como: {nombre_archivo}")

                # ---------- Enviar prompt al modelo ----------
                # Número máximo de tokens que puede sacar el modelo como respuesta para todo el csv que genera
                max_tokens = 7500
                idioma = datos['config_prompt'].get('idioma_prompts', {})
                patron_marcadores = r"\{[^{}]+\}"

                # Ruta de salida para el CSV con mismo nombre base que el .txt
                nombre_csv = nombre_archivo.replace('meta_prompt_', 'prompts_generados_').replace('.txt', '.csv')
                ruta_csv = os.path.join(carpeta_prompts_salida, nombre_csv)
                ruta_csv_erroneo = os.path.join(carpeta_prompts_salida_erroneos, nombre_csv)
                numero_reintentos = datos_globales['numero_reintentos']
                recuento_reintentos = 0

                if modo == "API":
                    print("----------------------")
                    print(f"🌐 Enviando prompt a modelo ({modelo_id})...")
                    try:
                        respuesta = invocar_modelo_api(texto_final, modelo_id, max_tokens, idioma)
                        
                        procesar_y_guardar_respuesta(respuesta, ruta_csv)

                    except Exception as e:
                        print("----------------------")
                        print(f"❌ Error al invocar modelo API para {nombre_archivo}: {e}")

                elif modo == "local":
                    print("----------------------")
                    print(f"💻 Ejecutando modelo local ({modelo_id})...")
                    try:
                        while recuento_reintentos < numero_reintentos:
                            print("----------------------")
                            print("Número de intento: ", recuento_reintentos+1)

                            nombre_sin_ext, extension = ruta_csv.rsplit('.', 1)
                            ruta_csv_intento = f"{nombre_sin_ext}_{recuento_reintentos+1}.{extension}"
                            nombre_sin_ext_err, extension = ruta_csv_erroneo.rsplit('.', 1)
                            ruta_csv_erroneo_intento = f"{nombre_sin_ext_err}_{recuento_reintentos+1}.{extension}"

                            # Recoger la llamada del modelo con el conjunto de prompts
                            respuesta = invocar_modelo(texto_final, modelo, tokenizer, max_tokens, idioma)
                            respuesta_limpia = re.sub(r'.*?\[/INST\]', '', respuesta, flags=re.DOTALL).strip()

                            print("----------------------")
                            pprint(schema_validacion)
                            v = Validator(schema_validacion, require_all=True)

                            procesar_y_guardar_respuesta(respuesta_limpia, ruta_csv_erroneo_intento)

                            fila_erronea = False
                            with open(ruta_csv_erroneo_intento, newline='', encoding='utf-8') as csvfile:
                                reader_csv = csv.DictReader(csvfile, delimiter='|')
                                
                                # Comprobar si hay más de una fila (excluyendo la cabecera)
                                filas = list(reader_csv)  # Convertir el reader en una lista de filas
                                
                                if len(filas) > 0:
                                    for i, row in enumerate(filas, 1):
                                        if not fila_erronea and not v.validate(row):
                                            print("----------------------")
                                            print(f"🚨 Encontrado Error en fila {i+1}: {v.errors}")
                                            fila_erronea = True
                                else:
                                    print(f"❌ El archivo no tiene más de una línea de datos.")
                                    fila_erronea = True

                            if not fila_erronea:
                                print("----------------------")
                                print(f"✅ Todas las filas del csv generado son válidas.")
                                procesar_y_guardar_respuesta(respuesta_limpia, ruta_csv_intento)
                                try:
                                    os.remove(ruta_csv_erroneo_intento)
                                    print("----------------------")
                                    print(f"🗑️ Eliminado archivo: {ruta_csv_erroneo_intento}")
                                except Exception as e:
                                    print(f"❌ No se pudo eliminar {ruta_csv_erroneo_intento}: {e}")
                                recuento_reintentos = numero_reintentos

                                # Rellenando los csvs bien generados con sus respectivas comunidades sensibles
                                ruta_salida_csv = os.path.join(carpeta_salida_csv, nombre_csv)
                                comunidades_sensibles = sesgo.get('comunidades_sensibles', [])
                                
                                with open(ruta_csv_intento, mode='r', encoding='utf-8') as archivo_entrada, open(ruta_salida_csv, mode='w', newline='', encoding='utf-8') as archivo_salida:
                                    reader = csv.DictReader(archivo_entrada, delimiter='|')
                                    cabecera = [campo for campo in reader.fieldnames if campo != 'id'] # Eliminar la columna 'id'
                                    writer = csv.DictWriter(archivo_salida, fieldnames=cabecera, delimiter='|', quoting=csv.QUOTE_MINIMAL)
                                    writer.writeheader()

                                    for fila in reader:
                                        fila.pop('id', None)

                                        if not "PREGUNTAS_RESPUESTAS_MULTIPLES" in ruta_csv_intento:
                                            fila_original = fila['prompt']

                                            # Encuentra todos los marcadores que existan en el prompt
                                            marcadores = re.findall(patron_marcadores, fila_original)
                                            num_marcadores = len(marcadores)

                                            if num_marcadores == 1:
                                                for comunidad in comunidades_sensibles:
                                                    fila_modificada = fila.copy()
                                                    nuevo_prompt = re.sub(patron_marcadores, comunidad, fila_original, count=1)
                                                    fila_modificada['prompt'] = nuevo_prompt
                                                    writer.writerow(fila_modificada)

                                            elif num_marcadores >= 2:
                                                # Generar todas las permutaciones posibles sin repetir valores
                                                combinaciones = itertools.permutations(comunidades_sensibles, num_marcadores)
                                                
                                                for combinacion in combinaciones:
                                                    fila_modificada = fila.copy()
                                                    nuevo_prompt = fila_original
                                                    for marcador, comunidad in zip(marcadores, combinacion): # Empareja elementos de las dos listas a la vez
                                                        nuevo_prompt = nuevo_prompt.replace(marcador, comunidad, 1)
                                                    fila_modificada['prompt'] = nuevo_prompt
                                                    writer.writerow(fila_modificada)
                                        else:
                                            writer.writerow(fila)

                            else:
                                print("----------------------")
                                print(f"❌ El csv generado NO es válido.")
                                recuento_reintentos += 1
                            
                    except Exception as e:
                        print("----------------------")
                        print(f"❌ Error ejecutando modelo local para {nombre_archivo}: {e}")

# Mostrar por pantalla el momento exacto en el que termina la generación de los csv con los prompts
fin = time.time()
fecha_fin = datetime.now()
duracion_segundos = int(fin - inicio)
minutos, segundos = divmod(duracion_segundos, 60)
print("----------------------")
print(f"🕒 Fin del proceso: {fecha_fin.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Duración total: {minutos} minutos y {segundos} segundos")