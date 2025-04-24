import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
import torch

# Rutas de carpetas
carpeta_plantillas_json = 'Plantillas evaluacion JSON'          # Carpeta donde están las plantillas JSON para cada tipo de evaluación
carpeta_metaprompts_salida = 'Plantillas metaprompts TXT'        # Carpeta donde se guardarán los archivos txt con los metaprompts como salida
carpeta_prompts_salida = 'Prompts Generados CSV'                # Carpeta donde se guardarán los archivos csv con los prompts generados por un modelo
os.makedirs(carpeta_metaprompts_salida, exist_ok=True)           # Crear la carpeta de metaprompts de salida si no existe
os.makedirs(carpeta_prompts_salida, exist_ok=True)           # Crear la carpeta de prompts de salida si no existe

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

# Función para invocar modelo via API
def invocar_modelo(prompt, modelo_id):
    tokenizer = AutoTokenizer.from_pretrained(modelo_id)
    model = AutoModelForCausalLM.from_pretrained(
        modelo_id,
        device_map="auto",
        torch_dtype=torch.float16  # O torch.float32 si tu GPU no tiene soporte para float16
    )
    
    # Confirma uso de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Usando dispositivo: {device}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.95,
        temperature=1,
        streamer=streamer
    )

    respuesta = tokenizer.decode(output[0], skip_special_tokens=True)
    return respuesta

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

                # Combinar toda la información relevante
                datos_combinados = {
                    **datos_globales,
                    **sesgo_base,
                    'contexto': contexto_nombre,
                    'escenarios': escenarios
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

                # ---------- Enviar prompt al modelo si modo_interaccion es API ----------
                modo = datos.get("modelo_generador", {}).get("modo_interaccion", "")
                modelo_id = datos.get("modelo_generador", {}).get("id_modelo", "")
                if modo == "API":
                    print(f"🌐 Enviando prompt a modelo ({modelo_id})...")
                    try:
                        respuesta = invocar_modelo(texto_final, modelo_id)
                        print(respuesta)
                        # Ruta de salida para CSV con mismo nombre base que el .txt
                        nombre_csv = nombre_archivo.replace('meta_prompt_', 'prompts_generados_').replace('.txt', '.csv')
                        ruta_csv = os.path.join(carpeta_prompts_salida, nombre_csv)

                        # Guardar respuesta como CSV (una fila, una columna por simplicidad)
                        with open(ruta_csv, 'w', encoding='utf-8') as f_csv:
                            f_csv.write(respuesta)
                            # f_csv.write(f'"{respuesta.strip().replace(chr(10), " ")}"\n')  # Eliminar saltos de línea y envolver en comillas
                        print(f"📄 Respuesta guardada como: {nombre_csv}")

                    except Exception as e:
                        print(f"❌ Error al invocar modelo API para {nombre_archivo}: {e}")

                elif modo == "local":
                    print(f"💻 Ejecutando modelo local ({modelo_id})...")
                    try:
                        respuesta = invocar_modelo(texto_final, modelo_id)

                        # Ruta de salida para CSV con mismo nombre base que el .txt
                        nombre_csv = nombre_archivo.replace('meta_prompt_', 'prompts_generados_').replace('.txt', '.csv')
                        ruta_csv = os.path.join(carpeta_prompts_salida, nombre_csv)

                        # Guardar respuesta como CSV (una fila, una columna por simplicidad)
                        with open(ruta_csv, 'w', encoding='utf-8') as f_csv:
                            f_csv.write(respuesta)
                            # f_csv.write(f'"{respuesta.strip().replace(chr(10), " ")}"\n')  # Eliminar saltos de línea y envolver en comillas
                        print(f"📄 Respuesta guardada como: {nombre_csv}")
                        
                    except Exception as e:
                        print(f"❌ Error ejecutando modelo local para {nombre_archivo}: {e}")