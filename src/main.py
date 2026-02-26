# Script principal que orquesta todo el flujo: generación de prompts, evaluación, análisis y visualización.
from modules.config_manager import ConfigManager
from modules.model_manager import ModelManager
from modules.prompt_generator import PromptGenerator
from modules.evaluator import Evaluator
from modules.analyzer import Analyzer
from modules.visualizer import Visualizer
from utils.helpers import (
    formatear_tiempo,
    invocar_modelo,
    calcular_array_comunidades,
    obtener_contexto_y_tipo_evaluacion,
    procesar_y_guardar_respuesta
)
from validadores.validador_insensible import ValidadorInsensible
from pprint import pprint
import re
import os
import pandas as pd
import csv
import json
import shutil
import datetime
from utils.reproducibility import set_seed
# Establecer una semilla para reproducibilidad (descomenta para activar)
# set_seed(72)

def main():
    config = ConfigManager()
    model_manager = ModelManager(config)
    model_manager.load_generator()
    model_manager.load_evaluator()
    model_manager.load_sentiment()

    prompt_generator = PromptGenerator(config, model_manager)
    evaluator = Evaluator(config, model_manager)
    analyzer = Analyzer(config)
    visualizer = Visualizer(config)

    carpeta_evaluacion_personalizada = 'evaluacion_personalizada'
    os.makedirs(carpeta_evaluacion_personalizada, exist_ok=True)
    
    abreviaciones = {
        "preguntas_respuestas_multiples": "PRM",
        "preguntas_cerradas_probabilidad": "PCP",
        "preguntas_prompt_injection": "PPI",
        "preguntas_agente": "PA",
        "preguntas_analisis_sentimiento": "PAS",
        "preguntas_cerradas_esperadas": "PCS"
    }

    print("----------------------")
    respuesta_prompts = input("¿Quieres utilizar el proceso de generación de prompts personalizado? ([Y]/n): ").strip().lower()
    if respuesta_prompts == 'n':
        print("Se va a proceder a comenzar con el proceso que hace uso de prompts predefinidos.")
        carpeta_salida_respuestas = 'evaluacion_por_defecto/respuestas_modelo_evaluado'
        carpeta_graficos = 'evaluacion_por_defecto/graficos'
        carpeta_plantillas = 'evaluacion_por_defecto/plantillas_evaluacion_por_defecto'
    else:
        print("Se va a proceder a comenzar con el proceso que hace uso de prompts personalizados.")
        carpeta_salida_respuestas = 'evaluacion_personalizada/respuestas_modelo_evaluado'
        carpeta_graficos = 'evaluacion_personalizada/graficos'
        carpeta_plantillas = 'evaluacion_personalizada/plantillas_evaluacion_json'
        carpeta_metaprompts_salida = 'evaluacion_personalizada/plantillas_metaprompts_txt'
        carpeta_prompts_salida = 'evaluacion_personalizada/prompts_generados_csv'
        carpeta_prompts_salida_erroneos = 'evaluacion_personalizada/prompts_generados_csv_erroneos'
        carpeta_salida_csv = 'evaluacion_personalizada/prompts_dataset'
        carpetas = [
            carpeta_graficos,
            carpeta_metaprompts_salida,
            carpeta_prompts_salida,
            carpeta_prompts_salida_erroneos,
            carpeta_salida_csv
        ]

        for carpeta in carpetas:
            if os.path.exists(carpeta):
                shutil.rmtree(carpeta)  # Borra la carpeta y su contenido
            os.makedirs(carpeta)  


    if os.path.exists(carpeta_salida_respuestas):
        shutil.rmtree(carpeta_salida_respuestas)
    os.makedirs(carpeta_salida_respuestas)

    if os.path.exists(carpeta_graficos):
        shutil.rmtree(carpeta_graficos)
    os.makedirs(carpeta_graficos)

    print("----------------------")
    print("🔍 Estimando la carga de trabajo prevista antes de ejecutar el modelo...")
    total_llamadas_evaluacion = 0
    total_llamadas_mejor_caso = 0
    total_llamadas_peor_caso = 0
    tiempo_max_por_llamada = 180
    plantillas_json = os.listdir(carpeta_plantillas)
    max_tokens = 7500

    for archivo_json in plantillas_json:
        if archivo_json.endswith('.json'):
            ruta_json = os.path.join(carpeta_plantillas, archivo_json)
            with open(ruta_json, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            numero_prompts = datos.get('numero_prompts', 0)
            numero_reintentos = datos.get('numero_reintentos', 1)
            sesgos = datos.get('sesgos_a_analizar', [])
            for sesgo in sesgos:
                contextos = sesgo.get('contextos', [])
                num_comunidades = len(sesgo.get('comunidades_sensibles', 1))
                for contexto in contextos:
                    total_llamadas_evaluacion += numero_prompts * num_comunidades
                    total_llamadas_mejor_caso += 1
                    total_llamadas_peor_caso += numero_reintentos

    tiempo_max_mejor = total_llamadas_mejor_caso * tiempo_max_por_llamada
    tiempo_max_peor = total_llamadas_peor_caso * tiempo_max_por_llamada
    tiempo_max_evaluacion = total_llamadas_evaluacion * tiempo_max_por_llamada

    print("----------------------")
    print(f"Plantillas de evaluación encontradas: {len(plantillas_json)} plantillas")
    print(f"Total de prompts finales a generar (estimado): {total_llamadas_evaluacion}")

    print("\nEstimación del número de llamadas que se realizarán al modelo generador y su duración:")
    print(f"- En el mejor de los casos (estimado): {total_llamadas_mejor_caso} llamadas, con una duración máxima de {formatear_tiempo(tiempo_max_mejor)}")
    print(f"- En el peor de los casos (estimado): {total_llamadas_peor_caso} llamadas, con una duración máxima de {formatear_tiempo(tiempo_max_peor)}")

    print("\nEstimación del número de llamadas que se realizarán al modelo a evaluar y su duración:")
    print(f"Número de llamadas a realizar (estimado): {total_llamadas_evaluacion} llamadas, con una duración máxima de {formatear_tiempo(tiempo_max_evaluacion)}")

    if respuesta_prompts != 'n':
        print("----------------------")
        respuesta = input("¿Quieres comenzar el proceso de generación de prompts? ([Y]/n): ").strip().lower()
        if respuesta == 'n':
            print("Proceso cancelado por el usuario.")
            exit(0)

    fecha_inicio = datetime.datetime.now()
    print("----------------------")
    print(f"🕒 Inicio del proceso: {fecha_inicio.strftime('%d-%m-%Y %H:%M:%S')}")
    
    if( respuesta_prompts != 'n'):
        with open('config/meta_prompt.txt', 'r', encoding='utf-8') as f:
            texto_base = f.read()
        for archivo_json in plantillas_json:
            if archivo_json.endswith('.json'):
                ruta_json = os.path.join(carpeta_plantillas, archivo_json)
                with open(ruta_json, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                datos_globales = prompt_generator.flatten_json({
                    k: v for k, v in datos.items() if k not in ['sesgos_a_analizar', 'config_prompt']
                })
                datos_globales.update(datos.get('config_prompt', {}))
                esquema_salida = datos['config_prompt'].get('esquema_salida', {})
                datos_globales['esquema_salida'] = json.dumps(esquema_salida, ensure_ascii=False)
                for sesgo in datos['sesgos_a_analizar']:
                    sesgo_base = {
                        'preocupacion_etica': sesgo.get('preocupacion_etica', ''),
                        'comunidades_sensibles': ', '.join(map(str, sesgo.get('comunidades_sensibles', []))),
                        'marcador': sesgo.get('marcador', '')
                    }
                    for contexto_obj in sesgo.get('contextos', []):
                        contexto_nombre = contexto_obj.get('contexto', '')
                        ejemplo_salida = ''.join(contexto_obj.get('ejemplo_salida', []))
                        datos_combinados = {
                            **datos_globales,
                            **sesgo_base,
                            'contexto': contexto_nombre,
                            'ejemplo_salida': ejemplo_salida
                        }
                        if 'tipos_inyeccion' in datos['config_prompt']:
                            cadena_tipos_inyeccion = ', '.join(f'"{e}"' for e in datos_globales.get('tipos_inyeccion', []))
                            datos_combinados['tipos_inyeccion'] = cadena_tipos_inyeccion[1:-1]
                        cadena_escenarios = ', '.join(f'"{e}"' for e in contexto_obj.get('escenarios', []))
                        datos_combinados['escenarios'] = cadena_escenarios[1:-1]
                        texto_intermedio = prompt_generator.sustituir_claves(texto_base, datos_combinados)
                        texto_final = prompt_generator.sustituir_claves(texto_intermedio, datos_combinados)

                        tipo_eval = datos_combinados['tipo_evaluacion'].replace(' ', '_').upper()
                        preocupacion = sesgo_base['preocupacion_etica'].replace(' ', '_').upper()
                        contexto_slug = contexto_nombre.replace(' ', '_').upper()
                        nombre_archivo = f"meta_prompt_{tipo_eval}_sesgo_{preocupacion}_contexto_{contexto_slug}.txt"
                        ruta_salida_metaprompt = os.path.join(carpeta_metaprompts_salida, nombre_archivo)
                        prompt_generator.save_prompt(texto_final, ruta_salida_metaprompt)
                        print("----------------------")
                        print(f"✔ Plantilla guardada como: {nombre_archivo}")

                        idioma = datos['config_prompt'].get('idioma_prompts', {})
                        nombre_csv = nombre_archivo.replace('meta_prompt_', 'prompts_generados_').replace('.txt', '.csv')
                        ruta_csv = os.path.join(carpeta_prompts_salida, nombre_csv)
                        ruta_csv_erroneo = os.path.join(carpeta_prompts_salida_erroneos, nombre_csv)
                        numero_reintentos = datos_globales['numero_reintentos']
                        recuento_reintentos = 0
                        marcador_plantilla = rf"{{{datos_combinados['marcador']}}}"

                        if config.modelo_generador['modo_interaccion'] == "API":
                            print("----------------------")
                            print(f"🌐 Enviando prompt a modelo ({config.modelo_generador['id_modelo']})...")
                            try:
                                respuesta = ""
                                
                                procesar_y_guardar_respuesta(respuesta, ruta_csv)

                            except Exception as e:
                                print("----------------------")
                                print(f"❌ Error al invocar modelo API para {nombre_archivo}: {e}")

                        elif config.modelo_generador['modo_interaccion'] == "local":
                            print("----------------------")
                            print(f"💻 Ejecutando modelo local ({config.modelo_generador['id_modelo']})...")
                            try:
                                while recuento_reintentos < numero_reintentos:
                                    print("----------------------")
                                    print("Número de intento: ", recuento_reintentos+1)
                                    nombre_sin_ext, extension = ruta_csv.rsplit('.', 1)
                                    ruta_csv_intento = f"{nombre_sin_ext}_{recuento_reintentos+1}.{extension}"
                                    nombre_sin_ext_err, extension = ruta_csv_erroneo.rsplit('.', 1)
                                    ruta_csv_erroneo_intento = f"{nombre_sin_ext_err}_{recuento_reintentos+1}.{extension}"
                                    print("----------------------")
                                    schema_str = prompt_generator.sustituir_claves(datos_globales['esquema_salida'], datos_combinados)
                                    schema_validacion = json.loads(schema_str)
                                    pprint(schema_validacion)
                                    v = ValidadorInsensible(schema_validacion, require_all=True)
                                    respuesta = invocar_modelo(
                                        texto_final,
                                        model_manager.models['generator'],
                                        model_manager.tokenizers['generator'],
                                        max_tokens,
                                        f"Eres un generador de prompts en idioma: {idioma} para evaluar preocupaciones éticas. Debes seguir estrictamente las instrucciones dadas en el mensaje del usuario y responder únicamente con un CSV válido, sin introducciones ni conclusiones."
                                    )
                                    respuesta_limpia = prompt_generator.limpiar_respuesta_generada(
                                        respuesta,
                                        datos.get('numero_prompts', 0),
                                        esquema_salida,
                                        marcador_plantilla,
                                        datos['tipo_evaluacion'],
                                        sesgo.get('comunidades_sensibles', [])
                                    )
                                    procesar_y_guardar_respuesta(respuesta_limpia, ruta_csv_erroneo_intento)
                                    fila_erronea = False
                                    with open(ruta_csv_erroneo_intento, newline='', encoding='utf-8') as csvfile:
                                        reader_csv = csv.DictReader(csvfile, delimiter='|')
                                        filas = list(reader_csv)
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
                                            print(f"🗑️  Eliminado archivo: {ruta_csv_erroneo_intento}")
                                        except Exception as e:
                                            print(f"❌ No se pudo eliminar {ruta_csv_erroneo_intento}: {e}")
                                        recuento_reintentos = numero_reintentos
                                        ruta_salida_csv = os.path.join(carpeta_salida_csv, nombre_csv)
                                        comunidades_sensibles = sesgo.get('comunidades_sensibles', [])
                                        with open(ruta_csv_intento, mode='r', encoding='utf-8') as archivo_entrada, open(ruta_salida_csv, mode='w', newline='', encoding='utf-8') as archivo_salida:
                                            reader = csv.DictReader(archivo_entrada, delimiter='|')
                                            cabecera = [campo for campo in reader.fieldnames if campo not in ('id', 'comunidad_sensible')]
                                            cabecera.append('comunidad_sensible')
                                            writer = csv.DictWriter(archivo_salida, fieldnames=cabecera, delimiter='|', quoting=csv.QUOTE_MINIMAL)
                                            writer.writeheader()
                                            for fila in reader:
                                                fila.pop('id', None)
                                                if not "PREGUNTAS_RESPUESTAS_MULTIPLES" in ruta_csv_intento:
                                                    fila_original = fila['prompt']
                                                    marcadores = re.findall(marcador_plantilla, fila_original)
                                                    num_marcadores = len(marcadores)
                                                    if num_marcadores == 1:
                                                        for comunidad in comunidades_sensibles:
                                                            fila_modificada = fila.copy()
                                                            nuevo_prompt = re.sub(marcador_plantilla, comunidad, fila_original, count=1)
                                                            fila_modificada['prompt'] = nuevo_prompt
                                                            fila_modificada['comunidad_sensible'] = comunidad
                                                            writer.writerow(fila_modificada)
                                                else:
                                                    fila_modificada = fila.copy()
                                                    comunidad_encontrada = False
                                                    for comunidad in comunidades_sensibles:
                                                        if comunidad in fila_modificada['prompt'] and not comunidad_encontrada:
                                                            fila_modificada['comunidad_sensible'] = comunidad
                                                            writer.writerow(fila_modificada)
                                                            comunidad_encontrada = True
                                    else:
                                        print("----------------------")
                                        print(f"❌ El csv generado NO es válido.")
                                        recuento_reintentos += 1
                            except Exception as e:
                                print("----------------------")
                                print(f"❌ Error ejecutando modelo local para {nombre_archivo}: {e}")
    
    if respuesta_prompts == 'n':
        ruta_prompts = 'evaluacion_por_defecto/prompts_por_defecto'
    else:
        ruta_prompts = carpeta_salida_csv

    df_acumulado = pd.DataFrame() # DataFrame con la info de todas las respuestas del modelo a evaluar y sus evaluaciones
    array_comunidades_sentimientos = []
    array_comunidades_probabilidad = []

    for nombre_archivo in os.listdir(ruta_prompts):
        if nombre_archivo.endswith(".csv"):
            ruta_prompts_csv = os.path.join(ruta_prompts, nombre_archivo)
            ruta_respuestas_salida = os.path.join(carpeta_salida_respuestas, nombre_archivo)
            with open(ruta_prompts_csv, newline='', encoding='utf-8') as archivo_csv_prompts:
                reader = csv.DictReader(archivo_csv_prompts, delimiter='|')
                filas_prompts = []

                if "PREGUNTAS_ANALISIS_SENTIMIENTO" in nombre_archivo:
                    array_comunidades_sentimientos = calcular_array_comunidades(
                        nombre_archivo, ruta_prompts_csv, plantillas_json, carpeta_plantillas, "analisis_sentimiento"
                    )
                if "CERRADAS_PROBABILIDAD" in nombre_archivo.upper():
                    array_comunidades_probabilidad = calcular_array_comunidades(
                        nombre_archivo, ruta_prompts_csv, plantillas_json, carpeta_plantillas, "cerradas_probabilidad"
                    )

                for fila_aux in reader:
                    prompt = fila_aux['prompt']
                    contexto, tipo_evaluacion = obtener_contexto_y_tipo_evaluacion(
                        nombre_archivo, plantillas_json, carpeta_plantillas
                    )
                    modelo, tokenizer = model_manager.get_model('evaluator')
                    if config.modelo_a_evaluar['modo_interaccion'] == "local":
                        respuesta = invocar_modelo(prompt, modelo, tokenizer, max_tokens, contexto)
                        respuesta = prompt_generator.limpiar_respuesta_generada_evaluacion(tipo_evaluacion, respuesta)
                    else:
                        respuesta = ""
                    fila_aux['tipo_evaluacion'] = tipo_evaluacion
                    fila_aux['respuesta_modelo'] = respuesta
                    filas_prompts.append(fila_aux)
                fieldnames = reader.fieldnames + ['respuesta_modelo', 'tipo_evaluacion']
                print("---------------------------------------------------------------------------------")
                print(f"\n🕒 Se han generado las respuestas para {nombre_archivo}: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
                with open(ruta_respuestas_salida, mode='w', newline='', encoding='utf-8') as archivo_respuestas_salida:
                    writer = csv.DictWriter(archivo_respuestas_salida, fieldnames=fieldnames, delimiter='|')
                    writer.writeheader()
                    writer.writerows(filas_prompts)
            df_resultados = pd.read_csv(ruta_respuestas_salida, delimiter='|')
            df_resultados['respuesta_modelo'] = df_resultados['respuesta_modelo'].astype(str).str.strip()
            modelo_sent, tokenizer_sent = model_manager.get_model('sentiment')
            def invocar_modelo_analisis_sentimiento(prompt, modelo, tokenizer):
                import torch
                import torch.nn.functional as F
                try:
                    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                        return ""
                    mensajes_tokenizados =  tokenizer(prompt, return_tensors="pt", truncation=True, padding=True,  max_length=512)
                    mensajes_tokenizados = {k: v.to("cuda") for k, v in mensajes_tokenizados.items()}
                    with torch.no_grad():
                        outputs = modelo(**mensajes_tokenizados)
                        logits = outputs.logits
                        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                    etiquetas = ['Negative', 'Neutral', 'Positive']
                    resultado = {etiqueta: round(prob.item(), 4) for etiqueta, prob in zip(etiquetas, probs)}
                    torch.cuda.empty_cache()
                    return json.dumps(resultado)
                except Exception as e:
                    return 'error'
            if df_resultados.empty:
                print(f"⚠️  El archivo {nombre_archivo} no contiene filas para evaluar.")
                df_resultados['resultado'] = pd.Series(dtype='object')
            else:
                df_resultados['resultado'] = [
                    evaluator.evaluar_respuestas(
                        fila, nombre_archivo,
                        modelo_analisis_de_sentimiento=modelo_sent,
                        tokenizer_analisis_sentimiento=tokenizer_sent,
                        invocar_modelo_analisis_sentimiento=invocar_modelo_analisis_sentimiento
                    )
                    for _, fila in df_resultados.iterrows()
                ]
            df_acumulado = pd.concat([df_acumulado, df_resultados], ignore_index=True)

    total = len(df_acumulado)
    if total > 0:
        df_acumulado = analyzer.analisis_avanzado_resultados(
            df_acumulado,
            array_comunidades_sentimientos,
            array_comunidades_probabilidad,
            carpeta_graficos,
            abreviaciones
        )
        visualizer.plot_resultados_generales(df_acumulado, carpeta_graficos)
        visualizer.plot_resultados_tipo_evaluacion(df_acumulado, carpeta_graficos)
        visualizer.plot_mapa_calor(df_acumulado, carpeta_graficos)
        visualizer.plot_outliers_analisis_sentimiento(df_acumulado, carpeta_graficos)
        visualizer.plot_outliers_cerradas_probabilidad(df_acumulado, carpeta_graficos)
        visualizer.plot_outliers_respuestas_multiples(df_acumulado, carpeta_graficos)
        visualizer.plot_interactive(df_acumulado, carpeta_graficos, abreviaciones)
        df_acumulado.to_csv(os.path.join(carpeta_graficos, 'resultados.csv'), sep='|', index=False)
        df_acumulado.to_excel(os.path.join(carpeta_graficos, 'resultados.xlsx'), index=False, sheet_name='Resultados')
        print(f"Resultados guardados en {carpeta_graficos}")
    else:
        print("No hay resultados")

    fecha_fin = datetime.datetime.now()
    duracion = fecha_fin - fecha_inicio
    print("----------------------")
    print(f"\n🕒 Fin del proceso: {fecha_fin.strftime('%d-%m-%Y %H:%M:%S')}")
    print(f"⏱️ Duración total: {str(duracion).split('.')[0]}")

if __name__ == "__main__":
    main()
