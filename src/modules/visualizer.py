import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from config.seed_utils import set_seed
# Establecer una semilla para reproducibilidad (descomenta para activar)
# set_seed(72)

# Clase encargada de la visualización y guardado de los resultados y gráficos del análisis.
class Visualizer:
    def __init__(self, config):
        self.config = config

    # Representación de resultados generales (vista rápida del rendimiento general)
    def plot_resultados_generales(self, df_acumulado, carpeta_graficos):
        resultados_filtrados = df_acumulado[df_acumulado['resultado'].isin(['acierto', 'error', 'fallo'])]
        resultados_counts = resultados_filtrados['resultado'].value_counts()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=resultados_counts.index, y=resultados_counts.values, palette='pastel')
        plt.title('Distribución de Resultados Generales')
        plt.ylabel('Cantidad de respuestas')
        plt.xlabel('Resultado')
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_graficos, 'resultados_generales.png'))

    # Distribución de aciertos/fallos/errores por tipo de evaluación
    def plot_resultados_tipo_evaluacion(self, df_acumulado, carpeta_graficos):
        df_filtrado = df_acumulado[df_acumulado['resultado'].isin(['acierto', 'error', 'fallo'])]
        plt.figure(figsize=(12, 6))
        sns.countplot(
            data=df_filtrado,
            x='tipo_evaluacion',
            hue='resultado',
            palette='Set2',
            edgecolor='black'
        )
        plt.title('Distribución de Aciertos, Fallos y Errores por Tipo de Evaluación', fontsize=14)
        plt.xlabel('Tipo de Evaluación')
        plt.ylabel('Cantidad de Respuestas')
        plt.legend(title='Resultado')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_graficos, 'resultados_tipo_evaluacion.png'))

    # Mapa de calor de proporciones por tipo de evaluación
    def plot_mapa_calor(self, df_acumulado, carpeta_graficos):
        df_mapa_calor = df_acumulado.groupby(['tipo_evaluacion', 'resultado']).size().unstack(fill_value=0)
        df_mapa_calor_prop = df_mapa_calor.div(df_mapa_calor.sum(axis=1), axis=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_mapa_calor_prop, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'Proporción'})
        plt.title('Proporción de Resultados por Tipo de Evaluación')
        plt.xlabel('Resultado')
        plt.ylabel('Tipo de Evaluación')
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_graficos, 'mapa_calor_tipo_evaluacion.png'))

    # Mapa interactivo por comunidad sensible (histograma facetado)
    def plot_interactive(self, df_acumulado, carpeta_graficos, abreviaciones):
        df_acumulado['tipo_evaluacion'] = df_acumulado['tipo_evaluacion'].replace(abreviaciones)
        df_acumulado['comunidad_sensible'] = df_acumulado['comunidad_sensible'].astype(str)
        fig = px.histogram(
            df_acumulado,
            x="tipo_evaluacion",
            color="resultado",
            barmode="group",
            facet_col="comunidad_sensible",
            category_orders={"resultado": ["acierto", "fallo", "error"]},
            title="Distribución por comunidad sensible"
        )
        for annotation in fig.layout.annotations:
            if 'comunidad_sensible=' in annotation.text:
                annotation.text = annotation.text.split('=')[1]
                annotation.textangle = -15
            if "tipo_evaluacion" in annotation.text:
                annotation.text = ""
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Cantidad de respuestas",
            legend_title="Resultado",
            margin=dict(l=20, r=20, t=120, b=100),  
            height=600,
            bargap=0.3
        )
        fig.update_xaxes(tickangle=-45)
        fig.write_html(os.path.join(carpeta_graficos, 'grafico_resultados_interactivo.html'))

    # Gráfico de z-scores y outliers para análisis de sentimiento
    def plot_outliers_analisis_sentimiento(self, df_acumulado, carpeta_graficos):
        if 'z_outlier' not in df_acumulado.columns:
            return
        datos = df_acumulado[df_acumulado['tipo_evaluacion'] == 'preguntas_analisis_sentimiento']
        if datos.empty:
            return
        datos = datos.dropna(subset=['z_neg', 'z_neu', 'z_pos', 'z_outlier'])
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        plt.suptitle("Distribución de resultados - Preguntas Análisis Sentimiento", fontsize=16)
        # Parte superior: líneas z_neg, z_neu, z_pos
        axes[0].plot(datos.index, datos['z_neg'], label='z_neg', color='red')
        axes[0].plot(datos.index, datos['z_neu'], label='z_neu', color='gray')
        axes[0].plot(datos.index, datos['z_pos'], label='z_pos', color='green')
        axes[0].axhline(2, color='black', linestyle='--', linewidth=1, label='umbral=2')
        axes[0].legend()
        axes[0].set_ylabel("Z-score")
        axes[0].set_title("Z-scores por entrada")
        sns.scatterplot(
            x=datos.index,
            y=['Outlier'] * len(datos),
            hue=datos['z_outlier'],
            palette={'positivo': 'green', 'negativo': 'red', 'neutral': 'gray', 'ninguno': 'blue'},
            ax=axes[1],
            s=60
        )
        for i, row in datos.iterrows():
            if row['z_outlier'] != 'ninguno':
                nombre = row['comunidad_sensible']
                axes[1].annotate(
                    nombre,
                    (i, 0),
                    textcoords="offset points",
                    xytext=(0, -80),
                    ha='center',
                    fontsize=7,
                    rotation=90,
                    clip_on=False
                )
        axes[1].set_title("Clasificación de outliers análisis sentimiento")
        axes[1].legend(
            title="Outlier",
            bbox_to_anchor=(0.5, 1.25),
            loc='lower center',
            ncol=4,
            frameon=True
        )
        axes[1].set_xticks([])
        axes[1].set_xlabel("")
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        plt.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(carpeta_graficos, 'outliers_analisis_sentimiento.png'))

    # Gráfico de z-score y outliers para cerradas probabilidad
    def plot_outliers_cerradas_probabilidad(self, df_acumulado, carpeta_graficos):
        if 'z_outlier' not in df_acumulado.columns:
            return
        datos = df_acumulado[df_acumulado['tipo_evaluacion'] == 'preguntas_cerradas_probabilidad']
        if datos.empty:
            return
        datos = datos.dropna(subset=['z_probabilidad', 'z_outlier'])
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        plt.suptitle("Z-score y clasificación de outliers - Preguntas Cerradas Probabilidad", fontsize=16)
        axes[0].plot(datos.index, datos['z_probabilidad'], color='purple', label='z_probabilidad')
        axes[0].axhline(1.5, color='green', linestyle='--', label='umbral superior')
        axes[0].axhline(-1.5, color='red', linestyle='--', label='umbral inferior')
        axes[0].set_ylabel("Z-score")
        axes[0].legend()
        axes[0].set_title("Z-score de probabilidad por entrada")
        # Parte inferior: z_outlier
        sns.scatterplot(
            x=datos.index,
            y=['Outlier'] * len(datos),
            hue=datos['z_outlier'],
            palette={'superior': 'green', 'inferior': 'red', 'neutral': 'gray'},
            ax=axes[1],
            s=60
        )
        for i, row in datos.iterrows():
            if row['z_outlier'] != 'neutral':
                nombre = row['comunidad_sensible']
                axes[1].annotate(
                    nombre,
                    (i, 0),
                    textcoords="offset points",
                    xytext=(0, -80),
                    ha='center',
                    fontsize=7,
                    rotation=90,
                    clip_on=False
                )
        axes[1].set_title("Clasificación de outliers análisis sentimiento")
        axes[1].legend(
            title="Outlier",
            bbox_to_anchor=(0.5, 1.25),
            loc='lower center',
            ncol=4,
            frameon=True
        )
        axes[1].set_xticks([])
        axes[1].set_xlabel("")
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        plt.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(carpeta_graficos, 'outliers_cerradas_probabilidad.png'))

    # Gráfico de balance, % fuera de contexto y outliers para respuestas múltiples
    def plot_outliers_respuestas_multiples(self, df_acumulado, carpeta_graficos):
        if 'z_outlier' not in df_acumulado.columns:
            return
        datos = df_acumulado[df_acumulado['tipo_evaluacion'] == 'preguntas_respuestas_multiples']
        if datos.empty:
            return
        datos = datos.dropna(subset=['balance_estereotipos', 'porcentaje_fuera_contexto', 'z_outlier'])
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 3, 1]})
        plt.suptitle("Distribución de resultados - Preguntas Respuestas Múltiples", fontsize=16)

        axes[0].plot(datos.index, datos['balance_estereotipos'], label='Balance Estereotipos', color='blue')
        axes[0].set_ylabel("Balance Estereotipos")
        axes[0].set_title("Balance de estereotipos por entrada")
        axes[0].legend()

        axes[1].plot(datos.index, datos['porcentaje_fuera_contexto'], label='% Fuera de Contexto', color='orange')
        axes[1].axhline(20, color='black', linestyle='--', linewidth=1, label='umbral=20%')
        axes[1].set_ylabel("% Fuera de Contexto")
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Porcentaje de respuestas fuera de contexto por entrada")
        axes[1].legend()

        sns.scatterplot(
            x=datos.index,
            y=['Outlier'] * len(datos),
            hue=datos['z_outlier'],
            palette={
                'antioestereotipada_y_fuera_de_contexto': 'purple',
                'estereotipada_y_fuera_de_contexto': 'brown',
                'antioestereotipada': 'green',
                'estereotipada': 'red',
                'fuera_de_contexto': 'black',
                'neutral': 'gray'
            },
            ax=axes[2],
            s=70
        )
        for i, row in datos.iterrows():
            if row['z_outlier'] != 'neutral':
                nombre = row['comunidad_sensible']
                axes[2].annotate(
                    nombre,
                    (i, 0),
                    textcoords="offset points",
                    xytext=(0, -80),
                    ha='center',
                    fontsize=7,
                    rotation=90,
                    clip_on=False
                )
        axes[2].set_yticks([])
        axes[2].set_title("Clasificación de outliers")
        axes[2].legend(
            title="Outlier",
            bbox_to_anchor=(0.5, 1.25),
            loc='lower center',
            ncol=2,
            frameon=True
        )
        axes[2].set_xticks([])
        axes[2].set_xlabel("")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(carpeta_graficos, 'outliers_respuestas_multiples.png'))
