<a name="top"></a>
# Herramienta para la evaluación automática de sesgos en modelos LLM.
**Proyecto de Fin de Grado** - **Programa Tutoría**

ETSISI UPM - Telefónica OICampus

## Tabla de contenidos
* [Descripción](#descripción)
* [Versiones](#versiones)
* [Tecnologías](#tecnologías)
* [Bibliografía](#bibliografía)

 ---
 
<a name="descripción"></a>
### 1. Descripción

Ahora más que nunca necesitamos conocer cómo se comportan los modelos LLM y cómo han ido interiorizando todos los datos que han ido aprendiendo.

Para lograr entender cómo de sesgado está un modelo LLM, se ha desarrolla esta herramienta que permite entender de un simple vistazo y, en función de una serie de: contextos, escenarios, comunidades sensibles y sesgos, cómo de sesgado está un modelo, ofreciendo valores realistas y cuantificables.

![Diagrama de Flujo](https://github.com/Pikeras72/Repositorio-TFG/blob/main/diagramas/Diagrama_de_flujo_Proceso_TFG.png)

[Subir⬆️](#top)

---

<a name="versiones"></a>
### 2. Versiones
#### ---- <Versión 0.1> --- <Actualización [05/05/2025]> ----

Versión inicial del proyecto.

**Incluye lo siguiente:**

- Se rellena, con la información de cada plantilla de evaluación de tipo .json propuesta, un metaprompt con una estructura predefinida.
- Se usan esos metaprompts rellenos de información para la generación de una colección de prompts en formato .csv para cada sesgo, escenario y contexto posibles, de cada tipo de evaluación.

**Puntos débiles:**

- Se usa solo un modelo descargado previamente en local para generar los prompts.
- Se deberían poder generar los prompts a partir de un modelo cargado vía API.
- Se tendría que poder elegir, dentro de un abanico de posibilidades, el modelo con el que queremos generar la colección de prompts.
- La estructura .csv que genera el modelo con los prompts no siempre coincide exactamente con el formato esperado.
- Para el tipo de evaluación: Respuestas múltiples, habría que modificar la forma de generar los prompts puesto que no se pueden hacer respuestas estereotipadas o antiestereotipadas a partir de una pregunta o cuestión si no se conoce la comunidad sensible que se analiza en cada caso.

**Mejoras futuras:**

- Verificar que el .csv generado con los prompts es válido. ([#1](https://github.com/Pikeras72/Repositorio-TFG/issues/1))
- Agregar una semilla para hacer los resultados reproducibles. ([#2](https://github.com/Pikeras72/Repositorio-TFG/issues/2))
- Revisar el método de generación de prompts para la evaluación por respuestas múltiples. ([#3](https://github.com/Pikeras72/Repositorio-TFG/issues/3))
- Probar a establecer previamente un rol concreto al modelo. ([#4](https://github.com/Pikeras72/Repositorio-TFG/issues/4))
- Añadir la opción de usar un modelo en local sin CUDA. ([#5](https://github.com/Pikeras72/Repositorio-TFG/issues/5))
- Implementar la llamada al modelo generador vía API. ([#6](https://github.com/Pikeras72/Repositorio-TFG/issues/6))
- Crear una colección de modelos para usar por defecto. ([#7](https://github.com/Pikeras72/Repositorio-TFG/issues/7))


#### ---- <Versión 0.2> --- <Actualización [14/05/2025]> ----

Esta versión incluye un avance en la validación, generación y limpieza de los prompts generados.

**Incluye lo siguiente:**

- Sustitución de los marcadores entre llaves '{}' por las comunidades sensibles correspondientes. ([#12](https://github.com/Pikeras72/Repositorio-TFG/issues/12))
- Eliminación de un par de columnas con información redundante de los csvs que creaba el modelo generador de prompts. ([#13](https://github.com/Pikeras72/Repositorio-TFG/issues/13))
- Dar la posibilidad de agregar una semilla para hacer los resultados reproducibles. ([#2](https://github.com/Pikeras72/Repositorio-TFG/issues/2))
- Validación de cada csv generado por el modelo. ([#1](https://github.com/Pikeras72/Repositorio-TFG/issues/1))
- Modificación de la forma en la que se generan los prompts para el tipo de evaluación por respuestas múltiples. ([#3](https://github.com/Pikeras72/Repositorio-TFG/issues/3))
- Permitir varias veces la generación de csvs para aquellas respuestas que no hayan pasado la validación del fichero. ([#8](https://github.com/Pikeras72/Repositorio-TFG/issues/8))
- Establecer un rol predeterminado al modelo generador de la forma: 'Eres un generador de prompts ...'. ([#4](https://github.com/Pikeras72/Repositorio-TFG/issues/4))

**Puntos débiles:**

- No se especifica con antelación al usuario de la herramienta el número de prompts que se van a generar.
- Los csv generados se pueden limpiar mejor de cabecera y conclusión para facilitar la validación de los mismos.
- No se conoce cómo de correcto o incorrecto es el fichero csv que se acaba de generar, no se muestra info.
- Si el proyecto sigue creciendo, se puede complicar el entender cómo es el flujo de los datos y cómo se transforman para llegar al resultado final.
- A veces el modelo generador tiene errores de ortografía para poner los distintos escenarios.

**Mejoras futuras:**

- Indicar con antelación el número de prompts a generar. ([#18](https://github.com/Pikeras72/Repositorio-TFG/issues/18))
- Mejorar la limpieza de los csv generados. ([#19](https://github.com/Pikeras72/Repositorio-TFG/issues/19))
- Hacer un esquema visual sobre el flujo de la generación del dataset. ([#20](https://github.com/Pikeras72/Repositorio-TFG/issues/20))
- Mostrar el porcentaje de filas correctas e incorrectas de cada csv durante su validación. ([#21](https://github.com/Pikeras72/Repositorio-TFG/issues/21))
- Sustituir los escenarios de los csv generados por números. ([#22](https://github.com/Pikeras72/Repositorio-TFG/issues/22))


#### ---- <Versión 0.3> --- <Actualización [22/05/2025]> ----

Con esta versión se mejora en gran medida la cantidad de prompts generados correctamente tras su limpieza y modificación.
Así como un esquema visual del proceso completo en forma de diagrama de flujo.

**Incluye lo siguiente:**

- Esquema visual en forma de diagrama de flujo del proceso completo de la herramienta. ([#20](https://github.com/Pikeras72/Repositorio-TFG/issues/20))
- Se muestra el porcentaje de filas correctas, modificadas, eliminadas y añadidas de cada csv que crea el modelo generado, antes de su validación. ([#21](https://github.com/Pikeras72/Repositorio-TFG/issues/21))
- Limpiar los csv generados, eliminando filas erróneas, introducciones o conclusiones que puedan aparecer. También se añade la cabecera si no aparece, y se borran carácteres extraños de los prompts. ([#19](https://github.com/Pikeras72/Repositorio-TFG/issues/19))
- Mejora en la sensibilidad de mayúsc. y minúsc. en el validador de csvs (librería Cerberus). ([#23](https://github.com/Pikeras72/Repositorio-TFG/issues/23))
- Se indica con antelación a comenzar el proceso, el número de prompts que se van a generar al completarlo con éxito, a lo que el usuario deberá dar autorización, o cancelarlo. [#18](https://github.com/Pikeras72/Repositorio-TFG/issues/18))

**Puntos débiles:**

- Aún falta recoger el modelo que se va a evaluar.
- Por lo tanto, también se tendrán que generar las respuestas del modelo a evaluar usando los prompts únicos generados.
- Y validar esos outputs con sus respectivas respuestas esperadas (Hacer esto para cada tipo de evaluación).
- Parece que al acabar el programa, se imprime información de iteraciones anteriores sin sentido, probablemente esté asociado con los threads.

**Mejoras futuras:**

- Generar respuestas del modelo a evaluar. ([#29](https://github.com/Pikeras72/Repositorio-TFG/issues/29))
- Recoger el modelo a evaluar. ([#30](https://github.com/Pikeras72/Repositorio-TFG/issues/30)
- Revisar cierre de los threads. ([#32](https://github.com/Pikeras72/Repositorio-TFG/issues/32))
- Validar respuestas de preguntas agente. ([#31](https://github.com/Pikeras72/Repositorio-TFG/issues/31))
- Validar respuestas de preguntas análisis de sentimiento. ([#33](https://github.com/Pikeras72/Repositorio-TFG/issues/33))
- Validar respuestas de preguntas cerradas esperadas. ([#34](https://github.com/Pikeras72/Repositorio-TFG/issues/34))
- Validar respuestas de preguntas cerradas de probabilidad. ([#35](https://github.com/Pikeras72/Repositorio-TFG/issues/35))
- Validar respuestas de preguntas con respuesta múltiple. ([#36](https://github.com/Pikeras72/Repositorio-TFG/issues/36))
- Validar respuestas de preguntas de prompt injection. ([#37](https://github.com/Pikeras72/Repositorio-TFG/issues/37))

---

<a name="tecnologías"></a>
### 3. Tecnologías


[Subir⬆️](#top)

---

<a name="bibliografía"></a>
### 4. Bibliografía


[Subir⬆️](#top)

---

## Licencia


## Autor

- Diego Ruiz Piqueras - ([Pikeras72](https://github.com/Pikeras72))

## Agradecimiento especial

- Santiago Rodriguez Sordo
- Almudena Bonet Medina
- Guillermo Iglesias Hernández - ([guillermoih](https://github.com/guillermoih))
