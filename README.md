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

Para lograr entender cómo de sesgado está un modelo LLM, se ha desarrolla esta herrmamienta que permite entender de un simple vistazo y, en función de una serie de: contextos, escenarios, comunidades sensibles y sesgos, cómo de sesgado está un modelo, ofreciendo valores realistas y cuantificables.

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

- Agregar una semilla para hacer los resultados reproducibles.
- Verificar que el .csv generado con los prompts es válido.
- Revisar el método de generación de prompts para la evaluación por respuestas múltiples.
- Probar a establecer previamente un rol concreto al modelo.
- Añadir la opción de usar un modelo en local sin CUDA.
- Implementar la llamada al modelo generador vía API.
- Crear una colección de modelos para usar por defecto.

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
- Guillermo Iglesias Hernández
