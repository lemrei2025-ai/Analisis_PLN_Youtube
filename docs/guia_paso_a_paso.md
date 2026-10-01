# Guía paso a paso – Proyecto Estadísticas de YouTube

**Curso:** Procesamiento de Lenguaje Natural · **Semana 7** · **Etapa 1: datos**

Esta guía acompaña al equipo desde la creación de las cuentas hasta la entrega de la etapa de datos. Cada paso explica **qué se hace**, **por qué se hace** y **cómo saber que quedó bien**. Al final de cada parte hay lecturas para profundizar.

**Resultado esperado al terminar la guía**

- Un repositorio del equipo en GitHub con el código del proyecto.
- Los datos de Kaggle y de la YouTube Data API descargados, validados y guardados en Google Drive.
- Tres reportes generados automáticamente: calidad de datos, análisis estadísticos y baselines.
- Respuestas argumentadas a las preguntas del taller: ¿cómo se accede a los datos?, ¿son de calidad y suficientes?, ¿cómo mejorarlos?, ¿qué variable predecir?, ¿qué análisis no triviales se pueden extraer?

**Tiempo estimado:** 30 minutos de preparación de cuentas (Parte 1) y entre 2 y 3 horas para el resto.

**Contenido**

- Parte 0. Panorama general y conceptos
- Parte 1. Preparación de cuentas y credenciales
- Parte 2. Primera sesión en Google Colab
- Parte 3. Recorrido por el código
- Parte 4. Datos reales de Kaggle y reporte de calidad
- Parte 5. Ampliación de los datos con la YouTube Data API
- Parte 6. Variable objetivo, leakage y suficiencia de datos
- Parte 7. Interpretación de los análisis estadísticos
- Parte 8. Ejercicios para mejorar los datos
- Parte 9. Guardar el trabajo y colaborar en equipo
- Parte 10. Automatización con GitHub Actions (MLOps)
- Parte 11. Entregables y lista de verificación
- Anexos: errores frecuentes, glosario y referencias

---

## Parte 0. Panorama general y conceptos

### 0.1 Qué se va a construir

El proyecto busca predecir el desempeño de un video de YouTube (vistas, likes o engagement) a partir de lo que dicen sus comentarios. En esta primera etapa **no se entrena todavía el modelo final**: se construye la base de datos sobre la que se entrenará, se mide su calidad y se decide qué se puede predecir con ella.

El trabajo se organiza como un *pipeline*: una secuencia de etapas automáticas que siempre se ejecutan en el mismo orden y producen el mismo resultado a partir de los mismos datos.

| Etapa | Qué hace | Archivo |
| --- | --- | --- |
| 1. Ingesta | Descarga los datos de Kaggle o de la YouTube API | `src/ytnlp/data/kaggle_source.py`, `youtube_api.py` |
| 2. Validación | Revisa reglas (tipos, rangos, nulos) y mide la calidad | `src/ytnlp/data/validate.py` |
| 3. Limpieza | Normaliza el texto de los comentarios | `src/ytnlp/data/clean.py` |
| 4. Features | Construye una tabla con una fila por video | `src/ytnlp/features/build_features.py` |
| 5. Análisis | Ejecuta pruebas estadísticas no triviales | `src/ytnlp/analysis/stats.py` |
| 6. Baselines | Entrena modelos simples y la curva de aprendizaje | `src/ytnlp/models/baseline.py` |

### 0.2 El entorno de trabajo

Todo se trabaja desde **Google Colab**, sin instalar nada en el computador. Colab es un servicio de Google que ejecuta notebooks de Python en la nube. Su limitación principal es que **la sesión es temporal**: se desconecta tras un periodo de inactividad o después de varias horas de uso, y el disco se borra. Por eso cada parte del proyecto vive en un servicio diferente:

| Componente | Servicio | Por qué |
| --- | --- | --- |
| Cómputo | Google Colab | Gratuito, sin instalación, con GPU disponible para etapas posteriores |
| Código | GitHub | Guarda el historial de cambios y permite trabajar en equipo |
| Datos y reportes | Google Drive | Persisten aunque la sesión de Colab se cierre |
| Credenciales | Secretos de Colab | Evitan escribir contraseñas o claves dentro del código |

### 0.3 Conceptos clave

- **API (Application Programming Interface):** puerta de acceso que un servicio ofrece a los programas. La YouTube Data API permite pedir, por código, las estadísticas y los comentarios de un video.
- **Clave de API o token:** contraseña que identifica a quien hace la petición. Nunca se escribe en el código ni se sube a GitHub; si alguien la obtiene, puede gastar la cuota o acceder a la cuenta.
- **Cuota:** límite de peticiones diarias que un servicio permite por proyecto.
- **MLOps:** conjunto de prácticas para que un modelo de machine learning sea reproducible, se pueda desplegar y se pueda monitorear. En la etapa de datos implica versionar el código, documentar de dónde vienen los datos, validarlos automáticamente y medir su comportamiento en el tiempo.
- **Leakage (fuga de información):** error que ocurre cuando el modelo recibe como entrada información que no tendría en la vida real o que contiene la respuesta. Produce métricas excelentes en pruebas y un modelo inútil en producción.

**Lecturas para profundizar**

- Preguntas frecuentes de Google Colab (límites de uso y duración de las sesiones): https://research.google.com/colaboratory/faq.html
- MLOps: entrega continua y automatización de pipelines de machine learning (Google Cloud): https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Reglas de machine learning de Google (buenas prácticas de ingeniería): https://developers.google.com/machine-learning/guides/rules-of-ml

---

## Parte 1. Preparación de cuentas y credenciales

Esta parte se realiza una sola vez. Cada integrante necesita una cuenta de Google y una de GitHub; las credenciales de Kaggle y de YouTube pueden ser de un solo integrante y compartirse dentro del equipo por un canal privado.

### 1.1 Repositorio del equipo en GitHub

**Qué es:** un repositorio es una carpeta de proyecto con historial de versiones. GitHub lo aloja en la nube.

**Pasos**

1. Crear una cuenta en GitHub (https://github.com) si no se tiene.
2. Abrir el repositorio plantilla publicado por el docente y pulsar **Use this template → Create a new repository**.
3. En *Owner*, elegir la cuenta de un integrante; en *Repository name*, escribir `youtube-stats-nlp`.
4. Elegir **Private** (recomendado) y pulsar **Create repository**.
5. En el repositorio nuevo, ir a **Settings → Collaborators → Add people** e invitar al resto del equipo.

**Cómo verificar:** el repositorio muestra las carpetas `src`, `notebooks`, `docs` y el archivo `README.md`.

> Nota para el docente: para publicar la plantilla se sube el contenido del proyecto a un repositorio y se activa **Settings → General → Template repository**.

### 1.2 Token de GitHub

**Para qué sirve:** permite que Colab clone un repositorio privado y suba cambios (*push*) sin escribir la contraseña de GitHub.

**Pasos**

1. En GitHub: foto de perfil → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. *Token name:* `colab-youtube-stats`. *Expiration:* hasta el final del semestre.
3. *Repository access:* **Only select repositories** → elegir el repositorio del equipo.
4. *Permissions → Repository permissions → Contents:* **Read and write**.
5. Pulsar **Generate token** y copiar el valor (empieza por `github_pat_`). GitHub solo lo muestra una vez.

**Lectura:** Administración de tokens de acceso personal: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

### 1.3 Credenciales de Kaggle

**Para qué sirven:** descargar el dataset *YouTube Statistics* por código, sin descargarlo a mano.

**Pasos**

1. Crear una cuenta en Kaggle (https://www.kaggle.com).
2. Abrir el dataset https://www.kaggle.com/datasets/advaypatil/youtube-statistics y revisar la pestaña *Data*: contiene `videos-stats.csv` y `comments.csv`.
3. Ir a https://www.kaggle.com/settings/api y pulsar **Generate New Token**. Copiar el valor: será el secreto `KAGGLE_API_TOKEN`.
4. Alternativa: **Create Legacy API Key** descarga un archivo `kaggle.json` con `username` y `key`; esos valores se usan como `KAGGLE_USERNAME` y `KAGGLE_KEY`.

**Lectura:** Documentación de la librería kagglehub: https://github.com/Kaggle/kagglehub

### 1.4 Clave de la YouTube Data API v3

**Para qué sirve:** ampliar el dataset con datos actuales y con más comentarios por video.

**Pasos**

1. Entrar a Google Cloud Console (https://console.cloud.google.com) con una cuenta de Google.
2. Crear un proyecto nuevo: selector de proyectos (parte superior) → **Proyecto nuevo** → nombre `youtube-stats-nlp` → **Crear**.
3. Ir a **APIs y servicios → Biblioteca**, buscar **YouTube Data API v3** y pulsar **Habilitar**.
4. Ir a **APIs y servicios → Credenciales → Crear credenciales → Clave de API**. Copiar la clave.
5. Pulsar **Editar clave de API** y, en *Restricciones de API*, elegir **Restringir clave** → marcar solo **YouTube Data API v3** → **Guardar**. Así, si la clave se filtra, no sirve para otros servicios.

**Cómo funciona la cuota:** cada proyecto tiene 10.000 unidades diarias. Las consultas que usa el proyecto (`videos.list`, `channels.list`, `commentThreads.list`) cuestan 1 unidad cada una. Por ejemplo, extraer 50 videos con hasta 100 comentarios cada uno cuesta cerca de 52 unidades: 1 llamada para las estadísticas de los 50 videos, 1 para los canales y 50 para los comentarios. La cuota se reinicia a medianoche, hora del Pacífico.

**Lecturas**

- Introducción a la YouTube Data API: https://developers.google.com/youtube/v3/getting-started
- Costo en cuota de cada método: https://developers.google.com/youtube/v3/determine_quota_cost
- Método `videos.list`: https://developers.google.com/youtube/v3/docs/videos/list
- Método `commentThreads.list`: https://developers.google.com/youtube/v3/docs/commentThreads/list
- Administración de claves de API en Google Cloud: https://docs.cloud.google.com/docs/authentication/api-keys

### 1.5 Carpeta compartida en Google Drive

1. Un integrante crea en Google Drive la carpeta `ytnlp-proyecto` y la comparte con el equipo como **Editor**.
2. Cada integrante que la recibe la abre en *Compartido conmigo*, pulsa clic derecho → **Organizar → Agregar acceso directo** y la ubica en **Mi unidad**. Sin este paso, Colab no la encuentra en la ruta `MyDrive/ytnlp-proyecto`.

---

## Parte 2. Primera sesión en Google Colab

### 2.1 Abrir el notebook de inicio

1. Entrar a Google Colab (https://colab.research.google.com).
2. **Archivo → Abrir notebook → pestaña GitHub**.
3. Si el repositorio es privado, marcar **Incluir repositorios privados** y autorizar a Colab.
4. Pegar la URL del repositorio del equipo y elegir `notebooks/00_colab_inicio.ipynb`.
5. Para conservar los cambios del notebook: **Archivo → Guardar una copia en GitHub** (Parte 9).

**Tipo de entorno:** en esta etapa basta con CPU (**Entorno de ejecución → Cambiar tipo de entorno de ejecución → CPU**). La GPU se reserva para cuando se usen modelos de lenguaje grandes (Parte 8).

**Lectura:** Uso de Colab con GitHub: https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb

### 2.2 Crear los secretos

1. En la barra lateral izquierda de Colab, abrir el panel **Secretos** (icono de llave).
2. Pulsar **Agregar secreto nuevo** por cada credencial y escribir **exactamente** estos nombres:

| Nombre | Valor | Obligatorio |
| --- | --- | --- |
| `KAGGLE_API_TOKEN` | Token de Kaggle (paso 1.3) | Sí (o el par heredado) |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Credenciales heredadas de `kaggle.json` | Solo si no se usa el token nuevo |
| `YOUTUBE_API_KEY` | Clave de la YouTube API (paso 1.4) | Para la Parte 5 |
| `GITHUB_TOKEN` | Token de GitHub (paso 1.2) | Si el repositorio es privado o para hacer *push* |

3. Activar el interruptor **Acceso del notebook** en cada secreto.

**Por qué así:** los secretos quedan asociados a la cuenta de Google, no al notebook. Si el notebook se comparte o se sube a GitHub, las claves no viajan con él.

### 2.3 Celda 1: clonar el repositorio e instalar dependencias

En el campo **REPO_URL** del formulario de la celda se escribe la URL del repositorio del equipo (por ejemplo `https://github.com/usuario/youtube-stats-nlp.git`) y se ejecuta la celda (Ctrl + Enter).

**Qué hace, en orden**

1. Si existe el secreto `GITHUB_TOKEN`, lo usa para autenticarse sin guardarlo en disco.
2. Clona el repositorio en `/content/youtube-stats-nlp`. Si ya existe, actualiza con `git pull`.
3. Instala solo los paquetes que faltan en Colab (`requirements-colab.txt`: pandera, emoji, python-dotenv y una versión reciente de kagglehub). No reinstala pandas ni numpy, para no obligar a reiniciar el entorno.
4. Agrega la carpeta `src` al *path* de Python, para poder importar el paquete `ytnlp`.

**Resultado esperado:** `Repositorio listo en /content/youtube-stats-nlp`.

### 2.4 Celda 2: montar Drive y cargar secretos

Al ejecutarla, Colab pide permiso para acceder a Google Drive; se acepta con la cuenta que tiene la carpeta del paso 1.5.

**Resultado esperado (ejemplo)**

```
Entorno
  entorno           Google Colab
  python            3.12.x
  pandas            2.2.x
  gpu               no
  repo              /content/youtube-stats-nlp
  datos y reportes  /content/drive/MyDrive/ytnlp-proyecto
Secretos
  KAGGLE_API_TOKEN  OK
  KAGGLE_USERNAME   falta
  KAGGLE_KEY        falta
  YOUTUBE_API_KEY   OK
  GITHUB_TOKEN      OK
```

Que aparezca `falta` en `KAGGLE_USERNAME` y `KAGGLE_KEY` es normal si se usa el token nuevo. La línea **datos y reportes** debe apuntar a Drive; si apunta a `/content/...`, los resultados se perderán al cerrar la sesión.

**Lectura:** Acceso a archivos y a Google Drive desde Colab: https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/io.ipynb

### 2.5 Sección 3: verificar que todo funciona

La celda ejecuta las **pruebas automáticas** (`pytest`) y luego el pipeline completo sobre una **muestra sintética** incluida en el repositorio.

- Las pruebas comprueban, entre otras cosas, que la validación detecta problemas sembrados a propósito (duplicados, likes mayores que vistas, comentarios sin video) y que la tabla de features no tiene leakage. El resultado esperado es una línea como `8 passed`.
- La muestra sintética **no es información real**: sirve para confirmar que el entorno funciona sin necesitar credenciales. Sus resultados no se usan como conclusiones.

**Lectura:** Introducción a pytest: https://docs.pytest.org/en/stable/getting-started.html

**Cada vez que se vuelve a trabajar:** basta con ejecutar las celdas 1 y 2. El código se actualiza desde GitHub y los datos siguen en Drive.

---

## Parte 3. Recorrido por el código

Antes de usar datos reales conviene entender la organización del proyecto. En Colab, el panel **Archivos** (icono de carpeta) permite navegar por `/content/youtube-stats-nlp` y abrir cualquier archivo con doble clic.

```
configs/config.yaml          parámetros: rutas, variable objetivo, cuota, pruebas
src/ytnlp/
  config.py                  lectura de la configuración y de las rutas
  colab.py                   Drive, secretos y push a GitHub desde Colab
  pipeline.py                ejecuta las 6 etapas en orden
  data/kaggle_source.py      descarga y estandariza el dataset de Kaggle
  data/youtube_api.py        extracción con la YouTube Data API
  data/validate.py           reglas de validación y reporte de calidad
  data/clean.py              limpieza de texto
  features/build_features.py tabla de features por video
  analysis/stats.py          análisis estadísticos
  models/baseline.py         baselines y curva de aprendizaje
notebooks/                   00_colab_inicio, 01_eda
tests/                       pruebas automáticas
docs/                        esta guía y la guía del taller
```

**Decisiones de diseño que conviene conocer**

1. **Esquema canónico.** Kaggle usa columnas como `Video ID` o `Published At`; la API usa otros nombres. Ambas fuentes se convierten a los mismos nombres en `snake_case` (`video_id`, `published_at`), así el resto del código funciona igual con cualquier fuente.
2. **Dos niveles de reglas.** Las *reglas duras* (por ejemplo, sentimiento fuera de {0, 1, 2} o conteos negativos) eliminan la fila y la reportan. Las *reglas blandas* (por ejemplo, likes mayores que vistas) solo se miden, para que el equipo decida qué hacer.
3. **Los emojis se convierten en texto, no se borran.** el emoji de fuego pasa a `:fire:`, porque expresa sentimiento.
4. **No se imputan métricas con la media.** Si un video no tiene vistas, se excluye solo de los análisis que necesitan vistas. Imputar con la media inventa información y reduce la varianza real.
5. **Una fila por video.** Los comentarios se agregan por video (promedio de sentimiento, polarización, longitud media, entre otros). El objetivo se predice por video, nunca por comentario.

**Lecturas**

- Validación de dataframes con Pandera: https://pandera.readthedocs.io/en/stable/
- Librería emoji para Python: https://pypi.org/project/emoji/
- Capítulo de normalización de texto y tokenización, *Speech and Language Processing* (Jurafsky y Martin): https://web.stanford.edu/~jurafsky/slp3/

---

## Parte 4. Datos reales de Kaggle y reporte de calidad

### 4.1 Ejecutar con Kaggle

Se ejecuta la **sección 4** del notebook de inicio. La primera vez descarga el dataset a Drive (`data/raw/kaggle/`); las siguientes lo reutiliza.

### 4.2 Leer el reporte de calidad

En la **sección 5** se elige `quality_report.md` en el formulario y se ejecuta la celda. También puede abrirse desde Drive (`ytnlp-proyecto/reports/`). Para cada bloque del reporte, el equipo debe anotar el valor y responder la pregunta guía:

| Bloque | Qué mide | Pregunta guía para el informe |
| --- | --- | --- |
| Reglas duras | Filas eliminadas y por qué regla | ¿Cuántas filas se perdieron? ¿Afecta a algún tipo de video en particular? |
| Volumen | Número de videos y comentarios por video | ¿Diez comentarios por video bastan para describir la reacción de la audiencia? |
| Completitud | Porcentaje de nulos por columna | ¿Qué columnas tienen nulos y qué se hace con ellas? |
| Unicidad | Videos y comentarios duplicados | ¿Los duplicados son errores o spam? |
| Consistencia | Likes > vistas, fechas futuras, vistas en cero | ¿Qué explicación tiene cada inconsistencia? |
| Integridad referencial | Comentarios cuyo video no existe | ¿Se descartan o se buscan los videos faltantes? |
| Ruido de texto | Comentarios vacíos, muy cortos, con URL, solo emojis o con escritura no latina | ¿Qué porcentaje de comentarios aporta poco texto? |
| Balance | Videos por keyword, distribución del sentimiento | ¿Alguna categoría o clase está subrepresentada? |
| Representatividad | Concentración de vistas en los videos más vistos | ¿El dataset representa a YouTube o solo a videos populares? |
| Vigencia | Edad de los videos | ¿Los datos siguen siendo actuales? |
| Monitoreo | Métricas que se guardan en cada ejecución | ¿Cambian de una extracción a otra? |

El archivo `monitoring_history.csv` acumula una fila por ejecución. Si en una extracción nueva el sentimiento medio o el vocabulario cambian mucho, los datos se están desplazando (*data drift*) y el modelo podría degradarse.

### 4.3 Notebook de EDA

Se abre `notebooks/01_eda.ipynb` (mismo procedimiento del paso 2.1), se ejecutan las celdas 1 y 2 y se elige `SOURCE = "kaggle"`. El notebook genera el **diccionario de datos** (tipo, porcentaje de nulos, valores únicos y un ejemplo por columna) y las distribuciones de vistas, engagement y sentimiento. Al final hay un espacio para las conclusiones del equipo.

**Lecturas**

- Dimensiones de calidad de datos (completitud, unicidad, consistencia, validez): https://pandera.readthedocs.io/en/stable/dataframe_schemas.html
- *Data drift* y monitoreo de datos (Google Cloud, sección de monitoreo del documento de MLOps): https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

---

## Parte 5. Ampliación de los datos con la YouTube Data API

### 5.1 Por qué ampliar

El dataset de Kaggle tiene pocos comentarios por video, es una foto única en el tiempo y no incluye variables importantes como la duración del video o los suscriptores del canal. La API permite corregir las tres cosas.

### 5.2 Ejecutar la extracción

1. En la **sección 6** del notebook, poner `LIMITE_VIDEOS = 50` para la primera prueba.
2. Ejecutar la celda. El extractor toma los Video IDs del dataset de Kaggle y, por cada video, descarga estadísticas actualizadas, duración, categoría, tags, suscriptores del canal y hasta 100 comentarios.
3. La extracción se guarda en Drive en `data/raw/api/AAAA-MM-DD/` (`videos.parquet` y `comments.parquet`), con la fecha y hora en la columna `extracted_at`.
4. A continuación se ejecuta el pipeline con esos datos (`run("api")`).

**Comportamientos que conviene conocer**

- Si un video tiene los comentarios desactivados o fue borrado, se omite sin detener el proceso.
- Si la cuota se agota, se guardan los comentarios obtenidos hasta ese momento y se puede continuar al día siguiente.
- Los datos de la API no traen etiqueta de sentimiento; el proyecto usa un léxico mínimo como respaldo. Reemplazarlo por un modelo preentrenado es uno de los ejercicios de la Parte 8.

### 5.3 Comparar fuentes

Con ambos reportes de calidad (Kaggle y API), el equipo compara comentarios por video, porcentaje de ruido y vocabulario. Esta comparación es la evidencia para la pregunta **¿cómo podemos mejorar los datos?**

### 5.4 Series de tiempo

Ejecutar la extracción en varios días sobre los mismos videos permite medir cómo crecen las vistas. Esto habilita una variable objetivo más realista: el crecimiento en 7 días. La Parte 10 explica cómo automatizarlo.

**Lecturas**

- Referencia del recurso `commentThreads`: https://developers.google.com/youtube/v3/docs/commentThreads
- Formato Parquet y por qué se prefiere a CSV para datos tabulares: https://parquet.apache.org/docs/overview/

---

## Parte 6. Variable objetivo, leakage y suficiencia de datos

### 6.1 Elegir la variable objetivo

La variable se configura en `configs/config.yaml`, en la línea `target`. En Colab, el archivo se abre con doble clic desde el panel **Archivos**, se edita y se guarda con Ctrl + S.

| Valor de `target` | Qué predice | Cuándo conviene |
| --- | --- | --- |
| `engagement` (predeterminado) | log de (likes + comentarios) / vistas | Es independiente del tamaño del canal y es lo más relacionado con lo que dicen los comentarios |
| `log_views` | log de las vistas | Es lo que pide el enunciado, pero depende sobre todo de la audiencia del canal |
| `log_likes` | log de los likes | Muy correlacionado con las vistas |
| `perf_class` | Clase bajo, medio o alto (por cuantiles de engagement) | Es más robusto y más fácil de explicar a un usuario final |

**Por qué se usa logaritmo:** las vistas se distribuyen con cola pesada (pocos videos concentran la mayoría de las vistas). Sin logaritmo, el error del modelo lo dominan unos pocos videos virales.

### 6.2 Evitar el leakage

Las columnas `likes`, `comments` y `views` del video **nunca** se usan como features: contienen la respuesta. El código lo verifica con una aserción y con una prueba automática. Además, la división entre entrenamiento y prueba se hace **por video**: si comentarios del mismo video quedaran en ambos lados, el modelo memorizaría el video.

### 6.3 Leer el reporte de baselines

`baseline_report.md` compara cuatro modelos sencillos:

1. **dummy:** predice siempre el promedio. Es el piso mínimo.
2. **contexto (sin texto):** keyword, día y hora de publicación, edad del video y características del título.
3. **contexto + features de comentarios:** agrega sentimiento, polarización, longitud y diversidad léxica.
4. **contexto + features + TF-IDF:** agrega las palabras de los comentarios.

**Cómo interpretarlo:** si el modelo 3 o el 4 superan claramente al modelo 2, los comentarios aportan información sobre el desempeño, lo cual es la hipótesis central del proyecto. Si no lo superan, eso también es un hallazgo que debe reportarse.

### 6.4 Curva de aprendizaje: ¿los datos son suficientes?

El reporte entrena el mejor modelo con el 10, 25, 50, 75 y 100 % de los datos de entrenamiento y mide el resultado en el mismo conjunto de prueba.

- **Si la métrica de prueba sigue subiendo al final**, conseguir más videos mejorará el modelo.
- **Si la curva se aplana**, más filas ayudan poco; conviene invertir en mejores features o mejores etiquetas.
- **Si la diferencia entre entrenamiento y prueba es grande**, el modelo se sobreajusta.

**Lecturas**

- Curvas de validación y de aprendizaje (scikit-learn): https://scikit-learn.org/stable/modules/learning_curve.html
- Fuga de datos en machine learning (scikit-learn, *Common pitfalls*): https://scikit-learn.org/stable/common_pitfalls.html
- TF-IDF y modelos lineales para texto (scikit-learn): https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

---

## Parte 7. Interpretación de los análisis estadísticos

`stats_report.md` contiene nueve análisis. Cada uno se presenta con la pregunta, la prueba usada, el resultado y una conclusión automática que el equipo debe **revisar y reescribir con sus palabras**. El informe debe incluir al menos tres, con su interpretación.

**Regla general:** un valor p pequeño indica que el efecto probablemente no se debe al azar, pero no dice si el efecto es grande. Por eso cada análisis reporta también un **tamaño de efecto** o un **intervalo de confianza**.

| # | Análisis | Idea | Cómo leer el resultado | Para profundizar |
| --- | --- | --- | --- | --- |
| 1 | Cola pesada en vistas | Comprobar si las vistas siguen una ley de potencias o una log-normal | `pct_vistas_en_top10pct` alto confirma concentración. La prueba de Vuong indica cuál distribución ajusta mejor; con p > 0,05 no se distinguen | Clauset, Shalizi y Newman (2009), *Power-law distributions in empirical data*: https://arxiv.org/abs/0706.1062 |
| 2 | Sentimiento vs. engagement | Correlación de rangos de Spearman con intervalo por bootstrap | Si el intervalo no incluye 0, hay asociación. Valores absolutos de ρ cercanos a 0,1 son débiles, a 0,3 moderados y a 0,5 fuertes | Correlación de Spearman (SciPy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html |
| 3 | Polarización vs. engagement | Igual que el 2, pero con la entropía del sentimiento (opiniones divididas) | Una ρ negativa indica que los videos polémicos tienen menos engagement relativo | Bootstrap (SciPy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html |
| 4 | Engagement por keyword | Kruskal-Wallis y comparaciones por pares de Dunn con corrección de Bonferroni | ε² mide el tamaño del efecto (0,01 pequeño, 0,08 moderado, 0,26 grande). Los pares significativos indican qué categorías difieren | Kruskal-Wallis (SciPy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html · Pruebas post hoc: https://scikit-posthocs.readthedocs.io/en/latest/ |
| 5 | Regresión con controles | ¿El sentimiento explica el engagement una vez se controla la keyword y la edad del video? | Los coeficientes estandarizados se comparan entre sí. Es asociación, no causalidad | Errores estándar robustos (statsmodels): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html |
| 6 | Leyes de Zipf y Heaps | Comprobar si el vocabulario se comporta como lenguaje natural | Pendiente de Zipf cercana a −1 y β de Heaps entre 0,4 y 0,6 son típicos. Desviaciones grandes indican ruido o varios idiomas | Jurafsky y Martin, capítulo 2: https://web.stanford.edu/~jurafsky/slp3/ |
| 7 | Palabras distintivas | Log-odds con prior de Dirichlet entre videos de desempeño alto y bajo | Palabras con \|z\| > 1,96 distinguen significativamente los grupos | Monroe, Colaresi y Quinn (2008), *Fightin' Words*: https://doi.org/10.1093/pan/mpn018 |
| 8 | Comentario más votado vs. promedio | Wilcoxon pareado por video | Si difiere, la audiencia premia con likes un tono distinto al promedio | Wilcoxon (SciPy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html |
| 9 | Información mutua | Cuánta información aporta cada feature sobre la clase de desempeño | Features con `p_perm` < 0,05 aportan información más allá del azar | Información mutua (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html |

**Advertencia de comparaciones múltiples:** al hacer muchas pruebas, algunas salen significativas por azar. Con 20 pruebas a α = 0,05, se espera en promedio un falso positivo. Por eso el análisis 4 aplica la corrección de Bonferroni.

---

## Parte 8. Ejercicios para mejorar los datos

Cada equipo elige **al menos dos** ejercicios, los implementa y reporta el efecto en los reportes de calidad y de baselines (antes y después).

### Ejercicio A. Etiquetar el sentimiento con un modelo preentrenado

El dataset de Kaggle trae sentimiento, pero no se sabe cómo se calculó; la API no lo trae. Un modelo multilingüe entrenado con publicaciones de redes sociales da etiquetas más confiables. Conviene activar la GPU (**Entorno de ejecución → Cambiar tipo → T4 GPU**).

```python
from transformers import pipeline
import pandas as pd
from ytnlp.config import path

clasificador = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=0,          # GPU; usar -1 para CPU
    truncation=True,
)
comentarios = pd.read_parquet(path("interim") / "comments.parquet")
predicciones = clasificador(comentarios["comment_raw"].fillna("").tolist(), batch_size=64)
mapa = {"negative": 0, "neutral": 1, "positive": 2}
comentarios["sentiment_modelo"] = [mapa[p["label"].lower()] for p in predicciones]
```

Después se mide el acuerdo con la etiqueta original usando el coeficiente kappa de Cohen (`sklearn.metrics.cohen_kappa_score`) sobre una muestra de 100 comentarios revisada a mano por el equipo.

- Modelo: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
- Pipelines de Hugging Face: https://huggingface.co/docs/transformers/main_classes/pipelines
- Kappa de Cohen: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

### Ejercicio B. Más comentarios por video

Aumentar `api.max_comments_per_video` en `configs/config.yaml` (por ejemplo a 300) y repetir la extracción. Comparar la curva de aprendizaje y la varianza del sentimiento promedio por video.

### Ejercicio C. Filtrar por idioma

Detectar el idioma de cada comentario (por ejemplo con la librería `lingua-language-detector`) y medir cómo cambian la ley de Zipf y las palabras distintivas al trabajar solo con inglés o solo con español.

- Lingua: https://github.com/pemistahl/lingua-py

### Ejercicio D. Tópicos de los comentarios

Aplicar BERTopic sobre los comentarios y probar si algún tópico se asocia con el desempeño (análisis tipo 2 o 4 de la Parte 7).

- BERTopic: https://maartengr.github.io/BERTopic/

### Ejercicio E. Muestreo representativo

Los videos del dataset de Kaggle están agrupados por palabra clave (columna `keyword`), lo que sugiere que se obtuvieron con búsquedas; las búsquedas tienden a favorecer videos populares. Construir una muestra con videos de canales pequeños y comparar la distribución de vistas (análisis 1).

---

## Parte 9. Guardar el trabajo y colaborar en equipo

### 9.1 Qué se guarda dónde

| Qué | Dónde | Cómo |
| --- | --- | --- |
| Código (`src/`, `configs/`, `tests/`) | GitHub | Sección 7 del notebook de inicio (`git_push`) |
| Notebooks | GitHub | **Archivo → Guardar una copia en GitHub** |
| Datos y reportes | Google Drive | Automático (la carpeta configurada en la celda 2) |

Los datos **no** se suben a GitHub: están en `.gitignore`, pueden ser grandes y pueden contener información personal de los autores de los comentarios.

### 9.2 Subir cambios de código

1. En la sección 7, activar `HACER_PUSH`, escribir un mensaje que describa el cambio (por ejemplo, "Aumenta comentarios por video a 300") y el nombre y correo del integrante.
2. Ejecutar. El *commit* queda a nombre del integrante y el token no se guarda en disco.

### 9.3 Trabajo en equipo

- Cada integrante trabaja en una **rama** propia (por ejemplo `analisis-polarizacion`) y propone sus cambios con un **pull request**. Otro integrante lo revisa antes de unirlo a `main`.
- Dos personas no deben editar el mismo notebook al mismo tiempo: Git no combina bien los cambios en archivos `.ipynb`. Cada uno puede hacer una copia con su nombre.
- La CI (Parte 10) ejecuta las pruebas en cada pull request: si fallan, el cambio rompió algo.

**Lecturas**

- Pro Git en español (capítulos 2, 3 y 6): https://git-scm.com/book/es/v2
- Pull requests en GitHub: https://docs.github.com/en/pull-requests

---

## Parte 10. Automatización con GitHub Actions (MLOps)

El repositorio incluye dos flujos de trabajo en `.github/workflows/`.

| Archivo | Cuándo se ejecuta | Qué hace |
| --- | --- | --- |
| `ci.yml` | En cada *push* a `main` y en cada pull request | Revisa el estilo del código, ejecuta las pruebas y el pipeline sobre la muestra, y publica los reportes como artefacto |
| `extract.yml` | Todos los días a las 06:17 UTC y de forma manual | Extrae datos con la YouTube API y genera el reporte de calidad |

**Activar la extracción diaria**

1. En GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
2. Crear `YOUTUBE_API_KEY` y `KAGGLE_API_TOKEN` (o `KAGGLE_USERNAME` y `KAGGLE_KEY`) con los mismos valores usados en Colab.
3. En la pestaña **Actions**, elegir *Extracción diaria (YouTube Data API)* → **Run workflow** para probarla manualmente.
4. Al terminar, los datos y reportes quedan como **artefacto** descargable durante 30 días.

Esta extracción diaria es la base de las series de tiempo (paso 5.4) y del monitoreo de *drift* (paso 4.2).

**Opcional: versionar los datos con DVC.** El archivo `dvc.yaml` define el pipeline para que `dvc repro` lo ejecute solo cuando cambian los datos o el código, y permite guardar los datos en un bucket de la nube con su historial.

**Lecturas**

- GitHub Actions: https://docs.github.com/en/actions
- Uso de secretos en GitHub Actions: https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets
- DVC, primeros pasos: https://doc.dvc.org/start
- MLflow, seguimiento de experimentos (etapas siguientes): https://mlflow.org/docs/latest/

---

## Parte 11. Entregables y lista de verificación

**Entregables de la etapa**

1. Enlace al repositorio del equipo (con acceso para el docente).
2. Notebook `01_eda.ipynb` ejecutado con datos reales y con las conclusiones del equipo.
3. Reportes `quality_report.md`, `stats_report.md` y `baseline_report.md` generados con datos reales.
4. Informe corto (máximo 4 páginas) que responda las preguntas del taller, con la matriz de mejoras de `docs/taller_semana7.md` completa.

**Lista de verificación final**

- [ ] Las celdas 1 y 2 se ejecutan sin errores y los datos se guardan en Drive.
- [ ] Las pruebas automáticas pasan.
- [ ] El pipeline se ejecutó con Kaggle y con al menos una extracción de la API.
- [ ] El diccionario de datos está documentado.
- [ ] La variable objetivo está justificada y no hay leakage.
- [ ] La curva de aprendizaje está interpretada (¿faltan datos?).
- [ ] Hay al menos tres análisis estadísticos interpretados con tamaño de efecto.
- [ ] Hay al menos dos mejoras implementadas y medidas (antes y después).
- [ ] Ninguna clave aparece en el código, en los notebooks ni en el historial de Git.
- [ ] La CI está en verde en GitHub.

---

## Anexo A. Errores frecuentes y soluciones

| Mensaje o síntoma | Causa probable | Solución |
| --- | --- | --- |
| `Se debe escribir en REPO_URL...` | No se escribió la URL en el formulario de la celda 1 | Escribir la URL del repositorio del equipo |
| `git clone falló: ... Repository not found` | URL incorrecta o repositorio privado sin `GITHUB_TOKEN` | Revisar la URL; crear el secreto y activar *Acceso del notebook* |
| La celda 2 muestra `falta` en todos los secretos | El acceso del notebook está desactivado o el nombre no coincide | Activar el interruptor y revisar mayúsculas y guiones bajos |
| `datos y reportes` apunta a `/content/...` | Drive no se montó | Volver a ejecutar la celda 2 y aceptar el permiso |
| `ModuleNotFoundError: No module named 'ytnlp'` | El entorno se reinició | Ejecutar de nuevo las celdas 1 y 2 |
| `Faltan los secretos de Kaggle` | No existe `KAGGLE_API_TOKEN` ni el par heredado | Crear el secreto (paso 1.3) |
| Error 401 o 403 de Kaggle | El token es inválido, expiró o se copió con espacios | Generar un token nuevo en kaggle.com/settings/api y actualizar el secreto |
| `accessNotConfigured` o `API key not valid` | La YouTube Data API v3 no está habilitada o la clave tiene otra restricción | Repetir el paso 1.4 |
| `Cuota diaria agotada` | Se superaron 10.000 unidades | Esperar al reinicio diario o reducir `LIMITE_VIDEOS` |
| `git push falló` | El token no tiene permiso *Contents: Read and write* o expiró | Generar un token nuevo (paso 1.2) |
| La sesión se desconectó | Inactividad o límite de horas | Reconectar y ejecutar las celdas 1 y 2; los datos siguen en Drive |

## Anexo B. Glosario

- **Artefacto (GitHub Actions):** archivo generado por un flujo de trabajo y guardado para descargarlo.
- **Baseline:** modelo simple que sirve como punto de comparación.
- **Bootstrap:** técnica que remuestrea los datos muchas veces para estimar la incertidumbre de un estadístico.
- **CI (integración continua):** ejecución automática de pruebas cada vez que cambia el código.
- **Data drift:** cambio en la distribución de los datos a lo largo del tiempo.
- **Engagement:** proporción de interacciones (likes y comentarios) por cada vista.
- **Feature:** variable de entrada del modelo.
- **Hapax:** palabra que aparece una sola vez en el corpus.
- **Parquet:** formato de archivo columnar, comprimido y con tipos de datos, más eficiente que CSV.
- **Pipeline:** secuencia de etapas automáticas y reproducibles.
- **Polarización:** grado en que las opiniones sobre un video están divididas; aquí se mide con la entropía de la distribución de sentimiento.
- **TF-IDF:** representación del texto que pondera cada palabra por su frecuencia en el documento y su rareza en el corpus.

## Anexo C. Referencias

**Plataformas**

- Google Colab, preguntas frecuentes: https://research.google.com/colaboratory/faq.html
- Colab y GitHub: https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb
- Colab, archivos y Google Drive: https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/io.ipynb
- Dataset YouTube Statistics: https://www.kaggle.com/datasets/advaypatil/youtube-statistics
- kagglehub: https://github.com/Kaggle/kagglehub
- YouTube Data API, introducción: https://developers.google.com/youtube/v3/getting-started
- YouTube Data API, costos de cuota: https://developers.google.com/youtube/v3/determine_quota_cost
- Claves de API en Google Cloud: https://docs.cloud.google.com/docs/authentication/api-keys
- Tokens de acceso personal de GitHub: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
- GitHub Actions: https://docs.github.com/en/actions
- Pro Git en español: https://git-scm.com/book/es/v2

**Datos, validación y MLOps**

- Pandera: https://pandera.readthedocs.io/en/stable/
- DVC: https://doc.dvc.org/start
- MLflow: https://mlflow.org/docs/latest/
- MLOps en Google Cloud: https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Reglas de machine learning: https://developers.google.com/machine-learning/guides/rules-of-ml

**PLN y estadística**

- Jurafsky y Martin, *Speech and Language Processing* (3.ª ed., borrador en línea): https://web.stanford.edu/~jurafsky/slp3/
- Clauset, Shalizi y Newman (2009), *Power-law distributions in empirical data*: https://arxiv.org/abs/0706.1062
- Monroe, Colaresi y Quinn (2008), *Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict*: https://doi.org/10.1093/pan/mpn018
- SciPy, estadística: https://docs.scipy.org/doc/scipy/reference/stats.html
- scikit-learn, curvas de aprendizaje: https://scikit-learn.org/stable/modules/learning_curve.html
- scikit-posthocs: https://scikit-posthocs.readthedocs.io/en/latest/
- Modelo de sentimiento multilingüe: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
- BERTopic: https://maartengr.github.io/BERTopic/
