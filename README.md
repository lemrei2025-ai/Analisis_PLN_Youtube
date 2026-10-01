# Taller Semana 7 – Proyecto Estadísticas de YouTube

**Curso:** Procesamiento de Lenguaje Natural · **Etapa 1 del proyecto: los datos**

El proyecto busca construir un modelo de PLN que prediga el desempeño de un video de YouTube (vistas, likes o engagement) a partir de sus comentarios. En esta primera etapa no se entrena todavía el modelo final: el trabajo se concentra en los datos. Al terminar el taller, cada equipo debe poder responder con evidencia:

1. ¿Cómo se accede a los datos?
2. ¿Son de buena calidad y son suficientes para un buen resultado?
3. ¿Cómo se pueden mejorar?
4. ¿Qué variable conviene predecir (vistas, likes, engagement)?
5. ¿Qué análisis estadísticos no triviales se pueden extraer?

**Todo el taller se hace en Google Colab, desde el navegador.** No hay que instalar nada en el computador ni crear una cuenta o un repositorio en GitHub. Cada equipo trabaja sobre una copia del notebook guardada en su propio Google Drive.

| Documento | Para qué sirve |
| --- | --- |
| Este README | Paso a paso resumido, de principio a fin |
| [Guía paso a paso](docs/guia_paso_a_paso.md) | Explicación detallada de cada paso, cómo interpretar los resultados, ejercicios y lecturas de consulta |
| [Preguntas del taller](docs/taller_semana7.md) | Relación entre cada pregunta del taller y el reporte que la responde; matriz de mejoras para el informe |

---

## Antes de empezar: qué se necesita

| Requisito | Quién | Tiempo aproximado |
| --- | --- | --- |
| Cuenta de Google (Gmail sirve) | Cada integrante | — |
| Cuenta de Kaggle y su token de API | Al menos un integrante | 5 minutos |
| Proyecto en Google Cloud con la YouTube Data API v3 y una clave de API | Al menos un integrante | 10 minutos |
| Carpeta compartida del equipo en Google Drive | Un integrante la crea y la comparte | 5 minutos |

Las credenciales de Kaggle y de YouTube pueden ser de un solo integrante y compartirse con el equipo por un canal privado. Nunca se escriben dentro de un notebook ni se publican.

**Enlaces del taller**

- Notebook de inicio: https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/00_colab_inicio.ipynb
- Notebook de EDA: https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/01_eda.ipynb

---

## Paso a paso para el estudiante

### Paso 1. Obtener el token de Kaggle

El token permite que el notebook descargue el dataset *YouTube Statistics* sin hacerlo a mano.

1. Crear una cuenta en https://www.kaggle.com (o iniciar sesión).
2. Abrir el dataset https://www.kaggle.com/datasets/advaypatil/youtube-statistics y revisar la pestaña *Data*: tiene los archivos `videos-stats.csv` y `comments.csv`.
3. Ir a https://www.kaggle.com/settings/api y pulsar **Generate New Token**.
4. Copiar el valor que aparece y guardarlo en un lugar seguro. Se usará en el paso 5 con el nombre `KAGGLE_API_TOKEN`.

### Paso 2. Obtener la clave de la YouTube Data API

La clave permite ampliar el dataset con datos actuales y con más comentarios por video.

1. Entrar a https://console.cloud.google.com con la cuenta de Google.
2. En el selector de proyectos (parte superior), elegir **Proyecto nuevo**, escribir `youtube-stats-nlp` y pulsar **Crear**.
3. Ir a **APIs y servicios → Biblioteca**, buscar **YouTube Data API v3** y pulsar **Habilitar**.
4. Ir a **APIs y servicios → Credenciales → Crear credenciales → Clave de API** y copiar la clave.
5. Pulsar **Editar clave de API**, en *Restricciones de API* elegir **Restringir clave**, marcar solo **YouTube Data API v3** y **Guardar**.

La clave se usará en el paso 5 con el nombre `YOUTUBE_API_KEY`. Cada proyecto tiene 10.000 unidades de cuota diarias; extraer 50 videos con sus comentarios consume alrededor de 52 unidades.

### Paso 3. Preparar la carpeta del equipo en Google Drive

Colab borra todo lo que hay en la sesión al desconectarse. Por eso los datos y los reportes se guardan en Drive.

1. Un integrante crea en Google Drive, dentro de **Mi unidad**, la carpeta `ytnlp-proyecto` y la comparte con el resto del equipo como **Editor**.
2. Cada integrante que la recibe la busca en **Compartido conmigo**, hace clic derecho → **Organizar → Agregar acceso directo** y la ubica en **Mi unidad**. Sin este paso, Colab no la encuentra.

### Paso 4. Abrir el notebook de inicio y guardar una copia

1. Abrir el enlace del **notebook de inicio** (sección *Enlaces del taller*). Se abre directamente en Google Colab.
2. Iniciar sesión con la cuenta de Google si Colab lo solicita.
3. **Antes de ejecutar nada**, ir a **Archivo → Guardar una copia en Drive**. Se abre una pestaña nueva con la copia.
4. Trabajar siempre en esa copia. Conviene renombrarla con el nombre del equipo (clic en el título, arriba a la izquierda) y moverla a la carpeta `ytnlp-proyecto` desde Google Drive.

El notebook abierto desde el enlace pertenece al taller y no guarda cambios; la copia es del equipo y es la que se entrega.

### Paso 5. Registrar las credenciales en los Secretos de Colab

1. En la barra lateral izquierda de Colab, abrir el panel **Secretos** (icono de llave).
2. Pulsar **Agregar secreto nuevo** y crear estos dos, escribiendo los nombres exactamente así:

   | Nombre | Valor |
   | --- | --- |
   | `KAGGLE_API_TOKEN` | Token del paso 1 |
   | `YOUTUBE_API_KEY` | Clave del paso 2 |

3. Activar el interruptor **Acceso del notebook** en cada secreto.

Los secretos quedan asociados a la cuenta de Google, no al notebook: si el notebook se comparte, las claves no viajan con él. Cada integrante que ejecute el notebook debe registrar los secretos en su propia cuenta.

### Paso 6. Ejecutar la celda 1: descargar el código del taller

Pulsar el botón de reproducir de la celda 1 (o Ctrl + Enter). No hay campos que llenar. La celda descarga el código del taller e instala las librerías que Colab no trae.

**Resultado esperado:** `Código del taller listo en /content/youtube-stats-nlp`

### Paso 7. Ejecutar la celda 2: conectar Drive y cargar los secretos

Al ejecutarla, Colab pide permiso para acceder a Google Drive; se acepta con la misma cuenta del paso 3.

**Resultado esperado (los valores pueden variar):**

```
Entorno
  entorno           Google Colab
  datos y reportes  /content/drive/MyDrive/ytnlp-proyecto
Secretos
  KAGGLE_API_TOKEN  OK
  KAGGLE_USERNAME   falta
  KAGGLE_KEY        falta
  YOUTUBE_API_KEY   OK
Configuración
  origen            base del taller
  target            engagement
  comentarios/video 100
```

Hay que revisar dos cosas: que **datos y reportes** apunte a `/content/drive/...` y que `KAGGLE_API_TOKEN` y `YOUTUBE_API_KEY` digan `OK`. Que `KAGGLE_USERNAME` y `KAGGLE_KEY` digan `falta` es normal.

### Paso 8. Verificar el entorno con la muestra (sección 3)

Ejecutar la celda de la sección 3. Corre las pruebas automáticas y el pipeline completo sobre una muestra **sintética** incluida en el taller.

**Resultado esperado:** una línea como `10 passed` y, al final, `Listo. Reportes en /content/drive/MyDrive/ytnlp-proyecto/reports`.

La muestra no son datos reales: solo confirma que todo funciona. Sus resultados no se usan como conclusiones.

### Paso 9. Definir la configuración del equipo (sección 4)

En el formulario de la sección 4 se eligen:

- `TARGET`: la variable que se va a predecir. Opciones: `engagement` (recomendada para empezar), `log_views`, `log_likes` o `perf_class`. La Parte 6 de la [guía](docs/guia_paso_a_paso.md) explica cuándo conviene cada una.
- `COMENTARIOS_POR_VIDEO`: cuántos comentarios descargar por video en el paso 11.

Ejecutar la celda. La configuración se guarda en Drive (`ytnlp-proyecto/config_equipo.yaml`) y se aplica en todas las sesiones y notebooks del equipo. Se puede cambiar en cualquier momento y volver a ejecutar los pasos siguientes.

### Paso 10. Procesar los datos reales de Kaggle y leer los reportes (secciones 5 y 6)

1. Ejecutar la sección 5. La primera vez descarga el dataset de Kaggle a Drive; después lo reutiliza. Luego valida, limpia, construye las features, corre los análisis estadísticos y entrena los modelos de referencia.
2. En la sección 6, elegir un reporte en el formulario y ejecutar la celda para verlo:

   | Reporte | Qué contiene | Pregunta del taller que responde |
   | --- | --- | --- |
   | `quality_report.md` | Nulos, duplicados, inconsistencias, ruido de texto, balance, representatividad | ¿Los datos son de calidad? |
   | `baseline_report.md` | Modelos sin texto vs. con comentarios y curva de aprendizaje | ¿Los comentarios aportan? ¿Los datos son suficientes? |
   | `stats_report.md` | Nueve análisis estadísticos con pregunta, prueba, resultado y conclusión | ¿Qué análisis no triviales se pueden extraer? |

Los reportes también se pueden abrir desde Google Drive, en `ytnlp-proyecto/reports/`.

### Paso 11. Ampliar los datos con la YouTube Data API (sección 7)

1. En el formulario de la sección 7, dejar `LIMITE_VIDEOS = 50` para la primera prueba.
2. Ejecutar la celda. Por cada video descarga estadísticas actualizadas, duración, categoría, suscriptores del canal y comentarios, y vuelve a correr el pipeline con esos datos.
3. La extracción queda guardada en Drive en `data/raw/api/AAAA-MM-DD/`.

Comparar el reporte de calidad de Kaggle con el de la API es la evidencia para responder **¿cómo se pueden mejorar los datos?** Ejecutar esta sección en días distintos con los mismos videos construye una serie de tiempo y alimenta el historial de monitoreo (`reports/monitoring_history.csv`).

### Paso 12. Explorar los datos con el notebook de EDA

1. Abrir el enlace del **notebook de EDA** y guardar una copia en Drive (igual que en el paso 4).
2. Ejecutar las celdas 1 y 2.
3. En el formulario, cambiar `SOURCE` a `kaggle` (o `api`) y ejecutar el resto de celdas.
4. El notebook genera el **diccionario de datos** (tipo, porcentaje de nulos, valores únicos y un ejemplo por columna) y las distribuciones de vistas, engagement y sentimiento.
5. Escribir las conclusiones del equipo en la última celda.

### Paso 13. Interpretar los resultados y mejorar los datos

1. Leer cada reporte con la ayuda de las Partes 4, 6 y 7 de la [guía](docs/guia_paso_a_paso.md), que explican cómo leer cada métrica y cada prueba estadística.
2. Elegir al menos tres análisis de `stats_report.md` y reescribir sus conclusiones con palabras del equipo, incluyendo el tamaño del efecto.
3. Implementar al menos dos de los ejercicios de mejora de la Parte 8 de la guía (por ejemplo, etiquetar el sentimiento con un modelo preentrenado o aumentar los comentarios por video) y comparar los reportes antes y después.
4. Completar la matriz de mejoras de [Preguntas del taller](docs/taller_semana7.md).

El código adicional se escribe en celdas nuevas de la copia del notebook, después de ejecutar las celdas 1 y 2.

### Paso 14. Entregar

1. Compartir con el docente, como *Lector* o *Comentador*, la copia del notebook de inicio y la del notebook de EDA, ambas ejecutadas con datos reales.
2. Compartir con el docente la carpeta `ytnlp-proyecto` de Drive, con los reportes generados.
3. Entregar un informe corto (máximo 4 páginas) que responda las cinco preguntas del taller con la evidencia de los reportes, e incluya la matriz de mejoras.

**Lista de verificación antes de entregar**

- [ ] La celda 2 muestra que los datos y reportes se guardan en Drive.
- [ ] El pipeline se ejecutó con Kaggle y con al menos una extracción de la API.
- [ ] El diccionario de datos está documentado en el notebook de EDA.
- [ ] La variable objetivo está justificada y se explica cómo se evita el leakage.
- [ ] La curva de aprendizaje está interpretada: ¿faltan datos o no?
- [ ] Hay al menos tres análisis estadísticos interpretados con tamaño de efecto.
- [ ] Hay al menos dos mejoras implementadas y medidas (antes y después).
- [ ] Ninguna clave aparece escrita en los notebooks.
- [ ] Los notebooks y la carpeta de Drive están compartidos con el docente.

---

## Al volver a trabajar otro día

1. Abrir **la copia del notebook guardada en Drive** (no el enlace original del taller).
2. Ejecutar las celdas 1 y 2.
3. Continuar desde la sección que se necesite. Los datos, los reportes y la configuración del equipo siguen en Drive; no hace falta repetir los pasos anteriores.

## Problemas frecuentes

| Mensaje o síntoma | Solución |
| --- | --- |
| Los cambios del notebook no se guardan | Se está trabajando en el notebook del enlace. Usar *Archivo → Guardar una copia en Drive* y continuar en la copia |
| La celda 2 muestra `falta` en `KAGGLE_API_TOKEN` o `YOUTUBE_API_KEY` | Revisar que el nombre del secreto esté escrito exactamente igual y que *Acceso del notebook* esté activado |
| `datos y reportes` apunta a `/content/...` | Drive no se montó: volver a ejecutar la celda 2 y aceptar el permiso |
| `ModuleNotFoundError: No module named 'ytnlp'` | El entorno se reinició: ejecutar de nuevo las celdas 1 y 2 |
| `Faltan los secretos de Kaggle` | Crear el secreto `KAGGLE_API_TOKEN` (pasos 1 y 5) |
| Error 401 o 403 de Kaggle | El token es inválido o se copió con espacios: generar uno nuevo y actualizar el secreto |
| `accessNotConfigured` o `API key not valid` | La YouTube Data API v3 no está habilitada en el proyecto o la clave tiene otra restricción (paso 2) |
| `Cuota diaria agotada` | Esperar al reinicio diario de la cuota o reducir `LIMITE_VIDEOS` |
| `El repositorio del taller no está configurado` | Avisar al docente |
| La sesión se desconectó | Reconectar y ejecutar las celdas 1 y 2; los datos siguen en Drive |

La [guía](docs/guia_paso_a_paso.md) tiene la lista completa de errores en el Anexo A.

## Para profundizar

| Tema | Dónde |
| --- | --- |
| Explicación detallada de cada paso | [Guía paso a paso](docs/guia_paso_a_paso.md), Partes 1 a 5 |
| Variable objetivo, leakage y curva de aprendizaje | Guía, Parte 6 |
| Cómo interpretar los nueve análisis estadísticos | Guía, Parte 7 |
| Ejercicios para mejorar los datos | Guía, Parte 8 |
| Monitoreo y prácticas de MLOps | Guía, Parte 10 |
| Lecturas y referencias | Guía, Anexo C |
| YouTube Data API | https://developers.google.com/youtube/v3/getting-started |
| Costos de cuota de la API | https://developers.google.com/youtube/v3/determine_quota_cost |
| Preguntas frecuentes de Google Colab | https://research.google.com/colaboratory/faq.html |

---

## Qué hace el código

El notebook ejecuta un *pipeline*: una secuencia de etapas automáticas que siempre corren en el mismo orden.

| Etapa | Qué hace | Archivo |
| --- | --- | --- |
| 1. Ingesta | Descarga los datos de Kaggle o de la YouTube API | `src/ytnlp/data/kaggle_source.py`, `youtube_api.py` |
| 2. Validación | Revisa reglas (tipos, rangos, nulos) y mide la calidad | `src/ytnlp/data/validate.py` |
| 3. Limpieza | Normaliza el texto de los comentarios | `src/ytnlp/data/clean.py` |
| 4. Features | Construye una tabla con una fila por video | `src/ytnlp/features/build_features.py` |
| 5. Análisis | Ejecuta pruebas estadísticas no triviales | `src/ytnlp/analysis/stats.py` |
| 6. Modelos de referencia | Compara modelos con y sin comentarios y traza la curva de aprendizaje | `src/ytnlp/models/baseline.py` |

El código se puede leer en este repositorio o, durante la sesión, en el panel **Archivos** de Colab (`/content/youtube-stats-nlp`). Se descarga de nuevo en cada sesión, así que no conviene modificarlo: las decisiones del equipo se toman en la sección 4 del notebook y los análisis propios se escriben en celdas nuevas.

```
.
├── notebooks/
│   ├── 00_colab_inicio.ipynb    # punto de entrada del taller
│   └── 01_eda.ipynb             # diccionario de datos, calidad y distribuciones
├── docs/
│   ├── guia_paso_a_paso.md      # guía detallada para estudiantes
│   └── taller_semana7.md        # preguntas del taller y matriz de mejoras
├── src/ytnlp/                   # código del pipeline
├── configs/config.yaml          # configuración base del taller
├── data/sample/                 # muestra sintética para pruebas (no son datos reales)
├── scripts/                     # generación de la muestra sintética
├── tests/                       # pruebas automáticas
└── requirements-colab.txt       # librerías que Colab no trae preinstaladas
```

**Datos utilizados**

- **Kaggle – YouTube Statistics** ([`advaypatil/youtube-statistics`](https://www.kaggle.com/datasets/advaypatil/youtube-statistics)): `videos-stats.csv` (título, Video ID, fecha, keyword, likes, comentarios, vistas) y `comments.csv` (Video ID, comentario, likes, sentimiento 0/1/2).
- **YouTube Data API v3**: `videos.list` (estadísticas, duración, categoría, tags), `commentThreads.list` (comentarios) y `channels.list` (suscriptores). Cada llamada cuesta 1 unidad de cuota de las 10.000 diarias.

