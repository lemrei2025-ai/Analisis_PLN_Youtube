# Taller Semana 7 – Proyecto Estadísticas de YouTube

Curso de Procesamiento de Lenguaje Natural. El proyecto busca un modelo que prediga el desempeño de un video de YouTube a partir de sus comentarios. Esta primera etapa se concentra en los datos: acceso, calidad, suficiencia, mejoras, variable objetivo y análisis estadísticos no triviales.

**El taller se desarrolla completamente en Google Colab. Los estudiantes no necesitan cuenta ni repositorio en GitHub.**

| Documento | Para quién | Contenido |
| --- | --- | --- |
| [Guía paso a paso](docs/guia_paso_a_paso.md) | Estudiantes | Explicación de cada paso, salidas esperadas y lecturas de consulta |
| [Preguntas del taller](docs/taller_semana7.md) | Estudiantes | Relación entre las preguntas del taller y los reportes; matriz de mejoras |
| Este README | Docente | Cómo publicar el taller y cómo está organizado |

## Enlaces para los estudiantes

- Notebook de inicio: https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/00_colab_inicio.ipynb
- Notebook de EDA: https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/01_eda.ipynb
- Guía paso a paso: https://github.com/USUARIO/REPOSITORIO/blob/main/docs/guia_paso_a_paso.md

## Cómo funciona

| Componente | Dónde vive | Quién lo modifica |
| --- | --- | --- |
| Código del taller (`src/`, `configs/`, notebooks, guías) | Este repositorio público en GitHub | Solo el docente |
| Copia del notebook de cada equipo | Google Drive del equipo (*Archivo → Guardar una copia en Drive*) | El equipo |
| Datos, reportes y configuración del equipo | Carpeta `MyDrive/ytnlp-proyecto` en Google Drive | El equipo, desde Colab |
| Credenciales (Kaggle, YouTube API) | Secretos de Colab de cada estudiante | Cada estudiante |
| Cómputo | Google Colab (CPU basta en esta etapa) | — |

En cada sesión, la celda 1 del notebook descarga el código del taller en modo solo lectura (`git clone` de este repositorio público, sin credenciales). La celda 2 monta Google Drive y carga los secretos. Todo lo que produce el equipo se escribe en Drive, por lo que se conserva aunque Colab cierre la sesión. Las decisiones del equipo, como la variable objetivo o los comentarios por video, se guardan en `config_equipo.yaml` dentro de Drive, sin tocar el código.

## Publicación del taller (docente)

1. **Crear el repositorio.** En GitHub: *New repository* → nombre (por ejemplo `taller-youtube-nlp`) → visibilidad **Public** → sin README ni `.gitignore` → *Create repository*. Debe ser público para que Colab descargue el código sin credenciales.
2. **Configurar la dirección del repositorio** en los notebooks y las guías (reemplaza el marcador `USUARIO/REPOSITORIO`):

   ```bash
   python scripts/configurar_repo.py usuario-github/taller-youtube-nlp
   ```

3. **Subir los archivos.** Desde la carpeta del proyecto:

   ```bash
   git init
   git add .
   git commit -m "Taller Semana 7: Proyecto Estadísticas de YouTube"
   git branch -M main
   git remote add origin https://github.com/usuario-github/taller-youtube-nlp.git
   git push -u origin main
   ```

   Alternativa sin consola: en la página del repositorio vacío, *uploading an existing file* y arrastrar el contenido de la carpeta. En ese caso se verifica que también suban los archivos ocultos `.gitignore` y `.github/workflows/ci.yml` (algunos sistemas los ocultan al arrastrar).

4. **Probar como estudiante.** Abrir el enlace del notebook de inicio, guardar una copia en Drive y ejecutar las celdas 1 a 3. Las pruebas deben terminar con `passed` y el pipeline de muestra debe generar los reportes en Drive.
5. **Compartir con los estudiantes** los tres enlaces de la sección *Enlaces para los estudiantes*.

Al publicar una corrección, basta con subir los cambios al repositorio: los estudiantes la reciben la próxima vez que ejecuten la celda 1.

La integración continua (`.github/workflows/ci.yml`) ejecuta las pruebas y el pipeline sobre la muestra en cada cambio que el docente sube; no requiere secretos.

## Estructura

```
.
├── configs/config.yaml          # configuración base: rutas, variable objetivo, parámetros
├── data/sample/                 # muestra sintética para pruebas (no son datos reales)
├── src/ytnlp/
│   ├── config.py                # configuración base + cambios del equipo en Drive
│   ├── colab.py                 # Drive, secretos y resumen del entorno en Colab
│   ├── pipeline.py              # orquesta todas las etapas
│   ├── data/
│   │   ├── kaggle_source.py     # descarga (kagglehub) y normaliza el dataset de Kaggle
│   │   ├── youtube_api.py       # extracción con YouTube Data API v3
│   │   ├── validate.py          # esquemas Pandera + reporte de calidad
│   │   └── clean.py             # limpieza de texto y de métricas
│   ├── features/build_features.py
│   ├── analysis/stats.py        # análisis estadísticos no triviales
│   └── models/baseline.py       # baselines + curva de aprendizaje
├── notebooks/
│   ├── 00_colab_inicio.ipynb    # punto de entrada en Google Colab
│   └── 01_eda.ipynb             # diccionario de datos, calidad y distribuciones
├── scripts/
│   ├── make_sample_data.py      # genera la muestra sintética
│   └── configurar_repo.py       # configura la dirección del repositorio (docente)
├── tests/
├── docs/
│   ├── guia_paso_a_paso.md      # guía para estudiantes
│   └── taller_semana7.md        # preguntas del taller y matriz de mejoras
├── requirements-colab.txt       # solo lo que Colab no trae preinstalado
├── requirements.txt             # entorno local / CI
└── .github/workflows/ci.yml     # pruebas automáticas del repositorio
```

## Reportes que genera el pipeline

| Archivo (en `reports/`) | Contenido |
| --- | --- |
| `quality_report.md` | Nulos, duplicados, reglas violadas, integridad referencial, ruido de texto, balance, representatividad |
| `stats_report.md` | Cola pesada, Spearman con bootstrap, Kruskal-Wallis + Dunn, regresión con controles, Zipf/Heaps, log-odds con prior de Dirichlet, Wilcoxon, información mutua |
| `baseline_report.md` | Baseline sin texto vs. con texto, curva de aprendizaje |
| `monitoring_history.csv` | Una fila por ejecución: volumen, nulos, sentimiento medio, vocabulario (detección de *drift*) |
| `figures/` | Gráficos de apoyo |

## Datos

**Kaggle – YouTube Statistics** ([`advaypatil/youtube-statistics`](https://www.kaggle.com/datasets/advaypatil/youtube-statistics)): `videos-stats.csv` (título, Video ID, fecha, keyword, likes, comments, views) y `comments.csv` (Video ID, comentario, likes, sentimiento 0/1/2). Las columnas se normalizan a `snake_case` en `kaggle_source.py`.

**YouTube Data API v3**: `videos.list` (estadísticas, duración, categoría, tags), `commentThreads.list` (comentarios) y `channels.list` (suscriptores). Cada una de estas llamadas cuesta 1 unidad de cuota y la cuota diaria por defecto es 10.000 unidades. `search.list` tiene además un límite propio de 100 llamadas diarias, por eso el extractor parte de una lista de Video IDs y no de búsquedas ([costos de cuota](https://developers.google.com/youtube/v3/determine_quota_cost)).

Cada extracción se guarda en `data/raw/api/<AAAA-MM-DD>/` en Parquet con la fecha de extracción, lo que permite construir series de tiempo y medir *drift*.

## Variable objetivo y leakage

Por defecto la variable objetivo es `engagement = (likes + comments) / views`, porque las vistas absolutas dependen sobre todo del tamaño del canal. Cada equipo puede cambiarla desde el notebook (sección 4). Las features **nunca** incluyen `likes`, `comments` ni `views` del video, y la división entre entrenamiento y prueba es por video.

## Prácticas de MLOps en esta etapa

- Código versionado en GitHub y ejecutado siempre en su última versión publicada.
- Datos crudos guardados por fecha de extracción, sin sobrescribir extracciones anteriores.
- Validación automática con reglas explícitas (Pandera) en cada ejecución.
- Métricas de monitoreo acumuladas en `monitoring_history.csv`.
- Credenciales en los Secretos de Colab, nunca en el código ni en los notebooks.
- Pruebas automáticas en cada cambio del repositorio (CI).

## Uso en un computador local (opcional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
make sample && make pipeline SOURCE=sample     # muestra sintética
cp .env.example .env                           # completar KAGGLE_API_TOKEN y YOUTUBE_API_KEY
make pipeline SOURCE=kaggle
make extract && make pipeline SOURCE=api
```
