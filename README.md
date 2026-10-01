# Estadísticas de YouTube con NLP

Proyecto 1 del curso de Procesamiento de Lenguaje Natural. El objetivo final es un modelo que prediga el desempeño de un video de YouTube a partir de sus comentarios, desplegado en la nube como API y con monitoreo.

**Etapa actual (Semana 7): datos.** El repositorio cubre acceso, validación, calidad, limpieza, features, análisis estadísticos no triviales y un baseline para responder:

- ¿Cómo se accede a los datos? → `src/ytnlp/data/`
- ¿Son de buena calidad y suficientes? → `reports/quality_report.md`, curva de aprendizaje
- ¿Cómo mejorarlos? → extracción por YouTube Data API v3 (más comentarios, canal, duración)
- ¿Qué variable predecir? → `engagement`, `log_views`, `perf_class` (configurable)
- ¿Qué análisis no triviales hay? → `reports/stats_report.md`

## Estructura

```
.
├── configs/config.yaml          # rutas, variable objetivo, parámetros
├── data/
│   ├── raw/                     # datos originales (versionados con DVC, no con Git)
│   ├── interim/                 # datos validados y limpios
│   ├── processed/               # tabla de features por video
│   └── sample/                  # muestra sintética pequeña para CI y pruebas
├── src/ytnlp/
│   ├── config.py
│   ├── colab.py                 # Drive, secretos y git push desde Colab
│   ├── pipeline.py              # orquesta todas las etapas
│   ├── data/
│   │   ├── kaggle_source.py     # descarga y normaliza el dataset de Kaggle
│   │   ├── youtube_api.py       # extracción con YouTube Data API v3
│   │   ├── validate.py          # esquemas Pandera + reporte de calidad
│   │   └── clean.py             # limpieza de texto y de métricas
│   ├── features/build_features.py
│   ├── analysis/stats.py        # análisis estadísticos no triviales
│   └── models/baseline.py       # baselines + curva de aprendizaje
├── scripts/make_sample_data.py
├── notebooks/
│   ├── 00_colab_inicio.ipynb    # punto de entrada en Google Colab
│   └── 01_eda.ipynb             # diccionario de datos, calidad y distribuciones
├── tests/
├── requirements.txt             # entorno local / CI
├── requirements-colab.txt       # solo lo que Colab no trae preinstalado
├── dvc.yaml                     # pipeline reproducible con DVC
├── .github/workflows/
│   ├── ci.yml                   # pruebas + pipeline sobre la muestra
│   └── extract.yml              # extracción diaria programada con la API
└── docs/taller_semana7.md
```

## Entorno de desarrollo: Google Colab + GitHub + Google Drive

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<usuario>/youtube-stats-nlp/blob/main/notebooks/00_colab_inicio.ipynb)

El proyecto se desarrolla en **Google Colab**: nadie necesita instalar nada en su computador. Como una sesión de Colab es temporal (el disco se borra al desconectarse), cada pieza vive donde no se pierde:

| Componente | Herramienta | Qué guarda o hace |
| --- | --- | --- |
| Cómputo y notebooks | Google Colab (CPU basta en esta etapa) | Ejecuta el pipeline, el EDA y los análisis |
| Código | GitHub | Versiona `src/`, configuración y notebooks; corre la CI y la extracción diaria (GitHub Actions) |
| Datos y reportes | Google Drive (`MyDrive/ytnlp-proyecto/`) | `data/raw`, `data/interim`, `data/processed`, `reports/` persisten entre sesiones |
| Credenciales | Secretos de Colab (🔑) y GitHub Secrets | `KAGGLE_USERNAME`, `KAGGLE_KEY`, `YOUTUBE_API_KEY`, `GITHUB_TOKEN`; nunca en el código |
| Despliegue (etapas siguientes) | Nube (ej. Cloud Run, Hugging Face Spaces) | API o interfaz del modelo y su monitoreo |

**Flujo de trabajo en Colab**

1. Subir este repositorio a GitHub (una vez) y reemplazar `<usuario>` en los enlaces y en la celda 1 de los notebooks.
2. Abrir `notebooks/00_colab_inicio.ipynb` con el botón *Abrir en Colab*.
3. Crear los secretos en el panel 🔑 de Colab y activar *Acceso del notebook*.
4. Ejecutar las celdas 1 y 2: clonan el repo en `/content`, instalan solo lo que falta (`requirements-colab.txt`), montan Drive y cargan los secretos.
5. Ejecutar el pipeline con la muestra, luego con Kaggle y luego con la API. Los reportes quedan en Drive.
6. Si cambió código, subirlo con la celda 7 (`git_push`) o con *Archivo → Guardar una copia en GitHub* para notebooks.

Al volver otro día solo se repiten las celdas 1 y 2. La variable de entorno `YTNLP_STORAGE` (la fija `ytnlp.colab.setup()`) indica dónde guardar datos y reportes; si no existe, se usan carpetas dentro del repositorio, por lo que todo también funciona en un computador local.

## Inicio rápido en un computador local (opcional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1) Probar todo con la muestra sintética (no requiere credenciales)
make sample
make pipeline SOURCE=sample

# 2) Con el dataset real de Kaggle
#    Requiere ~/.kaggle/kaggle.json (https://www.kaggle.com/settings -> API)
make pipeline SOURCE=kaggle

# 3) Ampliar con la YouTube Data API v3
cp .env.example .env            # y poner YOUTUBE_API_KEY
make extract                    # usa los Video IDs del dataset de Kaggle
make pipeline SOURCE=api
```

Los reportes quedan en `reports/`:

| Archivo | Contenido |
| --- | --- |
| `quality_report.md` | Nulos, duplicados, reglas violadas, integridad referencial, ruido de texto, balance |
| `stats_report.md` | Distribución de cola pesada, Spearman con bootstrap, Kruskal-Wallis + Dunn, Zipf/Heaps, log-odds con prior de Dirichlet, polarización, información mutua |
| `baseline_report.md` | Baseline sin texto vs. con texto, curva de aprendizaje |
| `figures/` | Gráficos de apoyo |

## Datos

**Kaggle – YouTube Statistics** (`advaypatil/youtube-statistics`): `videos-stats.csv` (título, Video ID, fecha, keyword, likes, comments, views) y `comments.csv` (Video ID, comentario, likes, sentimiento 0/1/2). Las columnas se normalizan a `snake_case` en `kaggle_source.py`.

**YouTube Data API v3**: `videos.list` (estadísticas, duración, categoría, tags), `commentThreads.list` (hasta N comentarios por video) y `channels.list` (suscriptores). Cada llamada `list` cuesta 1 unidad de cuota; `search.list` cuesta 100, por eso el extractor parte de una lista de Video IDs y no de búsquedas. La cuota diaria por defecto es 10.000 unidades (verificar en Google Cloud Console).

Cada extracción se guarda en `data/raw/api/<YYYY-MM-DD>/` en Parquet con la fecha de extracción, lo que permite construir series de tiempo y medir *drift*.

## Variable objetivo y leakage

Se configura en `configs/config.yaml` (`target`). Por defecto es `engagement = (likes + comments) / views`, porque las vistas absolutas dependen sobre todo del tamaño del canal. Las features **nunca** incluyen `likes`, `comments` ni `views` del video; el split es por video.

## MLOps

- Código en Git, datos con DVC (`dvc.yaml`) o un bucket con fecha en la ruta.
- Secretos en `.env` / GitHub Secrets (`YOUTUBE_API_KEY`), nunca en el repositorio.
- CI (`ci.yml`): lint + pruebas + pipeline completo sobre la muestra en cada push.
- Extracción programada (`extract.yml`): corre a diario y sube los Parquet como artefacto.
- El reporte de calidad incluye métricas de monitoreo (volumen, % nulos, sentimiento medio, tamaño de vocabulario) que se comparan entre extracciones.

## Equipo

| Nombre | Rol |
| --- | --- |
| _por definir_ | Datos / extracción |
| _por definir_ | Análisis estadístico |
| _por definir_ | MLOps / nube |
