# Taller Semana 7 – Proyecto Estadísticas de YouTube (etapa de datos)

Guía de cómo cada pregunta del taller se resuelve en este repositorio. El enunciado completo del taller (objetivos, agenda y rúbrica) es el documento entregado por el docente.

## Entorno de trabajo

Todo el taller se hace en **Google Colab**, abriendo `notebooks/00_colab_inicio.ipynb` desde GitHub. El código vive en GitHub, los datos y reportes en Google Drive y las credenciales en los Secretos de Colab. El paso a paso completo, con explicaciones y lecturas, está en [`guia_paso_a_paso.md`](guia_paso_a_paso.md).

Checklist del equipo antes de la sesión:

- [ ] Repositorio del equipo creado a partir de la plantilla del curso.
- [ ] Cuenta de Kaggle con token de API (`KAGGLE_API_TOKEN`).
- [ ] Proyecto en Google Cloud con la YouTube Data API v3 habilitada y una API key (`YOUTUBE_API_KEY`).
- [ ] Secretos creados en Colab con *Acceso del notebook* activado.
- [ ] Carpeta compartida en Google Drive para el equipo.
- [ ] Celdas 1 a 3 de `00_colab_inicio` ejecutadas sin errores.

## Preguntas del taller
| Pregunta del taller | Dónde se responde | Comando |
| --- | --- | --- |
| ¿Cómo se accede a los datos? | `src/ytnlp/data/kaggle_source.py`, `src/ytnlp/data/youtube_api.py` | `make pipeline SOURCE=kaggle`, `make extract` |
| ¿Son de calidad? | `src/ytnlp/data/validate.py` → `reports/quality_report.md` | `make pipeline` |
| ¿Son suficientes? | Curva de aprendizaje → `reports/baseline_report.md` | `make pipeline` |
| ¿Cómo mejorarlos? | Extractor de la API (más comentarios, canal, duración, series de tiempo); limpieza en `clean.py` | `make extract` |
| ¿Qué variables predecir? | `configs/config.yaml` (`target`), `build_features.targets()` | editar `target` |
| ¿Qué análisis no triviales? | `src/ytnlp/analysis/stats.py` → `reports/stats_report.md` | `make pipeline` |

## Actividades y archivos

| Actividad | Entregable en el repo |
| --- | --- |
| 2.1 Diccionario de datos | Sección en `notebooks/01_eda.ipynb` |
| 2.2 Script de la API | `src/ytnlp/data/youtube_api.py` |
| 3.1 Reporte de calidad | `reports/quality_report.md` (+ `monitoring_history.csv`) |
| 4.1 Matriz impacto × esfuerzo | Completar la tabla de abajo |
| 5.1 Variable objetivo + baseline | `reports/baseline_report.md` |

## Matriz de mejoras (completar por el equipo)

| Mejora | Impacto (1–5) | Esfuerzo (1–5) | ¿Se implementa en S7? |
| --- | --- | --- | --- |
| 100+ comentarios por video con la API | | | |
| Suscriptores y duración como contexto | | | |
| Re-etiquetar sentimiento con un transformer multilingüe | | | |
| Series de tiempo (extracción diaria) | | | |
| Filtro de idioma y spam | | | |
| Muestreo de videos poco vistos | | | |

## Decisiones a justificar en el informe

1. Variable objetivo principal y por qué (por defecto: log-engagement).
2. Qué se hace con nulos (no se imputan métricas del video con la media).
3. Si la etiqueta de sentimiento de Kaggle se usa como etiqueta o solo como feature.
4. Qué tan representativa es la muestra (videos de tendencia vs. aleatorios).
5. Riesgos de leakage y cómo se evitaron.
