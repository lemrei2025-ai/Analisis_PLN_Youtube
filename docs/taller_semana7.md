# Taller Semana 7 – Proyecto Estadísticas de YouTube (etapa de datos)

Relación entre las preguntas del taller y lo que produce el notebook. El enunciado completo del taller (objetivos, agenda y rúbrica) es el documento entregado por el docente.

## Entorno de trabajo

Todo el taller se hace en **Google Colab**, abriendo el enlace del notebook de inicio entregado por el docente y guardando una copia en Google Drive. No se necesita cuenta de GitHub. Los datos, los reportes y la configuración del equipo quedan en Google Drive y las credenciales en los Secretos de Colab. El paso a paso completo, con explicaciones y lecturas, está en [`guia_paso_a_paso.md`](guia_paso_a_paso.md).

Lista de verificación del equipo antes de la sesión:

- [ ] Cuenta de Google para cada integrante.
- [ ] Cuenta de Kaggle con token de API (`KAGGLE_API_TOKEN`).
- [ ] Proyecto en Google Cloud con la YouTube Data API v3 habilitada y una clave de API (`YOUTUBE_API_KEY`).
- [ ] Secretos creados en Colab con *Acceso del notebook* activado.
- [ ] Carpeta `ytnlp-proyecto` compartida en Google Drive para el equipo.
- [ ] Copia del notebook de inicio guardada en Drive y celdas 1 a 3 ejecutadas sin errores.

## Preguntas del taller

| Pregunta del taller | Dónde se responde | Sección del notebook de inicio |
| --- | --- | --- |
| ¿Cómo se accede a los datos? | Kaggle con `kagglehub` y YouTube Data API v3 (`kaggle_source.py`, `youtube_api.py`) | 5 y 7 |
| ¿Son de calidad? | `reports/quality_report.md` | 5 y 6 |
| ¿Son suficientes? | Curva de aprendizaje en `reports/baseline_report.md` | 5 y 6 |
| ¿Cómo mejorarlos? | Extracción con la API (más comentarios, canal, duración, series de tiempo) y ejercicios de la Parte 8 de la guía | 4 y 7 |
| ¿Qué variables predecir? | Campo `TARGET` de la configuración del equipo | 4 |
| ¿Qué análisis no triviales? | `reports/stats_report.md` | 5 y 6 |

## Actividades y entregables

| Actividad | Dónde queda |
| --- | --- |
| 2.1 Diccionario de datos | Copia del notebook `01_eda` en Drive |
| 2.2 Extracción con la API | `ytnlp-proyecto/data/raw/api/` en Drive |
| 3.1 Reporte de calidad | `ytnlp-proyecto/reports/quality_report.md` (+ `monitoring_history.csv`) |
| 4.1 Matriz impacto × esfuerzo | Tabla de abajo, en el informe |
| 5.1 Variable objetivo + baseline | `ytnlp-proyecto/reports/baseline_report.md` |

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
