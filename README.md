# 🍎 Clasificador de frutas: ¿fresca o podrida?

App web con **webcam en vivo** que clasifica 4 frutas (manzana, banana, naranja, fresa) como **frescas o podridas**.
Modelo desplegado: **EfficientNet-B0** (transfer learning). El ciclo automático congela su backbone y reentrena solo la cabeza.

## 1. Correrla en tu PC con VS Code

1. Abre esta carpeta en VS Code (`Archivo → Abrir carpeta`).
2. Abre la terminal (`Ctrl + ñ`) y crea el entorno:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Mac / Linux
   pip install -r requirements.txt
   ```
3. Copia `effnet_b0_tl.pt` desde Drive (`PTOYECTO1_VISION/models/`) a `models/`.
   Si existe `models/actual.pt`, se usa como modelo activo; de lo contrario, se usa el peso EfficientNet original.
4. Prueba el modelo sin interfaz: `python check_model.py una_foto.jpg`
5. Lanza la app: `python app.py` y abre http://127.0.0.1:7860 (el navegador pedirá permiso de cámara).

> El ViT tarda ~0.5 s por imagen en CPU; EfficientNet ~40 ms (tiempo real).
> La pestaña de aportes/reentrenamiento requiere el Volume y los secretos configurados para Databricks.

### Probar localmente el ciclo de mantenimiento

Con los datasets disponibles en carpetas por clase, ejecuta:

```bash
python train.py --train-dir ruta/al/dataset-original --new-data-dir ruta/a/datos-nuevos
python check_model.py --validation-dir ruta/a/validacion-fija --candidate models/candidato.pt
```

La validación debe estar separada de ambos directorios de entrenamiento. La evaluación guarda `metricas.json`;
si `f1_macro` no supera al modelo actual, termina con código distinto de cero y no se debe promover el candidato.

## 2. Enlace público temporal (sin cuenta, ideal para la exposición)

```bash
# Windows (PowerShell)
$env:GRADIO_SHARE="1"; python app.py
# Mac / Linux
GRADIO_SHARE=1 python app.py
```
Gradio imprime un enlace `https://xxxx.gradio.live` válido ~72 h, mientras tu PC siga encendida con la app corriendo.

## 3. Despliegue automático en Databricks Apps

El workflow `.github/workflows/desplegar-app.yml` despliega al hacer push a `main`.
Los pesos no forman parte del paquete de la app: Databricks Apps limita cada archivo de app a 10 MB,
así que la app carga `actual.pt` desde el Volume `workspace.default.modelos`.

### Preparación de Databricks

1. En Unity Catalog, usa el schema existente `workspace.default` y crea los Volumes `workspace.default.datos_nuevos` y `workspace.default.modelos`.
2. Configura en el bundle la app `clasificador-frutas`. El principal de servicio de la app recibe `WRITE_VOLUME`
   en `datos_nuevos`, `READ_VOLUME` en `modelos` y permiso de lectura al secreto del token de GitHub.
3. Crea un secret scope dedicado a esta app y guarda en él un token de GitHub con clave propia. No guardes el token en archivos del repo.
4. La identidad indicada por `DATABRICKS_TOKEN` en Actions también necesita desplegar la app y administrar los Volumes;
   para entrenar debe poder leer los Volumes de datos y escribir en `modelos`.

### Secrets y variables de GitHub

En **Settings → Secrets and variables → Actions** configura:

**Secrets**

- `DATABRICKS_HOST`: URL del workspace de Databricks.
- `DATABRICKS_TOKEN`: token de una identidad de despliegue con permisos de bundle y de los Volumes.

**Variables**

- `DATABRICKS_APP_SECRET_SCOPE`: scope que contiene el token de GitHub para la app.
- `DATABRICKS_APP_GITHUB_TOKEN_KEY`: nombre de la clave del token dentro de ese scope.
- `ORIGINAL_DATA_VOLUME`: ruta CLI del dataset original, por ejemplo `dbfs:/Volumes/workspace/default/<volume-original>`.
- `VALIDATION_DATA_VOLUME`: ruta CLI de la validación fija e independiente, por ejemplo `dbfs:/Volumes/workspace/default/<volume-validacion>`.

En **Settings → Actions → General**, permite que los workflows tengan permiso de escritura de contenidos;
la rama `main` también debe permitir a GitHub Actions publicar el commit del modelo validado.
El repositorio destino del `repository_dispatch` se obtiene automáticamente de `github.repository`.
El token de GitHub guardado en el scope debe ser un fine-grained token limitado a este repo con permiso
**Contents: write**, necesario para `repository_dispatch`.

### Requisitos de datos y CLI

El dataset original y la validación deben tener carpetas con los nombres exactos de `config.json`, por ejemplo:

```text
<raiz>/
  FreshApple/
  FreshBanana/
  FreshOrange/
  FreshStrawberry/
  RottenApple/
  RottenBanana/
  RottenOrange/
  RottenStrawberry/
```

Mantén la validación en un Volume distinto de `datos_nuevos`; nunca se copia al conjunto de entrenamiento.
Las imágenes aportadas desde la app se guardan en `/Volumes/workspace/default/datos_nuevos/<clase>/`.

El workflow `.github/workflows/reentrenar.yml` se ejecuta por `repository_dispatch` (`datos-nuevos`),
manualmente o cada lunes a las 09:00 UTC. Descarga los tres conjuntos con Databricks CLI, entrena en CPU,
compara ambos modelos con el mismo conjunto fijo y solo promueve si sube `f1_macro`. Las métricas quedan
como artefacto de Actions por 90 días. Si la evaluación falla o no mejora, no cambia el modelo activo ni despliega.
Cuando sí mejora, se versiona `models/actual.pt`, se actualiza el Volume y el mismo workflow despliega la app
(el push realizado con el token de Actions no dispara otro workflow).

El recurso Apps en bundles requiere Databricks CLI 0.239.0 o posterior; el campo `value_from` de la
configuración inline de Apps se incorporó en una versión posterior (el soporte se integró en el CLI el
16-ene-2026). El workflow instala la versión actual con `databricks/setup-cli@main`; si fijas una versión,
verifica que incluya ese campo. La disponibilidad de Databricks Apps, Unity Catalog Volumes y `bundle run`
depende de la región/workspace y del plan; verifica que estén habilitados en Free Edition.

## 4. Archivos

| Archivo | Para qué sirve |
|---|---|
| `app.py` | Interfaz Gradio: webcam, clasificación, aporte de datos y solicitud de reentrenamiento |
| `model.py` | Arquitecturas, preprocesamiento (letterbox 224×224 + normalización) y textos de resultado |
| `train.py` | Extracción de embeddings congelados y ajuste reproducible de la cabeza EfficientNet |
| `check_model.py` | Prueba individual o comparación de candidato/modelo activo en validación fija |
| `config.json` | Nombres de clase y estadísticas de normalización |
| `models/` | Pesos entrenados (`.pt`) |
| `databricks.yml`, `app.yaml` | Recurso de Databricks Apps y configuración de ejecución |
| `.github/workflows/` | Despliegue y ciclo automático de reentrenamiento |

## 5. Limitaciones

- Con fotos de otra cámara o iluminación el acierto puede ser menor que en el test.
- Solo conoce 4 frutas: ante otros objetos responderá igualmente una de las 8 clases. El umbral de confianza ayuda, pero no lo evita.
- El workflow semanal requiere que el dataset original y la validación estén disponibles en los Volumes configurados.
