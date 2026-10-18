"""App de clasificación de frutas frescas/podridas con webcam (Gradio)."""
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# pyrefly: ignore [missing-import]
import gradio as gr

from PIL import Image

from model import CLASSES, DEFECTO, disponibles, etiquetas, obtener, resumen

MODELOS_OK = disponibles()
if not MODELOS_OK:
    raise SystemExit("No se encontró un modelo disponible. Verifica MODEL_VOLUME en la configuración de la app.")
POR_DEFECTO = DEFECTO if DEFECTO in MODELOS_OK else MODELOS_OK[0]
obtener(POR_DEFECTO)  # carga el modelo principal al iniciar (evita demora en el primer fotograma)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_COUNT = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NEW_DATA_DIR = Path(os.getenv("NEW_DATA_VOLUME", "/Volumes/workspace/default/datos_nuevos"))


def clasificar(img, clave, umbral):
    if img is None:
        return {}, "Esperando imagen…"
    t0 = time.perf_counter()
    p = obtener(clave).probs(img)
    ms = (time.perf_counter() - t0) * 1000
    return etiquetas(p), resumen(p, umbral, ms, clave)


def _file_path(upload):
    path = getattr(upload, "path", upload)
    return Path(path)


def guardar_aportes(files, clase):
    if not files:
        raise gr.Error("Selecciona al menos una imagen.")
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    if len(files) > MAX_UPLOAD_COUNT:
        raise gr.Error(f"Puedes subir como máximo {MAX_UPLOAD_COUNT} imágenes por envío.")
    if clase not in CLASSES:
        raise gr.Error("La clase seleccionada no está en config.json.")

    valid_files = []
    for upload in files:
        source = _file_path(upload)
        extension = source.suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            raise gr.Error(f"Formato no permitido: {source.name}. Usa JPG, JPEG o PNG.")
        try:
            size = source.stat().st_size
        except OSError as exc:
            raise gr.Error(f"No se pudo leer el archivo {source.name}: {exc}") from exc
        if size > MAX_UPLOAD_BYTES:
            raise gr.Error(f"{source.name} supera el límite de 10 MB.")
        try:
            content = source.read_bytes()
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise gr.Error(f"{source.name} no es una imagen válida: {exc}") from exc
        valid_files.append((extension, content))

    volume_str = os.getenv("NEW_DATA_VOLUME", "")
    if volume_str.startswith("/Volumes/"):
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            saved_paths = []
            for extension, content in valid_files:
                name = f"{uuid.uuid4().hex}{extension}"
                remote_path = f"{volume_str.rstrip('/')}/{clase}/{name}"
                w.files.upload(remote_path, io.BytesIO(content), overwrite=True)
                saved_paths.append(remote_path)
            return f"Se guardaron {len(saved_paths)} imágenes en `{volume_str}/{clase}`."
        except Exception as exc:
            raise gr.Error(f"Error al guardar en el Volume con Databricks SDK: {exc}") from exc

    destination_dir = NEW_DATA_DIR / clase
    destination_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for extension, content in valid_files:
        name = f"{uuid.uuid4().hex}{extension}"
        destination = destination_dir / name
        temporary = destination.with_name(f".{name}.tmp")
        try:
            with temporary.open("xb") as output:
                output.write(content)
            temporary.replace(destination)
        except OSError:
            if temporary.exists():
                temporary.unlink()
            raise
        saved_paths.append(destination)

    return f"Se guardaron {len(saved_paths)} imágenes en `{NEW_DATA_DIR / clase}`."


def solicitar_reentrenamiento():
    token = os.getenv("GITHUB_TOKEN")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if not token:
        raise gr.Error("Falta configurar el secreto GITHUB_TOKEN como recurso de la app.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise gr.Error("Configura GITHUB_REPOSITORY con el formato propietario/repositorio.")

    payload = json.dumps({"event_type": "datos-nuevos", "client_payload": {}}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/dispatches",
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 204:
                raise gr.Error(f"GitHub respondió con HTTP {response.status} al solicitar el workflow.")
    except urllib.error.HTTPError as exc:
        raise gr.Error(
            f"GitHub rechazó la solicitud (HTTP {exc.code}). Verifica el token y su permiso de contenido."
        ) from exc
    except urllib.error.URLError as exc:
        raise gr.Error(f"No se pudo contactar con GitHub: {exc.reason}") from exc
    return "Solicitud enviada. El workflow de reentrenamiento ya puede comenzar."


ACERCA = """
### Cómo funciona
1. La cámara captura un fotograma.
2. Se ajusta a **224×224 sin deformar** (letterbox) y se normaliza igual que en el entrenamiento.
3. El modelo calcula 8 puntajes y **softmax** los convierte en probabilidades.
4. Se muestra la clase más probable, separada en **fruta** y **estado** (fresca/podrida).
5. Si la confianza está bajo el umbral, la app dice **"No estoy seguro"**.

*How it works: the webcam frame is resized to 224×224 with padding, normalized, and a neural network outputs
a probability for each of the 8 classes (4 fruits × fresh/rotten).*

### Modelos entrenados (resultados en el conjunto de test)
| Modelo | Accuracy | Tamaño | ms/img CPU (Colab) |
|---|---|---|---|
| CNN desde cero | 95.6 % | 4.9 MB | 121 |
| **EfficientNet-B0 (transfer learning)** | **100 %** | 17.7 MB | **38** |
| ViT-B/16 | 100 % | 343 MB | 509 |

### Limitaciones
- El test viene de la misma fuente que el entrenamiento: con tu cámara el acierto puede ser menor.
- La app solo conoce 4 frutas. Con otros objetos igualmente responderá una de las 8 clases; el umbral ayuda, pero no lo evita.
"""

# ── CSS personalizado para mejorar el botón de la webcam ────────────────────
CUSTOM_CSS = """
/* Oculta el ícono de grabación circular rojo que aparece junto al texto */
.record-button svg { display: none !important; }

/* Estilos generales del botón de cámara */
button.record-button, button[aria-label*='record'], button[aria-label*='grabar'] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    cursor: pointer !important;
}
"""

# ── JavaScript: renombra el botón "Grabar" usando MutationObserver ────────────
CUSTOM_JS = """
function() {
    function patchBotones() {
        document.querySelectorAll('button').forEach(function(btn) {
            var txt = btn.innerText ? btn.innerText.trim() : '';
            if (txt === 'Grabar' || txt === 'Record') {
                btn.innerText = '\u25b6 Iniciar an\u00e1lisis';
                btn.style.background = '#6366f1';
                btn.style.color = 'white';
                btn.style.borderRadius = '8px';
                btn.style.fontWeight = '600';
                btn.style.padding = '6px 16px';
                btn.style.border = 'none';
                btn.style.cursor = 'pointer';
            } else if (txt === 'Stop' || txt === 'Detener') {
                btn.innerText = '\u23f9 Detener an\u00e1lisis';
                btn.style.background = '#ef4444';
                btn.style.color = 'white';
            }
        });
    }
    // Ejecutar al cargar y observar cambios en el DOM
    var observer = new MutationObserver(patchBotones);
    observer.observe(document.body, { childList: true, subtree: true });
    patchBotones();
}
"""

with gr.Blocks(title="Clasificador de frutas · fresca o podrida") as demo:
    gr.Markdown("# 🍎 Clasificador de frutas: ¿fresca o podrida?\n"
                "Apunta la fruta a la cámara · *Point a fruit at the camera*")
    with gr.Row():
        modelo = gr.Dropdown(MODELOS_OK, value=POR_DEFECTO, label="Modelo")
        umbral = gr.Slider(0.30, 0.95, value=0.60, step=0.05, label="Umbral de confianza")

    with gr.Tabs():
        # ── Pestaña 1: streaming continuo ────────────────────────────────────
        with gr.Tab("📷 Webcam en vivo"):
            gr.Markdown(
                "> 💡 **Cómo usar:** presiona **▶ Iniciar análisis** para encender la webcam "
                "y comenzar la clasificación automática. Presiona **⏹ Detener análisis** para pausarla."
            )
            with gr.Row():
                # streaming=True → Gradio envía frames continuamente
                # El botón que aparece en la UI es "Record" / "Grabar";
                # las instrucciones de arriba aclaran su función real.
                cam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="pil",
                    label="📡 Cámara en vivo  (presiona el botón de la cámara para iniciar/detener)",
                )
                with gr.Column():
                    lbl_v = gr.Label(num_top_classes=3, label="Top 3")
                    txt_v = gr.Markdown("*Esperando imagen de la cámara…*")
            cam.stream(
                clasificar,
                inputs=[cam, modelo, umbral],
                outputs=[lbl_v, txt_v],
                stream_every=0.7,
                time_limit=900,
                trigger_mode="always_last",
            )

        # ── Pestaña 2: captura de foto única ─────────────────────────────────
        with gr.Tab("🖼️ Subir o capturar foto"):
            gr.Markdown(
                "> **Cómo usar:** selecciona la fuente (📁 archivo, 📋 portapapeles o 📷 cámara). "
                "Con la cámara, presiona el **ícono de captura** (círculo) para tomar la foto "
                "y luego haz clic en **Clasificar**."
            )
            with gr.Row():
                # SIN streaming=True → el componente entra en modo snapshot:
                # el botón de la webcam muestra el ícono de captura de foto,
                # no el de grabación de video.
                img = gr.Image(
                    sources=["upload", "webcam", "clipboard"],
                    type="pil",
                    label="Imagen (sube un archivo, pega del portapapeles o captura con la cámara)",
                )
                with gr.Column():
                    lbl_f = gr.Label(num_top_classes=3, label="Top 3")
                    txt_f = gr.Markdown("*Captura o sube una imagen y presiona Clasificar.*")
            btn = gr.Button("🔍 Clasificar", variant="primary")
            btn.click(clasificar, [img, modelo, umbral], [lbl_f, txt_f], api_name="clasificar_foto")

        with gr.Tab("ℹ️ Acerca de"):
            gr.Markdown(ACERCA)

        with gr.Tab("📤 Aportar datos"):
            gr.Markdown(
                "Sube imágenes nuevas y asígnales su clase real. Pulsa **Guardar imágenes** y después "
                "**Reentrenar ahora**. Se guardan en Unity Catalog y no se incorporan a la validación fija."
            )
            aportes = gr.File(
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png"],
                type="filepath",
                label="Imágenes JPG, JPEG o PNG (máximo 20; 10 MB por imagen)",
            )
            clase_aporte = gr.Dropdown(CLASSES, value=CLASSES[0], label="Clase")
            guardar_btn = gr.Button("Guardar imágenes", variant="primary")
            estado_aporte = gr.Markdown()
            guardar_btn.click(
                guardar_aportes,
                inputs=[aportes, clase_aporte],
                outputs=estado_aporte,
                api_name="guardar_aportes",
            )
            reentrenar_btn = gr.Button("Reentrenar ahora", variant="secondary")
            estado_reentrenamiento = gr.Markdown()
            reentrenar_btn.click(
                solicitar_reentrenamiento,
                outputs=estado_reentrenamiento,
                api_name="solicitar_reentrenamiento",
            )

demo.queue(max_size=20)

if __name__ == "__main__":
    port_env = os.environ.get("DATABRICKS_APP_PORT") or os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
    port = int(port_env) if port_env else 7860
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=port,
        theme=gr.themes.Soft(),
        share=bool(os.getenv("GRADIO_SHARE")),
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
    )
