from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from uuid import uuid4
import os
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="DinoLab API")

OLLAMA_URL = "http://ollama:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
COLAB_API_URL = os.getenv("COLAB_API_URL", "").rstrip("/")
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Las tareas continúan aunque el navegador deje de esperar.
executor = ThreadPoolExecutor(max_workers=2)
jobs = {}
jobs_lock = Lock()


class NewDinoRequest(BaseModel):
    nombre: str | None = Field(default=None, max_length=60)


class ChatRequest(BaseModel):
    nombre: str = Field(min_length=1, max_length=60)
    identidad: str = Field(min_length=1, max_length=5000)
    mensaje: str = Field(min_length=1, max_length=2000)


def consultar_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        texto = response.json().get("response", "").strip()
        if not texto:
            raise HTTPException(
                status_code=502,
                detail="Ollama devolvió una identidad vacía."
            )
        return texto
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="No se pudo conectar con Ollama."
        ) from exc


def consultar_colab(endpoint: str, datos: dict, timeout: int = 300) -> dict:
    if not COLAB_API_URL:
        raise HTTPException(
            status_code=503,
            detail="La URL de la API de Colab no está configurada."
        )

    try:
        response = requests.post(
            f"{COLAB_API_URL}/{endpoint}",
            json=datos,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "No se pudo acceder a Colab. Comprueba que la sesión "
                "siga activa y que el túnel de ngrok continúe funcionando."
            )
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Colab devolvió una respuesta que no es JSON válido."
        ) from exc


def crear_dinosaurio(nombre_solicitado: str | None) -> dict:
    nombre = (nombre_solicitado or "").strip()

    if not nombre:
        resultado_nombre = consultar_colab(
            "generate-name",
            {
                "temp": 0.8,
                "top_k": 5,
                "top_p": 0.9,
                "max_len": 30
            },
            timeout=120
        )
        nombre = str(resultado_nombre.get("nombre", "")).strip()

        if not nombre:
            raise HTTPException(
                status_code=502,
                detail="Colab no devolvió un nombre válido."
            )

    prompt = f"""
Eres un creador de dinosaurios ficticios para una aplicación educativa.
Crea la identidad de un dinosaurio llamado {nombre}.
Responde en español con estos apartados:
Nombre, especie ficticia, personalidad, hábitat, alimentación,
habilidad especial y una breve descripción visual.
Sé creativo, coherente y apto para todas las edades.
No cambies el nombre indicado.
"""
    identidad = consultar_ollama(prompt)

    imagen = consultar_colab(
        "generate-image",
        {
            "nombre": nombre,
            "identidad": identidad
        },
        timeout=360
    )

    imagen_base64 = imagen.get("imagen_base64")
    if not imagen_base64:
        raise HTTPException(
            status_code=502,
            detail="Colab no devolvió la imagen esperada."
        )

    return {
        "nombre": nombre,
        "identidad": identidad,
        "formato": imagen.get("formato", "png"),
        "imagen_base64": imagen_base64
    }


def ejecutar_tarea(job_id: str, nombre: str | None):
    with jobs_lock:
        jobs[job_id]["estado"] = "generando"

    try:
        resultado = crear_dinosaurio(nombre)
        with jobs_lock:
            jobs[job_id]["estado"] = "completado"
            jobs[job_id]["resultado"] = resultado
    except Exception as exc:
        detalle = (
            exc.detail if isinstance(exc, HTTPException)
            else "Ocurrió un error durante la generación del dinosaurio."
        )
        with jobs_lock:
            jobs[job_id]["estado"] = "error"
            jobs[job_id]["detalle"] = detalle


@app.get("/")
def inicio():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def salud():
    return {
        "estado": "ok",
        "modelo": OLLAMA_MODEL,
        "colab_configurado": bool(COLAB_API_URL)
    }


@app.post("/api/new-dino")
def nuevo_dinosaurio(req: NewDinoRequest):
    job_id = str(uuid4())

    with jobs_lock:
        jobs[job_id] = {"estado": "en_cola"}

    executor.submit(ejecutar_tarea, job_id, req.nombre)

    return {"job_id": job_id, "estado": "en_cola"}


@app.get("/api/new-dino/{job_id}")
def estado_dinosaurio(job_id: str):
    with jobs_lock:
        tarea = jobs.get(job_id)
        if tarea is None:
            raise HTTPException(
                status_code=404,
                detail="No se encontró esta tarea. Intenta crear otro dinosaurio."
            )
        return dict(tarea)


@app.post("/api/chat")
def conversar(req: ChatRequest):
    prompt = f"""
Actúa como el dinosaurio ficticio llamado {req.nombre}.
Respeta siempre esta identidad y no contradigas sus características:
{req.identidad}

Responde en primera persona, en español, de manera amigable,
como si hablaras con un visitante de DinoLab. Mantén las respuestas
claras y breves. No afirmes que eres un dinosaurio real.

Visitante: {req.mensaje}
{req.nombre} responde:
"""
    respuesta = consultar_ollama(prompt)
    return {"respuesta": respuesta}
