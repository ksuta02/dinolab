# DinoLab 🦖

Aplicación educativa para generar dinosaurios ficticios mediante inteligencia artificial.

## Funcionalidades

- Generación de nombres de dinosaurios con un modelo recurrente entrenado en Google Colab.
- Generación de imágenes de dinosaurios.
- Chat con un asistente basado en Ollama y Llama 3.2.
- Interfaz web para interactuar con los dinosaurios generados.

## Arquitectura

1. **Interfaz web:** HTML, CSS y JavaScript.
2. **API:** Python y FastAPI.
3. **Generador de nombres e imágenes:** Google Colab mediante una API y ngrok.
4. **Asistente conversacional:** Ollama con Llama 3.2.
5. **Contenedores:** Docker.
6. **Acceso público:** Cloudflare Tunnel.
7. **Infraestructura:** Amazon SageMaker Notebook Instance.

## Parámetros de muestreo

Se compararon temperaturas de 0.7, 1.0 y 1.3, manteniendo `top_k = 5`, `top_p = 0.9` y `max_len = 30`.

## Tecnologías

Python, FastAPI, PyTorch, Google Colab, Ollama, Llama 3.2, Docker, Amazon SageMaker y Cloudflare Tunnel.
