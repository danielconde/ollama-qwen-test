"""Prueba mínima de conexión entre Python, uv y Ollama."""

from __future__ import annotations

import sys

from ollama import ResponseError, chat


MODEL = "qwen3:8b"


def main() -> int:
    """Envía una consulta de prueba al modelo local."""

    print(f"Conectando con Ollama mediante el modelo: {MODEL}\n")

    try:
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente técnico especializado en "
                        "ciberseguridad. Responde de forma clara y breve."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Confirma que la conexión funciona y explica "
                        "en una frase qué es un SOC."
                    ),
                },
            ],
            options={
                "temperature": 0.2,
            },
            think=False,
        )

        answer = response.message.content

        if not answer:
            print("Ollama devolvió una respuesta vacía.", file=sys.stderr)
            return 1

        print("Respuesta de Qwen3:\n")
        print(answer)

        return 0

    except ResponseError as error:
        print(
            f"Error devuelto por Ollama: {error.error}",
            file=sys.stderr,
        )
        return 1

    except ConnectionError:
        print(
            "No se puede conectar con Ollama en localhost:11434.",
            file=sys.stderr,
        )
        print(
            "Comprueba que Ollama está iniciado.",
            file=sys.stderr,
        )
        return 1

    except Exception as error:
        print(
            f"Error inesperado: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())