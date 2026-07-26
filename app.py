"""Interfaz web mínima para consultar Qwen3 mediante Ollama."""

from __future__ import annotations

import os

import streamlit as st
from ollama import Client, ResponseError


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SYSTEM_PROMPT = (
    "Eres un asistente técnico especializado en ciberseguridad. "
    "Responde de forma clara, precisa y estructurada. "
    "Cuando no tengas información suficiente, indícalo expresamente."
)


def nanoseconds_to_seconds(value: int | None) -> float:
    """Convierte nanosegundos a segundos."""

    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_tokens_per_second(
    token_count: int | None,
    duration_ns: int | None,
) -> float:
    """Calcula la velocidad aproximada de generación."""

    if not token_count or not duration_ns:
        return 0.0

    duration_seconds = nanoseconds_to_seconds(duration_ns)

    if duration_seconds == 0:
        return 0.0

    return token_count / duration_seconds


def query_ollama(prompt: str) -> dict[str, object]:
    """Envía una consulta a Ollama y devuelve respuesta y métricas."""

    client = Client(host=OLLAMA_HOST)

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.2,
        },
        think=False,
    )

    output_tokens = response.eval_count or 0
    generation_duration = response.eval_duration or 0

    return {
        "answer": response.message.content or "",
        "model": response.model or OLLAMA_MODEL,
        "input_tokens": response.prompt_eval_count or 0,
        "output_tokens": output_tokens,
        "total_duration": nanoseconds_to_seconds(response.total_duration),
        "load_duration": nanoseconds_to_seconds(response.load_duration),
        "generation_duration": nanoseconds_to_seconds(generation_duration),
        "tokens_per_second": calculate_tokens_per_second(
            output_tokens,
            generation_duration,
        ),
        "done_reason": response.done_reason or "unknown",
    }


st.set_page_config(
    page_title="Qwen3 Ollama Test",
    page_icon="🤖",
    layout="centered",
)

st.title("Qwen3 + Ollama")
st.caption("Prueba local de integración mediante Python, uv y Streamlit")

st.info(
    f"Modelo: `{OLLAMA_MODEL}` · Servidor: `{OLLAMA_HOST}`"
)

question = st.text_area(
    "Pregunta",
    placeholder="Escribe una consulta para Qwen3...",
    height=150,
)

ask_button = st.button(
    "Preguntar",
    type="primary",
    use_container_width=True,
)

if ask_button:
    clean_question = question.strip()

    if not clean_question:
        st.warning("Escribe una pregunta antes de continuar.")

    else:
        try:
            with st.spinner("Consultando Qwen3..."):
                result = query_ollama(clean_question)

            st.subheader("Respuesta")

            st.text_area(
                "Respuesta del modelo",
                value=str(result["answer"]),
                height=250,
                disabled=True,
                label_visibility="collapsed",
            )

            st.subheader("Métricas")

            column_1, column_2, column_3 = st.columns(3)

            column_1.metric(
                "Tokens de entrada",
                int(result["input_tokens"]),
            )

            column_2.metric(
                "Tokens de salida",
                int(result["output_tokens"]),
            )

            column_3.metric(
                "Tokens por segundo",
                f'{float(result["tokens_per_second"]):.2f}',
            )

            column_4, column_5, column_6 = st.columns(3)

            column_4.metric(
                "Tiempo total",
                f'{float(result["total_duration"]):.2f} s',
            )

            column_5.metric(
                "Carga del modelo",
                f'{float(result["load_duration"]):.2f} s',
            )

            column_6.metric(
                "Generación",
                f'{float(result["generation_duration"]):.2f} s',
            )

            with st.expander("Información técnica"):
                st.json(
                    {
                        "model": result["model"],
                        "ollama_host": OLLAMA_HOST,
                        "done_reason": result["done_reason"],
                        "temperature": 0.2,
                        "thinking_enabled": False,
                    }
                )

        except ConnectionError:
            st.error(
                "No se puede conectar con Ollama en "
                f"{OLLAMA_HOST}. Comprueba que Ollama está iniciado."
            )

        except ResponseError as error:
            st.error(f"Ollama devolvió un error: {error.error}")

        except Exception as error:
            st.exception(error)