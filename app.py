"""Chat local con Qwen3, Ollama y Streamlit."""

from __future__ import annotations

import os
from typing import TypedDict

import streamlit as st
from ollama import Client, ResponseError


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SYSTEM_PROMPT = (
    "Eres un asistente técnico especializado en ciberseguridad. "
    "Responde de forma clara, precisa y estructurada. "
    "Diferencia hechos, inferencias e hipótesis. "
    "Cuando no tengas información suficiente, indícalo expresamente."
)


class ChatMessage(TypedDict):
    """Mensaje almacenado en el historial del chat."""

    role: str
    content: str


class ResponseMetrics(TypedDict):
    """Métricas devueltas por Ollama."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_duration_seconds: float
    load_duration_seconds: float
    prompt_duration_seconds: float
    generation_duration_seconds: float
    tokens_per_second: float
    done_reason: str


def nanoseconds_to_seconds(value: int | None) -> float:
    """Convierte nanosegundos a segundos."""

    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_tokens_per_second(
    output_tokens: int,
    generation_duration_ns: int,
) -> float:
    """Calcula la velocidad aproximada de generación."""

    generation_seconds = nanoseconds_to_seconds(generation_duration_ns)

    if output_tokens == 0 or generation_seconds == 0:
        return 0.0

    return output_tokens / generation_seconds


def query_ollama(
    messages: list[ChatMessage],
    temperature: float,
) -> tuple[str, ResponseMetrics]:
    """Envía el historial a Ollama y devuelve respuesta y métricas."""

    client = Client(host=OLLAMA_HOST)

    ollama_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *messages,
    ]

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=ollama_messages,
        options={
            "temperature": temperature,
        },
        think=False,
    )

    answer = response.message.content or ""

    input_tokens = response.prompt_eval_count or 0
    output_tokens = response.eval_count or 0
    generation_duration_ns = response.eval_duration or 0

    metrics: ResponseMetrics = {
        "model": response.model or OLLAMA_MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_duration_seconds": nanoseconds_to_seconds(
            response.total_duration
        ),
        "load_duration_seconds": nanoseconds_to_seconds(
            response.load_duration
        ),
        "prompt_duration_seconds": nanoseconds_to_seconds(
            response.prompt_eval_duration
        ),
        "generation_duration_seconds": nanoseconds_to_seconds(
            generation_duration_ns
        ),
        "tokens_per_second": calculate_tokens_per_second(
            output_tokens,
            generation_duration_ns,
        ),
        "done_reason": response.done_reason or "unknown",
    }

    return answer, metrics


def initialize_session_state() -> None:
    """Inicializa el historial y las métricas de la sesión."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None


def clear_conversation() -> None:
    """Elimina el historial y las métricas actuales."""

    st.session_state.messages = []
    st.session_state.last_metrics = None


st.set_page_config(
    page_title="Qwen3 Ollama Chat",
    page_icon="🤖",
    layout="wide",
)

initialize_session_state()

with st.sidebar:
    st.header("Configuración")

    st.text_input(
        "Modelo",
        value=OLLAMA_MODEL,
        disabled=True,
    )

    st.text_input(
        "Servidor Ollama",
        value=OLLAMA_HOST,
        disabled=True,
    )

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.5,
        value=0.2,
        step=0.1,
        help=(
            "Valores bajos producen respuestas más deterministas. "
            "Valores altos aumentan la variabilidad."
        ),
    )

    st.divider()

    st.metric(
        "Mensajes en contexto",
        len(st.session_state.messages),
    )

    st.button(
        "Limpiar conversación",
        on_click=clear_conversation,
        use_container_width=True,
    )

st.title("Qwen3 + Ollama")
st.caption("Chat local con memoria temporal y métricas de inferencia")

st.info(
    f"Modelo activo: `{OLLAMA_MODEL}` · "
    f"Temperatura: `{temperature:.1f}`"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Escribe una pregunta para Qwen3...",
    max_chars=5000,
)

if prompt:
    clean_prompt = prompt.strip()

    if clean_prompt:
        user_message: ChatMessage = {
            "role": "user",
            "content": clean_prompt,
        }

        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(clean_prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Consultando Qwen3..."):
                    answer, metrics = query_ollama(
                        messages=st.session_state.messages,
                        temperature=temperature,
                    )

                if not answer:
                    st.error("Ollama devolvió una respuesta vacía.")

                else:
                    st.markdown(answer)

                    assistant_message: ChatMessage = {
                        "role": "assistant",
                        "content": answer,
                    }

                    st.session_state.messages.append(
                        assistant_message
                    )
                    st.session_state.last_metrics = metrics

        except ConnectionError:
            st.error(
                "No se puede conectar con Ollama en "
                f"`{OLLAMA_HOST}`. Comprueba que está iniciado."
            )

        except ResponseError as error:
            st.error(f"Ollama devolvió un error: {error.error}")

        except Exception as error:
            st.exception(error)

metrics = st.session_state.last_metrics

if metrics:
    st.divider()
    st.subheader("Métricas de la última respuesta")

    column_1, column_2, column_3, column_4 = st.columns(4)

    column_1.metric(
        "Tokens de entrada",
        metrics["input_tokens"],
    )

    column_2.metric(
        "Tokens de salida",
        metrics["output_tokens"],
    )

    column_3.metric(
        "Tokens totales",
        metrics["total_tokens"],
    )

    column_4.metric(
        "Tokens por segundo",
        f'{metrics["tokens_per_second"]:.2f}',
    )

    column_5, column_6, column_7, column_8 = st.columns(4)

    column_5.metric(
        "Tiempo total",
        f'{metrics["total_duration_seconds"]:.2f} s',
    )

    column_6.metric(
        "Carga del modelo",
        f'{metrics["load_duration_seconds"]:.2f} s',
    )

    column_7.metric(
        "Procesamiento prompt",
        f'{metrics["prompt_duration_seconds"]:.2f} s',
    )

    column_8.metric(
        "Generación",
        f'{metrics["generation_duration_seconds"]:.2f} s',
    )

    with st.expander("Información técnica"):
        st.json(
            {
                "model": metrics["model"],
                "done_reason": metrics["done_reason"],
                "temperature": temperature,
                "thinking_enabled": False,
                "messages_in_context": len(
                    st.session_state.messages
                ),
            }
        )