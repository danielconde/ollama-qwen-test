"""Agente local con Qwen3, Ollama, Streamlit y una herramienta segura."""
from __future__ import annotations

import os
import streamlit as st
from typing import TypedDict

from ollama import ResponseError

from agent import AgentMetrics, run_agent
from audit import read_recent_events


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

class ChatMessage(TypedDict, total=False):
    """Mensaje de conversación compatible con Ollama."""

    role: str
    content: str
    tool_name: str


def initialize_session_state() -> None:
    """Inicializa los datos almacenados durante la sesión."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None

    if "last_tool_result" not in st.session_state:
        st.session_state.last_tool_result = None


def clear_conversation() -> None:
    """Limpia el historial del agente."""

    st.session_state.messages = []
    st.session_state.last_metrics = None
    st.session_state.last_tool_result = None


st.set_page_config(
    page_title="Qwen3 IP Agent",
    page_icon="🛡️",
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
        "Servidor",
        value=OLLAMA_HOST,
        disabled=True,
    )

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
    )

    st.divider()

    st.write("Herramientas disponibles")

    st.code(
        "analyze_ip_address\nget_local_time",
        language="text",
    )

    st.button(
        "Limpiar conversación",
        on_click=clear_conversation,
        use_container_width=True,
    )

st.title("Agente local de análisis de IP")
st.caption("Qwen3 + Ollama + tool calling controlado")

st.warning(
    "La herramienta solo clasifica la dirección IP. "
    "No consulta reputación, geolocalización ni inteligencia de amenazas."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Pregunta algo o solicita el análisis de una dirección IP...",
    max_chars=2000,
)

if prompt:
    clean_prompt = prompt.strip()

    if clean_prompt:
        user_message = {
            "role": "user",
            "content": clean_prompt,
        }

        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(clean_prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Ejecutando agente..."):
                    answer, metrics, tool_result = run_agent(
                        conversation=st.session_state.messages,
                        temperature=temperature,
                    )

                if not answer:
                    st.error("El agente devolvió una respuesta vacía.")
                else:
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    st.session_state.last_metrics = metrics
                    st.session_state.last_tool_result = tool_result

        except ConnectionError:
            st.error(
                f"No se puede conectar con Ollama en `{OLLAMA_HOST}`."
            )

        except ResponseError as error:
            st.error(f"Ollama devolvió un error: {error.error}")

        except Exception as error:
            st.exception(error)

metrics = st.session_state.last_metrics

if metrics:
    st.divider()
    st.subheader("Ejecución del agente")

    col_1, col_2, col_3, col_4 = st.columns(4)

    col_1.metric(
        "Herramienta utilizada",
        "Sí" if metrics["tool_used"] else "No",
    )

    col_2.metric(
        "Iteraciones",
        metrics["iterations"],
    )

    col_3.metric(
        "Tokens totales",
        metrics["total_tokens"],
    )

    col_4.metric(
        "Tokens por segundo",
        f'{metrics["tokens_per_second"]:.2f}',
    )

    col_5, col_6, col_7, col_8 = st.columns(4)

    col_5.metric(
        "Tokens de entrada",
        metrics["input_tokens"],
    )

    col_6.metric(
        "Tokens de salida",
        metrics["output_tokens"],
    )

    col_7.metric(
        "Duración total",
        f'{metrics["total_duration_seconds"]:.2f} s',
    )

    col_8.metric(
        "Tiempo herramienta",
        f'{metrics["tool_execution_ms"]:.2f} ms',
    )

    if st.session_state.last_tool_result:
        with st.expander("Resultado técnico de la herramienta"):
            st.json(st.session_state.last_tool_result)

    with st.expander("Trazabilidad"):
        st.json(
            {
                "model": metrics["model"],
                "tool_used": metrics["tool_used"],
                "tool_name": metrics["tool_name"],
                "iterations": metrics["iterations"],
                "done_reason": metrics["done_reason"],
                "temperature": temperature,
                "maximum_tool_calls": 1,
                "thinking_enabled": False,
            }
        )


st.divider()
st.subheader("Auditoría local")

recent_events = read_recent_events(limit=10)

if recent_events:
    with st.expander("Últimos eventos registrados"):
        st.json(recent_events)
else:
    st.caption("Todavía no existen eventos de auditoría.")