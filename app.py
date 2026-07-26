"""Agente local con Qwen3, Ollama, Streamlit y una herramienta segura."""

from __future__ import annotations

import ipaddress
import json
import os
import time
from typing import Any, TypedDict

import streamlit as st
from ollama import Client, ResponseError


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SYSTEM_PROMPT = """
Eres un agente técnico especializado en ciberseguridad.

Dispones de una herramienta llamada analyze_ip_address para analizar
direcciones IPv4 o IPv6.

Reglas obligatorias:

1. Usa la herramienta cuando el usuario solicite analizar, validar,
   clasificar o investigar una dirección IP.
2. No inventes propiedades de una IP.
3. Trata el resultado de la herramienta como evidencia técnica.
4. Diferencia claramente hechos e interpretación.
5. No afirmes que una IP es maliciosa porque la herramienta no consulta
   reputación ni inteligencia de amenazas.
6. Cuando no sea necesario utilizar la herramienta, responde directamente.
7. Responde en español de forma clara y concisa.
""".strip()


class ChatMessage(TypedDict, total=False):
    """Mensaje de conversación compatible con Ollama."""

    role: str
    content: str
    tool_name: str


class AgentMetrics(TypedDict):
    """Métricas de la ejecución del agente."""

    model: str
    tool_used: bool
    tool_name: str | None
    tool_execution_ms: float
    agent_iterations: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_duration_seconds: float
    tokens_per_second: float
    done_reason: str


def analyze_ip_address(ip_address: str) -> dict[str, Any]:
    """
    Analiza una dirección IPv4 o IPv6 sin realizar consultas externas.

    La función no determina reputación, geolocalización ni actividad
    maliciosa.
    """

    clean_ip = ip_address.strip()

    try:
        parsed_ip = ipaddress.ip_address(clean_ip)
    except ValueError:
        return {
            "valid": False,
            "input": clean_ip,
            "error": "La dirección proporcionada no es una IPv4 o IPv6 válida.",
        }

    return {
        "valid": True,
        "ip": str(parsed_ip),
        "version": parsed_ip.version,
        "is_private": parsed_ip.is_private,
        "is_global": parsed_ip.is_global,
        "is_loopback": parsed_ip.is_loopback,
        "is_link_local": parsed_ip.is_link_local,
        "is_multicast": parsed_ip.is_multicast,
        "is_reserved": parsed_ip.is_reserved,
        "is_unspecified": parsed_ip.is_unspecified,
        "is_documentation_range": is_documentation_address(parsed_ip),
        "limitations": [
            "No se ha consultado reputación.",
            "No se ha realizado geolocalización.",
            "No se ha consultado inteligencia de amenazas.",
            "No se puede determinar si la IP es maliciosa.",
        ],
    }


def is_documentation_address(
    parsed_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Determina si una IP pertenece a un rango reservado para documentación."""

    documentation_networks = [
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
        ipaddress.ip_network("2001:db8::/32"),
    ]

    return any(parsed_ip in network for network in documentation_networks)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_ip_address",
            "description": (
                "Valida y clasifica una dirección IPv4 o IPv6. "
                "No consulta reputación, geolocalización ni fuentes externas."
            ),
            "parameters": {
                "type": "object",
                "required": ["ip_address"],
                "properties": {
                    "ip_address": {
                        "type": "string",
                        "description": (
                            "Dirección IPv4 o IPv6 que debe analizarse."
                        ),
                    }
                },
                "additionalProperties": False,
            },
        },
    }
]


def nanoseconds_to_seconds(value: int | None) -> float:
    """Convierte nanosegundos a segundos."""

    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_tokens_per_second(
    output_tokens: int,
    generation_duration_ns: int,
) -> float:
    """Calcula la velocidad de generación."""

    duration = nanoseconds_to_seconds(generation_duration_ns)

    if output_tokens == 0 or duration == 0:
        return 0.0

    return output_tokens / duration


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Ejecuta únicamente herramientas explícitamente permitidas."""

    start_time = time.perf_counter()

    if tool_name == "analyze_ip_address":
        ip_address = arguments.get("ip_address")

        if not isinstance(ip_address, str):
            result = {
                "valid": False,
                "error": "El parámetro ip_address debe ser una cadena.",
            }
        else:
            result = analyze_ip_address(ip_address)
    else:
        result = {
            "error": f"Herramienta no autorizada: {tool_name}",
        }

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return result, elapsed_ms


def run_agent(
    conversation: list[dict[str, Any]],
    temperature: float,
) -> tuple[str, AgentMetrics, dict[str, Any] | None]:
    """
    Ejecuta un ciclo controlado de tool calling.

    El agente tiene un máximo de una llamada a herramienta y una respuesta
    final. Este límite evita bucles de ejecución.
    """

    client = Client(host=OLLAMA_HOST)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *conversation,
    ]

    first_response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=TOOLS,
        think=False,
        options={
            "temperature": temperature,
        },
    )

    total_input_tokens = first_response.prompt_eval_count or 0
    total_output_tokens = first_response.eval_count or 0
    total_duration_ns = first_response.total_duration or 0
    total_generation_duration_ns = first_response.eval_duration or 0

    tool_used = False
    tool_name: str | None = None
    tool_execution_ms = 0.0
    tool_result: dict[str, Any] | None = None
    iterations = 1

    tool_calls = first_response.message.tool_calls or []

    if tool_calls:
        tool_used = True

        selected_call = tool_calls[0]
        tool_name = selected_call.function.name
        arguments = selected_call.function.arguments or {}

        tool_result, tool_execution_ms = execute_tool(
            tool_name=tool_name,
            arguments=arguments,
        )

        messages.append(first_response.message.model_dump())

        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            }
        )

        final_response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOLS,
            think=False,
            options={
                "temperature": temperature,
            },
        )

        iterations = 2

        total_input_tokens += final_response.prompt_eval_count or 0
        total_output_tokens += final_response.eval_count or 0
        total_duration_ns += final_response.total_duration or 0
        total_generation_duration_ns += final_response.eval_duration or 0

        answer = final_response.message.content or ""
        done_reason = final_response.done_reason or "unknown"

    else:
        answer = first_response.message.content or ""
        done_reason = first_response.done_reason or "unknown"

    metrics: AgentMetrics = {
        "model": first_response.model or OLLAMA_MODEL,
        "tool_used": tool_used,
        "tool_name": tool_name,
        "tool_execution_ms": tool_execution_ms,
        "agent_iterations": iterations,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_duration_seconds": nanoseconds_to_seconds(
            total_duration_ns
        ),
        "tokens_per_second": calculate_tokens_per_second(
            total_output_tokens,
            total_generation_duration_ns,
        ),
        "done_reason": done_reason,
    }

    return answer, metrics, tool_result


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
        "analyze_ip_address",
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
        metrics["agent_iterations"],
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
                "iterations": metrics["agent_iterations"],
                "done_reason": metrics["done_reason"],
                "temperature": temperature,
                "maximum_tool_calls": 1,
                "thinking_enabled": False,
            }
        )