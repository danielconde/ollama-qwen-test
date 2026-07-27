"""Lógica principal del agente local basado en Ollama."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, TypedDict

from ollama import Client, ResponseError

from audit import write_audit_event
from tools import analyze_ip_address, execute_tool, get_local_time


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
MAXIMUM_TOOL_CALLS = 1


SYSTEM_PROMPT = """
Eres un agente local preciso y orientado a la seguridad.

Dispones de herramientas controladas para obtener información verificable.

Reglas obligatorias:

1. Usa analyze_ip_address cuando el usuario solicite analizar,
   validar o clasificar una dirección IP.

2. Usa get_local_time cuando el usuario pregunte por la fecha,
   la hora, el día actual, la zona horaria o información temporal
   actual del sistema.

3. No inventes resultados que puedan obtenerse mediante una herramienta.

4. No solicites ni ejecutes herramientas que no estén disponibles.

5. Después de usar una herramienta, explica el resultado con claridad.

6. Diferencia los datos proporcionados por una herramienta de tus
   propias interpretaciones.

7. No afirmes que una IP es maliciosa si la herramienta solamente
   proporciona clasificación técnica.
""".strip()


TOOLS = [
    analyze_ip_address,
    get_local_time,
]


class AgentMetrics(TypedDict):
    """Métricas generadas durante una ejecución del agente."""

    model: str
    tool_used: bool
    tool_name: str | None
    iterations: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_duration_seconds: float
    tokens_per_second: float
    tool_execution_ms: float
    done_reason: str | None
    temperature: float
    maximum_tool_calls: int
    thinking_enabled: bool


def nanoseconds_to_seconds(value: int | None) -> float:
    """Convierte nanosegundos en segundos."""

    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_tokens_per_second(
    output_tokens: int,
    duration_seconds: float,
) -> float:
    """Calcula los tokens de salida producidos por segundo."""

    if duration_seconds <= 0:
        return 0.0

    return output_tokens / duration_seconds


def run_agent(
    conversation: list[dict[str, Any]],
    temperature: float = 0.2,
    ollama_host: str | None = None,
    ollama_model: str | None = None,
) -> tuple[str, AgentMetrics, dict[str, Any] | None]:
    """
    Ejecuta una iteración completa del agente.

    El agente puede responder directamente o solicitar una herramienta.
    Como control de seguridad, solamente se permite una llamada a
    herramienta por ejecución.
    """

    resolved_host = (
        ollama_host
        or os.getenv("OLLAMA_HOST")
        or DEFAULT_OLLAMA_HOST
    )

    resolved_model = (
        ollama_model
        or os.getenv("OLLAMA_MODEL")
        or DEFAULT_OLLAMA_MODEL
    )

    client = Client(host=resolved_host)
    execution_id = str(uuid.uuid4())

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *conversation,
    ]

    first_response = client.chat(
        model=resolved_model,
        messages=messages,
        tools=TOOLS,
        think=False,
        options={
            "temperature": temperature,
        },
    )

    tool_calls = first_response.message.tool_calls or []

    write_audit_event(
        {
            "execution_id": execution_id,
            "event_type": "model_decision",
            "model": first_response.model or resolved_model,
            "tool_requested": bool(tool_calls),
            "requested_tool_count": len(tool_calls),
        }
    )

    tool_used = False
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None
    tool_execution_ms = 0.0
    iterations = 1

    total_input_tokens = first_response.prompt_eval_count or 0
    total_output_tokens = first_response.eval_count or 0
    total_duration_ns = first_response.total_duration or 0
    done_reason = first_response.done_reason

    if tool_calls:
        tool_used = True
        iterations = 2

        selected_tool_call = tool_calls[0]
        tool_name = selected_tool_call.function.name
        arguments = selected_tool_call.function.arguments or {}

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        tool_result, tool_execution_ms = execute_tool(
            tool_name,
            arguments,
        )

        write_audit_event(
            {
                "execution_id": execution_id,
                "event_type": "tool_execution",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": tool_result,
                "execution_ms": round(tool_execution_ms, 3),
            }
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
            model=resolved_model,
            messages=messages,
            think=False,
            options={
                "temperature": temperature,
            },
        )

        answer = final_response.message.content or ""

        total_input_tokens += final_response.prompt_eval_count or 0
        total_output_tokens += final_response.eval_count or 0
        total_duration_ns += final_response.total_duration or 0
        done_reason = final_response.done_reason

    else:
        answer = first_response.message.content or ""

    total_duration_seconds = nanoseconds_to_seconds(
        total_duration_ns
    )

    metrics: AgentMetrics = {
        "model": first_response.model or resolved_model,
        "tool_used": tool_used,
        "tool_name": tool_name,
        "iterations": iterations,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_duration_seconds": total_duration_seconds,
        "tokens_per_second": calculate_tokens_per_second(
            total_output_tokens,
            total_duration_seconds,
        ),
        "tool_execution_ms": tool_execution_ms,
        "done_reason": done_reason,
        "temperature": temperature,
        "maximum_tool_calls": MAXIMUM_TOOL_CALLS,
        "thinking_enabled": False,
    }

    write_audit_event(
        {
            "execution_id": execution_id,
            "event_type": "agent_completed",
            "model": metrics["model"],
            "tool_used": tool_used,
            "tool_name": tool_name,
            "iterations": iterations,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_duration_seconds": total_duration_seconds,
            "done_reason": done_reason,
        }
    )

    return answer, metrics, tool_result