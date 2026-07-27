"""Pruebas unitarias de las herramientas locales."""

from __future__ import annotations

from tools import analyze_ip_address, execute_tool, get_local_time


def test_analyze_private_ipv4() -> None:
    """Una IPv4 privada debe identificarse correctamente."""

    result = analyze_ip_address("192.168.1.25")

    assert result["success"] is True
    assert result["valid"] is True
    assert result["ip"] == "192.168.1.25"
    assert result["version"] == 4
    assert result["is_private"] is True
    assert result["is_global"] is False


def test_analyze_public_ipv4() -> None:
    """Una IPv4 global debe identificarse correctamente."""

    result = analyze_ip_address("8.8.8.8")

    assert result["success"] is True
    assert result["valid"] is True
    assert result["version"] == 4
    assert result["is_private"] is False
    assert result["is_global"] is True


def test_analyze_documentation_address() -> None:
    """Los rangos de documentación deben detectarse."""

    result = analyze_ip_address("203.0.113.25")

    assert result["success"] is True
    assert result["valid"] is True
    assert result["is_documentation_range"] is True


def test_analyze_loopback_address() -> None:
    """La dirección loopback debe detectarse."""

    result = analyze_ip_address("127.0.0.1")

    assert result["success"] is True
    assert result["is_loopback"] is True
    assert result["is_global"] is False


def test_analyze_ipv6() -> None:
    """La herramienta debe admitir IPv6."""

    result = analyze_ip_address("2001:db8::1")

    assert result["success"] is True
    assert result["valid"] is True
    assert result["version"] == 6
    assert result["is_documentation_range"] is True


def test_analyze_invalid_ip() -> None:
    """Una IP inválida debe devolver un error controlado."""

    result = analyze_ip_address("999.50.20.1")

    assert result["success"] is False
    assert result["valid"] is False
    assert "error" in result


def test_analyze_rejects_excessive_length() -> None:
    """La entrada no debe superar la longitud máxima."""

    result = analyze_ip_address("1" * 100)

    assert result["success"] is False
    assert result["valid"] is False
    assert "longitud máxima" in result["error"]


def test_get_local_time() -> None:
    """La herramienta debe devolver fecha y hora."""

    result = get_local_time()

    assert result["success"] is True
    assert result["date"]
    assert result["time"]
    assert result["local_datetime"]
    assert result["timezone"]
    assert result["utc_offset"]
    assert result["source"] == "Sistema operativo local"


def test_execute_ip_tool() -> None:
    """El dispatcher debe ejecutar la herramienta de IP."""

    result, execution_ms = execute_tool(
        "analyze_ip_address",
        {
            "ip_address": "10.0.0.1",
        },
    )

    assert result["success"] is True
    assert result["is_private"] is True
    assert execution_ms >= 0


def test_execute_time_tool() -> None:
    """El dispatcher debe ejecutar la herramienta de hora."""

    result, execution_ms = execute_tool(
        "get_local_time",
        {},
    )

    assert result["success"] is True
    assert result["time"]
    assert execution_ms >= 0


def test_time_tool_rejects_arguments() -> None:
    """get_local_time no debe aceptar argumentos."""

    result, execution_ms = execute_tool(
        "get_local_time",
        {
            "timezone": "Europe/Madrid",
        },
    )

    assert result["success"] is False
    assert "no admite parámetros" in result["error"]
    assert execution_ms >= 0


def test_ip_tool_rejects_non_string_argument() -> None:
    """La herramienta de IP debe validar el tipo."""

    result, execution_ms = execute_tool(
        "analyze_ip_address",
        {
            "ip_address": 12345,
        },
    )

    assert result["success"] is False
    assert "debe ser una cadena" in result["error"]
    assert execution_ms >= 0


def test_unauthorized_tool_is_blocked() -> None:
    """Una herramienta no autorizada debe bloquearse."""

    result, execution_ms = execute_tool(
        "run_command",
        {
            "command": "ipconfig",
        },
    )

    assert result["success"] is False
    assert result["error"] == "Herramienta no autorizada: run_command"
    assert execution_ms >= 0