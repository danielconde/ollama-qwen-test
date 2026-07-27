"""Herramientas locales disponibles para el agente."""

from __future__ import annotations

import ipaddress
import time
from typing import Any


def is_documentation_address(
    parsed_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Comprueba si una IP pertenece a un rango de documentación."""

    documentation_networks = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
        ipaddress.ip_network("2001:db8::/32"),
    )

    return any(parsed_ip in network for network in documentation_networks)


def analyze_ip_address(ip_address: str) -> dict[str, Any]:
    """
    Valida y clasifica una dirección IPv4 o IPv6.

    No realiza consultas externas de reputación, geolocalización
    ni inteligencia de amenazas.
    """

    clean_ip = ip_address.strip()

    try:
        parsed_ip = ipaddress.ip_address(clean_ip)
    except ValueError:
        return {
            "success": False,
            "valid": False,
            "input": clean_ip,
            "error": "La dirección no es una IPv4 o IPv6 válida.",
        }

    return {
        "success": True,
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
            "No puede determinarse si la IP es maliciosa.",
        ],
    }


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """
    Ejecuta una herramienta incluida explícitamente en la lista permitida.

    Devuelve el resultado y el tiempo de ejecución en milisegundos.
    """

    start_time = time.perf_counter()

    if tool_name == "analyze_ip_address":
        ip_address = arguments.get("ip_address")

        if not isinstance(ip_address, str):
            result = {
                "success": False,
                "error": "El parámetro ip_address debe ser una cadena.",
            }
        else:
            result = analyze_ip_address(ip_address)

    else:
        result = {
            "success": False,
            "error": f"Herramienta no autorizada: {tool_name}",
        }

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return result, elapsed_ms