"""Starter authentication helpers for backend API routes.

The first version intentionally stays simple: if no API key environment
variable is configured, local development routes remain open. When
``ALZ_API_KEY`` or ``ALZ_BACKEND_API_KEY`` is set, protected routes require
the same value in the ``X-API-Key`` header.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from typing import Annotated

from fastapi import Header, HTTPException, status


@dataclass(slots=True, frozen=True)
class AuthContext:
    """Authentication state attached to protected route calls."""

    enabled: bool
    authenticated: bool
    scheme: str = "api_key"


def _configured_api_key() -> str | None:
    """Return the configured development API key, if any."""

    return getenv("ALZ_API_KEY") or getenv("ALZ_BACKEND_API_KEY")


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> AuthContext:
    """Require an API key only when one is configured for this environment."""

    expected_key = _configured_api_key()
    if not expected_key:
        return AuthContext(enabled=False, authenticated=True)

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
    return AuthContext(enabled=True, authenticated=True)

