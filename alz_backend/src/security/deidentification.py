"""De-identification helpers for research exports and audit-safe logging."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

DEFAULT_PHI_KEYS = frozenset(
    {
        "first_name",
        "last_name",
        "preferred_name",
        "name",
        "date_of_birth",
        "dob",
        "medical_record_number",
        "mrn",
        "phone",
        "email",
        "address",
        "address_line_1",
        "address_line_2",
        "city",
        "postal_code",
        "emergency_contact_name",
        "emergency_contact_phone",
        "patient_id",
    }
)


def pseudonymize_identifier(identifier: str, *, salt: str = "alz_backend_research") -> str:
    """Create a deterministic pseudonymous identifier for research linkage."""

    digest = hashlib.sha256(f"{salt}:{identifier}".encode("utf-8")).hexdigest()
    return f"anon_{digest[:16]}"


def redact_text(value: str, *, replacement: str = "[REDACTED]") -> str:
    """Redact common email and phone-like patterns from free text."""

    redacted = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", replacement, value)
    redacted = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", replacement, redacted)
    return redacted


def deidentify_mapping(
    payload: Mapping[str, Any],
    *,
    phi_keys: set[str] | frozenset[str] = DEFAULT_PHI_KEYS,
    replacement: str = "[REDACTED]",
) -> dict[str, Any]:
    """Recursively remove or redact PHI/PII-like keys from a mapping."""

    def _clean(value: Any, *, key: str | None = None) -> Any:
        if key is not None and key.lower() in phi_keys:
            return replacement
        if isinstance(value, Mapping):
            return {str(nested_key): _clean(nested_value, key=str(nested_key)) for nested_key, nested_value in value.items()}
        if isinstance(value, list):
            return [_clean(item) for item in value]
        if isinstance(value, str):
            return redact_text(value, replacement=replacement)
        return deepcopy(value)

    return {str(key): _clean(value, key=str(key)) for key, value in payload.items()}


def assert_no_phi_keys(payload: Mapping[str, Any], *, phi_keys: set[str] | frozenset[str] = DEFAULT_PHI_KEYS) -> None:
    """Fail if a research export payload contains obvious PHI/PII keys."""

    violations: list[str] = []

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                key_text = str(key)
                nested_path = f"{path}.{key_text}" if path else key_text
                if key_text.lower() in phi_keys:
                    violations.append(nested_path)
                _walk(nested_value, nested_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                _walk(item, f"{path}[{index}]")

    _walk(payload, "")
    if violations:
        raise ValueError(f"Research export contains PHI/PII-like keys: {', '.join(sorted(violations))}")

