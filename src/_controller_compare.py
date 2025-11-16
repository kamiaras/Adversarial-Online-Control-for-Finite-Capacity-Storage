"""
Internal helpers for controller comparison utilities.

These helpers normalize user-provided controller simulation payloads into a
consistent list of ``(name, simulation_dict)`` tuples so that
``compare_controllers`` can work with both the legacy ``(sim, name)`` arguments
and newer conveniences like dictionaries or iterables of controller outputs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

REQUIRED_KEYS = ("b", "u", "c")


def normalize_controller_payload(controller_args: Sequence[Any]) -> List[Tuple[str, dict]]:
    """
    Normalize inputs for compare_controllers into (name, sim_dict) tuples.

    Supports:
        - Legacy alternating arguments: compare_controllers(sim1, "A", sim2, "B")
        - Iterable of controller entries: [(sim1, "A"), {"name": "B", "sim": sim2}, ...]
        - Mapping from names to simulations: {"A": sim1, "B": sim2}
    """
    if not controller_args:
        raise ValueError("Provide at least one controller output to compare.")

    structured = _normalize_structured_payload(controller_args)
    if structured is not None:
        return structured

    return _normalize_from_pairs(controller_args)


def _normalize_structured_payload(controller_args: Sequence[Any]) -> Optional[List[Tuple[str, dict]]]:
    if len(controller_args) != 1:
        return None

    candidate = controller_args[0]
    mapping_result = _normalize_from_mapping(candidate)
    if mapping_result is not None:
        return mapping_result

    iterable_result = _normalize_from_iterable(candidate)
    if iterable_result is not None:
        return iterable_result

    return None


def _normalize_from_mapping(candidate: Any) -> Optional[List[Tuple[str, dict]]]:
    if not isinstance(candidate, dict):
        return None

    if not candidate:
        raise ValueError("Provide at least one controller output to compare.")

    if all(_is_simulation_dict(v) for v in candidate.values()):
        return [_validate_and_package(name, sim) for name, sim in candidate.items()]

    if any(_is_simulation_dict(v) for v in candidate.values()):
        raise ValueError(
            "When passing a dict, every value must be a simulation dictionary containing keys 'b', 'u', and 'c'."
        )

    return None


def _normalize_from_iterable(candidate: Any) -> Optional[List[Tuple[str, dict]]]:
    if isinstance(candidate, dict):
        return None  # already handled separately

    if not isinstance(candidate, (list, tuple)):
        return None

    if not candidate:
        raise ValueError("Provide at least one controller output to compare.")

    normalized: List[Tuple[str, dict]] = []
    for entry in candidate:
        normalized.append(_coerce_entry(entry))
    return normalized


def _normalize_from_pairs(controller_args: Sequence[Any]) -> List[Tuple[str, dict]]:
    if len(controller_args) < 4 or len(controller_args) % 2 != 0:
        raise ValueError(
            "Provide controller data as (sim, name) pairs, a list of controller entries, "
            "or a dict mapping display names to simulation outputs."
        )

    controller_data: List[Tuple[str, dict]] = []
    for idx in range(0, len(controller_args), 2):
        sim = controller_args[idx]
        name = controller_args[idx + 1]
        controller_data.append(_validate_and_package(name, sim))
    return controller_data


def _coerce_entry(entry: Any) -> Tuple[str, dict]:
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        first, second = entry
        if isinstance(first, str):
            return _validate_and_package(first, second)
        if isinstance(second, str):
            return _validate_and_package(second, first)

    if isinstance(entry, dict):
        if "name" in entry:
            sim_candidate = None
            for key in ("sim", "data", "output", "result"):
                if key in entry:
                    sim_candidate = entry[key]
                    break
            if sim_candidate is not None:
                return _validate_and_package(entry["name"], sim_candidate)
            if _is_simulation_dict(entry):
                return _validate_and_package(entry["name"], entry)
        if len(entry) == 1:
            (name, sim), = entry.items()
            if isinstance(name, str):
                return _validate_and_package(name, sim)

    raise ValueError(
        "Each controller entry must provide a display name and simulation dictionary with keys 'b', 'u', and 'c'."
    )


def _validate_and_package(name: Any, sim: Any) -> Tuple[str, dict]:
    if sim is None:
        raise ValueError(f"Simulation data for controller '{name}' is None.")
    if not isinstance(sim, dict):
        raise TypeError(f"Simulation for '{name}' must be provided as a dictionary.")
    for required_key in REQUIRED_KEYS:
        if required_key not in sim:
            raise KeyError(f"Simulation for '{name}' missing key '{required_key}'.")
    return str(name), sim


def _is_simulation_dict(candidate: Any) -> bool:
    return isinstance(candidate, dict) and all(key in candidate for key in REQUIRED_KEYS)
