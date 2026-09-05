"""Validation contracts for the conditional online formal configuration set."""

from __future__ import annotations

from typing import Any

from .system_config import LongLiveSystemConfig
from .tethermem import TETHER_METHODS


REQUIRED_CONFIG_IDS = {
    "rag_dense",
    "legacy_final",
    "legacy_final_system",
    "best_causal_or_codesign",
}


def validate_system_method_freeze(payload: dict[str, Any]) -> list[str]:
    errors = []
    if payload.get("status") != "frozen_after_system_calibration":
        errors.append("system method set is not frozen after calibration")
    if payload.get("formal_results_used") is not False:
        errors.append("method freeze must occur before formal results")
    configs = payload.get("configs")
    if not isinstance(configs, list) or len(configs) != 4:
        errors.append("formal online set must contain exactly four configurations")
        configs = []
    ids = [str(item.get("config_id")) for item in configs if isinstance(item, dict)]
    if len(ids) != len(set(ids)):
        errors.append("formal configuration ids must be unique")
    if set(ids) != REQUIRED_CONFIG_IDS:
        errors.append("formal configuration ids do not match the frozen four-cell set")
    for config in configs:
        if not isinstance(config, dict):
            errors.append("formal configurations must be objects")
            continue
        method = str(config.get("method"))
        if method in TETHER_METHODS and not TETHER_METHODS[method].online_speed_pareto_eligible:
            errors.append(f"offline oracle cannot enter formal online set: {method}")
        if config.get("online") is not True:
            errors.append(f"formal configuration must be online: {config.get('config_id')}")
        system = config.get("longlive_system")
        if system is not None:
            try:
                LongLiveSystemConfig.from_mapping(system)
            except (TypeError, ValueError) as error:
                errors.append(
                    f"invalid system config {config.get('config_id')}: {error}"
                )
        if method == "rag_dense":
            if float(config.get("history_density", -1)) != 1.0:
                errors.append("rag_dense formal baseline must use density 1.0")
        elif float(config.get("history_density", -1)) > 0.25:
            errors.append(f"online sparse config exceeds 25% budget: {config.get('config_id')}")
    for field in ("calibration_audit", "profile_audit", "quality_gate"):
        value = payload.get(field)
        if not isinstance(value, dict) or len(str(value.get("sha256", ""))) != 64:
            errors.append(f"{field} must include a SHA-256 digest")
    return errors
