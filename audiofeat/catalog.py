from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections import defaultdict
from typing import Iterable


DEFAULT_COMPONENTS: tuple[str, ...] = (
    "audiofeat.temporal",
    "audiofeat.spectral",
    "audiofeat.pitch",
    "audiofeat.voice",
    "audiofeat.cepstral",
    "audiofeat.stats",
    "audiofeat.io",
    "audiofeat.validation",
    "audiofeat.preprocessing",
    "audiofeat.rhythm",
    "audiofeat.segmentation",
)


def _doc_summary(obj: object) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line.rstrip(".")
    return "No description available."


def _safe_signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "()"


def _iter_component_modules(component: str) -> list[str]:
    package = importlib.import_module(component)
    if not hasattr(package, "__path__"):
        return [component]

    modules = [component]
    for mod in pkgutil.iter_modules(package.__path__, prefix=f"{component}."):
        modules.append(mod.name)
    return modules


def _iter_public_functions(module_name: str) -> list[dict[str, str]]:
    module = importlib.import_module(module_name)
    results: list[dict[str, str]] = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        results.append(
            {
                "name": name,
                "module": module_name,
                "signature": _safe_signature(obj),
                "description": _doc_summary(obj),
            }
        )
    results.sort(key=lambda x: (x["module"], x["name"]))
    return results


def build_feature_catalog(
    components: Iterable[str] = DEFAULT_COMPONENTS,
) -> dict[str, dict[str, object]]:
    """
    Build an auto-discovered catalog of user-facing functions grouped by component.

    The catalog is generated from the current codebase, so README/CLI output can stay
    aligned with exported capabilities as modules evolve.
    """
    catalog: dict[str, dict[str, object]] = {}
    for component in components:
        try:
            module_names = _iter_component_modules(component)
            features: list[dict[str, str]] = []
            errors: list[str] = []
            for module_name in module_names:
                try:
                    features.extend(_iter_public_functions(module_name))
                except Exception as exc:
                    errors.append(f"{module_name}: {exc}")

            features.sort(key=lambda x: (x["module"], x["name"]))
            catalog[component] = {
                "status": "ok" if not errors else "partial",
                "feature_count": len(features),
                "features": features,
                "errors": errors,
            }
        except Exception as exc:
            catalog[component] = {
                "status": "error",
                "feature_count": 0,
                "features": [],
                "errors": [str(exc)],
            }
    return catalog


def catalog_to_markdown(catalog: dict[str, dict[str, object]]) -> str:
    """Render a markdown table from `build_feature_catalog()` output."""
    lines: list[str] = []
    lines.append("| Component | Function | Signature | Description |")
    lines.append("|---|---|---|---|")
    for component, payload in sorted(catalog.items()):
        features = payload.get("features", [])
        if not features:
            status = payload.get("status", "error")
            lines.append(f"| `{component}` | _none_ | - | status: `{status}` |")
            continue
        for item in features:
            lines.append(
                "| "
                f"`{component}` | `{item['name']}` | `{item['signature']}` | {item['description']} |"
            )
    return "\n".join(lines)


def summarize_catalog(catalog: dict[str, dict[str, object]]) -> dict[str, int]:
    """Return lightweight summary counts for quick diagnostics."""
    status_counts = defaultdict(int)
    total_features = 0
    for payload in catalog.values():
        status_counts[str(payload.get("status", "error"))] += 1
        total_features += int(payload.get("feature_count", 0))
    out = dict(status_counts)
    out["total_components"] = len(catalog)
    out["total_features"] = total_features
    return out
