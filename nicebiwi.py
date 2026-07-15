from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt

from nicegui import app, events, ui

WORKSPACE_ROOT = Path(__file__).resolve().parent
SIBLING_BIWIPY = WORKSPACE_ROOT.parent / "biwipy"
if SIBLING_BIWIPY.exists() and str(SIBLING_BIWIPY) not in sys.path:
    sys.path.insert(0, str(SIBLING_BIWIPY))

from biwipy import RouteAnalyzer, Simulator
from biwipy.core.cyclist_params import (
    CyclistBehavior,
    create_amateur_profile,
    create_competitive_profile,
    create_pro_profile,
)
from biwipy.weather import WeatherProvider, RoughnessProvider
from biwipy.weather.grib_finder import build_grib_list
from biwipy.analysis.anareswind import (
    print_summary_statistics,
    smooth_segments,
    plot_wind_rose,
    plot_elevation_profile,
    plot_segments_evolution,
    compare_scenarios,
)
from biwipy.visualization.interactive_map import create_interactive_map


GENERATED_MAPS_DIR = WORKSPACE_ROOT / ".generated_maps"
GENERATED_MAPS_ROUTE = "/generated-maps"
GENERATED_MAPS_DIR.mkdir(parents=True, exist_ok=True)
try:
    app.add_static_files(GENERATED_MAPS_ROUTE, str(GENERATED_MAPS_DIR))
except Exception:
    # Route may already be registered when hot-reloading the app.
    pass

def load_translations(base_dir: Path) -> dict[str, dict[str, str]]:
    translations: dict[str, dict[str, str]] = {}
    i18n_dir = base_dir / "i18n"

    for lang in ("fr", "en"):
        lang_file = i18n_dir / f"{lang}.json"
        with lang_file.open("r", encoding="utf-8") as f:
            lang_map = json.load(f)

        if not isinstance(lang_map, dict):
            raise RuntimeError(f"Invalid translation file format: {lang_file}")

        for key, value in lang_map.items():
            if not isinstance(value, str):
                raise RuntimeError(f"Invalid translation value for key '{key}' in {lang_file}")
            translations.setdefault(key, {})[lang] = value

    return translations


TRANSLATIONS = load_translations(WORKSPACE_ROOT)
REPLAY_SMOOTHING_WINDOW = 7

PROFILE_PRESETS = {
    "conservative": create_amateur_profile,
    "competitive": create_competitive_profile,
    "pro": create_pro_profile,
}

PRESET_TO_MODES = {
    "conservative": {"uphill": "conservative", "downhill": "conservative", "corner": "conservative"},
    "competitive": {"uphill": "realistic", "downhill": "realistic", "corner": "realistic"},
    "pro": {"uphill": "aggressive", "downhill": "aggressive", "corner": "aggressive"},
}

MODE_LABEL_KEYS = {
    "conservative": "mode_conservative",
    "realistic": "mode_realistic",
    "aggressive": "mode_aggressive",
}

GUI_SETTINGS_PATH = Path.home() / ".config" / "biwipy" / "GUIsettings.json"
RUN_HISTORY_PATH = Path.home() / ".config" / "biwipy" / "run_history.json"
MAX_RUN_HISTORY = 80
DEFAULT_PERSISTED_PARAMS = {
    "general_grib_dir": ".",
    "cyclist_params_dir": "/home/arquey/.config/biwipy/GUIparam/",
    "CdA": 0.50,
    "Cr": 0.005,
    "mass_kg": 85.0,
    "weather_model": "GFS",
    "weather_pas": 1,
    "weather_use_raster_roughness": False,
    "map_enable_animation": False,
    "map_distance_from_finish": False,
    "process_gpx_verbose": True,
}

state: dict[str, Any] = {
    "language": "fr",
    "gpx_result": None,
    "gpx_source_result": None,
    "gpx_source_name": None,
    "gpx_name": None,
    "status_message": None,
    "needs_setup": False,
    "setup_error": None,
    "simulation_result": None,
    "simulation_result_run_id": None,
    "simulation_markdown": "",
    "run_history": [],
    "selected_run_id": None,
    "compare_run_ids": [],
    "history_kind_filter": "all",
    "history_sort_order": "newest",
    "results_active_tab": "summary",
    "plot_wind_rose_img": None,
    "plot_elevation_img": None,
    "plot_speed_wind_img": None,
    "plots_gpx_name": None,
    "plot_compare_img": None,
    "interactive_map_url": None,
    "interactive_map_run_id": None,
    "interactive_map_title": None,
    "segments_cache": {},  # in-memory only: run_id -> raw segments list (not persisted)
    "is_busy": False,
    "busy_message": "",
    "cut_form": {"p1_km": "", "p2_km": ""},
    "selected_cyclist_profile_file": "",
    "cyclist_profile_name": "",
    "active_behavior_thresholds": {},
    "profile_manager_open": False,
    "profile_form": {
        "CdA": 0.50,
        "Cr": 0.005,
        "mass_kg": 85.0,
        "profile_preset": "competitive",
        "uphill_mode": "realistic",
        "downhill_mode": "realistic",
        "corner_mode": "realistic",
        "behavior_thresholds": {},  # optional: dict of customizable threshold parameters
    },
    "setup_form": dict(DEFAULT_PERSISTED_PARAMS),
    "params": {
        **DEFAULT_PERSISTED_PARAMS,
        "sim_target_mode": "p0",
        "sim_p0": 220.0,
        "sim_v0_kmh": 30.0,
        "sim_start": (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M"),
        "sim_duration_h": 4,
        "profile_preset": "competitive",
        "uphill_mode": "realistic",
        "downhill_mode": "realistic",
        "corner_mode": "realistic",
        "weather_model": "GFS",
        "weather_pas": 1,
        "weather_use_raster_roughness": False,
        "map_enable_animation": False,
        "map_distance_from_finish": False,
        "process_gpx_verbose": True,
        "smoothing_window": 17,
    },
}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _serialize_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        return str(value)
    except Exception:
        return None


def run_kind_label_key(run_kind: str) -> str:
    mapping = {
        "replay_no_weather": "run_type_replay_no_weather",
        "replay_weather": "run_type_replay_weather",
        "future_no_weather": "run_type_future_no_weather",
        "future_weather": "run_type_future_weather",
    }
    return mapping.get(run_kind, "run_type_unknown")


def load_run_history() -> None:
    if not RUN_HISTORY_PATH.exists():
        state["run_history"] = []
        return

    try:
        with RUN_HISTORY_PATH.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise RuntimeError("run history must be a JSON array")
        history = [entry for entry in raw if isinstance(entry, dict)]
        history.sort(key=lambda entry: str(entry.get("created_at", "")), reverse=True)
        state["run_history"] = history[:MAX_RUN_HISTORY]
    except Exception as exc:
        state["run_history"] = []
        ui.notify(tr("history_load_error", error=exc), type="warning")


def save_run_history() -> None:
    try:
        RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUN_HISTORY_PATH.open("w", encoding="utf-8") as file:
            json.dump(state["run_history"], file, ensure_ascii=False, indent=2)
    except Exception as exc:
        ui.notify(tr("history_save_error", error=exc), type="warning")


def build_run_record(
    run_kind: str,
    result: Any,
    gpx_result: Any,
    weather_context: dict[str, Any] | None = None,
    target_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = state["params"]
    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("%Y%m%dT%H%M%S%fZ")

    speed_avg = _float_or_none(getattr(getattr(result, "speed", None), "avg", None))
    power_avg = _float_or_none(getattr(getattr(result, "power", None), "avg", None))
    p0_calibrated = _float_or_none(getattr(getattr(result, "power", None), "P0_calibrated", None))
    distance_km = _float_or_none(getattr(getattr(result, "distance", None), "total_km", None))
    wind_grade = getattr(getattr(result, "wind_score", None), "grade", None)
    wind_reason = getattr(getattr(result, "wind_score", None), "reason", None)

    route_duration_s: float | None = None
    if gpx_result is not None and getattr(gpx_result, "duration", None) is not None:
        route_duration_s = _float_or_none(getattr(gpx_result.duration, "total_seconds", lambda: None)())

    # Total run duration must come from simulation output, not raw GPX duration.
    simulation_duration_s = _float_or_none(getattr(getattr(result, "time", None), "total_seconds", None))

    return {
        "id": run_id,
        "created_at": now_utc.isoformat(),
        "kind": run_kind,
        "source_gpx": state.get("gpx_name") or "",
        "metrics": {
            "speed_kmh": speed_avg,
            "power_w": power_avg,
            "p0_w": p0_calibrated,
            "distance_km": distance_km,
            "duration_s": simulation_duration_s,
            "windscore_grade": str(wind_grade) if wind_grade else None,
            "windscore_reason": str(wind_reason) if wind_reason else None,
        },
        "route": {
            "segments": len(gpx_result.segments) if gpx_result is not None and getattr(gpx_result, "segments", None) else 0,
            "distance_km": _float_or_none(getattr(gpx_result, "distance_km", None)) if gpx_result is not None else None,
            "has_timestamps": bool(getattr(gpx_result, "has_timestamps", False)) if gpx_result is not None else False,
            "start": _serialize_dt(getattr(gpx_result, "t_start", None)) if gpx_result is not None else None,
            "end": _serialize_dt(getattr(gpx_result, "t_end", None)) if gpx_result is not None else None,
            "duration_s": route_duration_s,
        },
        "params": {
            "CdA": _float_or_none(params.get("CdA")),
            "Cr": _float_or_none(params.get("Cr")),
            "mass_kg": _float_or_none(params.get("mass_kg")),
            "profile_preset": str(params.get("profile_preset", "")),
            "uphill_mode": str(params.get("uphill_mode", "")),
            "downhill_mode": str(params.get("downhill_mode", "")),
            "corner_mode": str(params.get("corner_mode", "")),
            "weather_model": str(params.get("weather_model", "")),
            "weather_pas": int(params.get("weather_pas", 1)),
            "sim_target_mode": str(params.get("sim_target_mode", "p0")),
            "sim_p0": _float_or_none(params.get("sim_p0")),
            "sim_v0_kmh": _float_or_none(params.get("sim_v0_kmh")),
            "sim_start": str(params.get("sim_start", "")),
            "sim_duration_h": int(params.get("sim_duration_h", 0) or 0),
        },
        "weather": weather_context or {},
        "target": target_context or {},
        "summary_markdown": state.get("simulation_markdown") or "",
        "summary_statistics": capture_summary_statistics(
            result,
            label=state.get("gpx_name") or "Run",
        ),
    }


def add_run_to_history(record: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = list(state.get("run_history", []))
    history.insert(0, record)
    history = history[:MAX_RUN_HISTORY]
    state["run_history"] = history
    state["selected_run_id"] = record.get("id")
    compare_ids: list[str] = list(state.get("compare_run_ids", []))
    state["compare_run_ids"] = [run_id for run_id in compare_ids if any(r.get("id") == run_id for r in history)][:2]
    save_run_history()


def get_run_by_id(run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    for record in state.get("run_history", []):
        if str(record.get("id")) == str(run_id):
            return record
    return None


def get_active_run() -> dict[str, Any] | None:
    selected = get_run_by_id(state.get("selected_run_id"))
    if selected is not None:
        return selected
    history: list[dict[str, Any]] = state.get("run_history", [])
    return history[0] if history else None


def select_run(run_id: str) -> None:
    if get_run_by_id(run_id) is None:
        return
    state["selected_run_id"] = run_id
    refresh_all()


def toggle_compare_run(run_id: str, enabled: bool) -> None:
    compare_ids: list[str] = list(state.get("compare_run_ids", []))
    if enabled:
        if run_id not in compare_ids:
            compare_ids.append(run_id)
        compare_ids = compare_ids[-2:]
    else:
        compare_ids = [value for value in compare_ids if value != run_id]
    state["compare_run_ids"] = compare_ids
    refresh_all()


def set_history_kind_filter(value: str) -> None:
    state["history_kind_filter"] = value
    refresh_all()


def clear_run_history() -> None:
    state["run_history"] = []
    state["selected_run_id"] = None
    state["compare_run_ids"] = []
    save_run_history()
    refresh_all()


def _cache_run_segments(run_id: str, result: Any) -> None:
    if not run_id:
        return
    try:
        state["segments_cache"][run_id] = result.get_segments()
        # Keep cache bounded to 20 most recently added entries
        if len(state["segments_cache"]) > 20:
            oldest = list(state["segments_cache"].keys())[:-20]
            for k in oldest:
                del state["segments_cache"][k]
    except Exception:
        pass


def delete_run_history_entry(run_id: str) -> None:
    history: list[dict[str, Any]] = list(state.get("run_history", []))
    history = [record for record in history if str(record.get("id")) != str(run_id)]
    state["run_history"] = history

    if str(state.get("selected_run_id")) == str(run_id):
        state["selected_run_id"] = history[0].get("id") if history else None

    compare_ids: list[str] = list(state.get("compare_run_ids", []))
    state["compare_run_ids"] = [rid for rid in compare_ids if str(rid) != str(run_id)]

    state["segments_cache"].pop(str(run_id), None)

    save_run_history()
    refresh_all()


def open_delete_run_confirmation(run_id: str) -> None:
    run = get_run_by_id(run_id)
    if run is None:
        return

    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(tr("run_delete_confirm_title")).classes("text-subtitle1 text-weight-medium")
        ui.label(get_run_title(run)).classes("text-body2 text-grey-8")
        ui.label(tr("run_delete_confirm_text")).classes("text-body2 text-grey-7")

        def _confirm_delete() -> None:
            delete_run_history_entry(run_id)
            dialog.close()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(tr("action_cancel"), on_click=dialog.close).props("flat")
            ui.button(tr("run_delete"), on_click=_confirm_delete).props("unelevated color=negative")

    dialog.open()


def open_clear_history_confirmation() -> None:
    if not state.get("run_history"):
        return

    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(tr("run_clear_history_confirm_title")).classes("text-subtitle1 text-weight-medium")
        ui.label(tr("run_clear_history_confirm_text")).classes("text-body2 text-grey-7")

        def _confirm_clear() -> None:
            clear_run_history()
            dialog.close()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(tr("action_cancel"), on_click=dialog.close).props("flat")
            ui.button(tr("run_clear_history"), on_click=_confirm_clear).props("unelevated color=negative")

    dialog.open()


def history_kind_filter_options() -> dict[str, str]:
    return {
        "all": tr("history_filter_all"),
        "replay_no_weather": tr("run_type_replay_no_weather"),
        "replay_weather": tr("run_type_replay_weather"),
        "future_no_weather": tr("run_type_future_no_weather"),
        "future_weather": tr("run_type_future_weather"),
    }


def history_sort_options() -> dict[str, str]:
    return {
        "newest": tr("history_sort_newest"),
        "oldest": tr("history_sort_oldest"),
        "speed_desc": tr("history_sort_speed_desc"),
        "windscore_best": tr("history_sort_windscore_best"),
    }


def set_history_sort_order(value: str) -> None:
    state["history_sort_order"] = value
    refresh_all()


def set_results_active_tab(value: str) -> None:
    if value in {"summary", "details", "history"}:
        state["results_active_tab"] = value


def get_filtered_history() -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = state.get("run_history", [])
    run_kind = str(state.get("history_kind_filter", "all"))
    if run_kind == "all":
        filtered = list(history)
    else:
        filtered = [record for record in history if str(record.get("kind", "")) == run_kind]

    sort_order = str(state.get("history_sort_order", "newest"))
    if sort_order == "oldest":
        filtered.sort(key=lambda record: str(record.get("created_at", "")), reverse=False)
    elif sort_order == "speed_desc":
        filtered.sort(
            key=lambda record: (
                _float_or_none(record.get("metrics", {}).get("speed_kmh")) or -1.0,
                str(record.get("created_at", "")),
            ),
            reverse=True,
        )
    elif sort_order == "windscore_best":
        filtered.sort(
            key=lambda record: (
                windscore_grade_rank(record.get("metrics", {}).get("windscore_grade")),
                str(record.get("created_at", "")),
            ),
            reverse=True,
        )
    else:
        filtered.sort(key=lambda record: str(record.get("created_at", "")), reverse=True)
    return filtered


def can_select_for_compare(run_id: str) -> bool:
    compare_ids: list[str] = list(state.get("compare_run_ids", []))
    return run_id in compare_ids or len(compare_ids) < 2


def windscore_grade_rank(value: Any) -> int:
    grade = str(value or "").strip().upper()
    mapping = {
        "A": 6,
        "B": 5,
        "C": 4,
        "D": 3,
        "E": 2,
        "F": 1,
    }
    return mapping.get(grade, 0)


def run_kind_chip_color(run_kind: str) -> str:
    colors = {
        "replay_no_weather": "primary",
        "replay_weather": "teal",
        "future_no_weather": "indigo",
        "future_weather": "orange",
    }
    return colors.get(run_kind, "grey")


def get_run_weather_model_label(record: dict[str, Any]) -> str:
    weather = record.get("weather", {}) if isinstance(record.get("weather", {}), dict) else {}
    enabled = bool(weather.get("enabled"))
    if not enabled:
        return tr("weather_disabled")

    model = str(weather.get("model") or "").strip().upper()
    if not model:
        params = record.get("params", {}) if isinstance(record.get("params", {}), dict) else {}
        model = str(params.get("weather_model") or "").strip().upper()
    if model == "NONE":
        return tr("weather_model_none")

    weather_roughness_raster = bool(weather.get("roughness_raster"))
    if weather_roughness_raster and model:
        return f"{model} + {tr('history_tag_raster')}"

    return model or tr("not_available")


def run_weather_chip_color(record: dict[str, Any]) -> str:
    weather = record.get("weather", {}) if isinstance(record.get("weather", {}), dict) else {}
    if not bool(weather.get("enabled")):
        return "grey"

    model = str(weather.get("model") or "").strip().upper()
    if model == "IFS":
        return "deep-orange"
    if model == "GFS":
        return "teal"
    if model == "NONE":
        return "grey"
    return "blue-grey"


def format_history_time(value: str | None) -> str:
    if not value:
        return tr("not_available")
    try:
        dt = datetime.fromisoformat(value)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


def format_duration_seconds(duration_s: float | None) -> str:
    if duration_s is None:
        return tr("not_available")
    total_seconds = int(duration_s)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_run_title(record: dict[str, Any]) -> str:
    kind = str(record.get("kind", ""))
    kind_label = tr(run_kind_label_key(kind))
    gpx_name = str(record.get("source_gpx", "")).strip() or tr("route_pending")
    return f"{kind_label} | {gpx_name}"


def get_run_title_with_weather(record: dict[str, Any]) -> str:
    weather_label = get_run_weather_model_label(record)
    return f"{get_run_title(record)} [{weather_label}]"


def get_run_reference_datetime(record: dict[str, Any]) -> str | None:
    kind = str(record.get("kind", ""))
    route = record.get("route", {}) if isinstance(record.get("route", {}), dict) else {}
    weather = record.get("weather", {}) if isinstance(record.get("weather", {}), dict) else {}
    params = record.get("params", {}) if isinstance(record.get("params", {}), dict) else {}

    if kind in {"replay_no_weather", "replay_weather"}:
        return route.get("start")
    if kind == "future_weather":
        return weather.get("start_utc") or params.get("sim_start")
    return record.get("created_at")


def get_run_reference_time_text(record: dict[str, Any]) -> str:
    return format_history_time(get_run_reference_datetime(record))


def _format_delta(a: float | None, b: float | None, unit: str, precision: int) -> str:
    if a is None or b is None:
        return tr("not_available")
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{precision}f} {unit}"


def _windscore_delta(a: str | None, b: str | None) -> str:
    if not a and not b:
        return tr("not_available")
    if not a:
        return f"{tr('not_available')} -> {b}"
    if not b:
        return f"{a} -> {tr('not_available')}"
    if a == b:
        return f"{a} = {b}"
    return f"{a} -> {b}"


def _format_metric(value: float | None, unit: str, precision: int) -> str:
    if value is None:
        return tr("not_available")
    return f"{value:.{precision}f} {unit}" if unit else f"{value:.{precision}f}"


def _coerce_float(raw_value: Any, fallback: float, min_value: float) -> float:
    try:
        value = float(raw_value)
        return value if value >= min_value else fallback
    except Exception:
        return fallback


def _coerce_int(raw_value: Any, fallback: int, allowed: set[int]) -> int:
    try:
        value = int(raw_value)
        return value if value in allowed else fallback
    except Exception:
        return fallback


def _coerce_bool(raw_value: Any, fallback: bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return fallback
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    text = str(raw_value).strip().lower()
    if text in {"1", "true", "yes", "on", "oui"}:
        return True
    if text in {"0", "false", "no", "off", "non"}:
        return False
    return fallback


def normalize_gui_settings(raw: dict[str, Any]) -> dict[str, Any]:
    model = str(raw.get("weather_model", DEFAULT_PERSISTED_PARAMS["weather_model"]))
    model = model.upper().strip()
    if model == "AUCUN":
        model = "NONE"
    if model not in {"GFS", "IFS", "NONE"}:
        model = DEFAULT_PERSISTED_PARAMS["weather_model"]

    grib_dir = str(raw.get("general_grib_dir", DEFAULT_PERSISTED_PARAMS["general_grib_dir"]))
    grib_dir = grib_dir.strip() or DEFAULT_PERSISTED_PARAMS["general_grib_dir"]

    cyclist_params_dir = str(raw.get("cyclist_params_dir", DEFAULT_PERSISTED_PARAMS["cyclist_params_dir"]))
    cyclist_params_dir = cyclist_params_dir.strip() or DEFAULT_PERSISTED_PARAMS["cyclist_params_dir"]

    return {
        "general_grib_dir": grib_dir,
        "cyclist_params_dir": cyclist_params_dir,
        "CdA": _coerce_float(raw.get("CdA"), DEFAULT_PERSISTED_PARAMS["CdA"], 0.05),
        "Cr": _coerce_float(raw.get("Cr"), DEFAULT_PERSISTED_PARAMS["Cr"], 0.0001),
        "mass_kg": _coerce_float(raw.get("mass_kg"), DEFAULT_PERSISTED_PARAMS["mass_kg"], 20.0),
        "weather_model": model,
        "weather_pas": _coerce_int(raw.get("weather_pas"), DEFAULT_PERSISTED_PARAMS["weather_pas"], {1, 3}),
        "weather_use_raster_roughness": _coerce_bool(raw.get("weather_use_raster_roughness"), DEFAULT_PERSISTED_PARAMS["weather_use_raster_roughness"]),
        "map_enable_animation": _coerce_bool(raw.get("map_enable_animation"), DEFAULT_PERSISTED_PARAMS["map_enable_animation"]),
        "map_distance_from_finish": _coerce_bool(raw.get("map_distance_from_finish"), DEFAULT_PERSISTED_PARAMS["map_distance_from_finish"]),
        "process_gpx_verbose": _coerce_bool(raw.get("process_gpx_verbose"), DEFAULT_PERSISTED_PARAMS["process_gpx_verbose"]),
    }


def collect_persisted_settings_from_state() -> dict[str, Any]:
    params = state["params"]
    return {
        "general_grib_dir": str(params["general_grib_dir"]),
        "cyclist_params_dir": str(params["cyclist_params_dir"]),
        "CdA": float(params["CdA"]),
        "Cr": float(params["Cr"]),
        "mass_kg": float(params["mass_kg"]),
        "weather_model": str(params["weather_model"]),
        "weather_pas": int(params["weather_pas"]),
        "weather_use_raster_roughness": bool(params["weather_use_raster_roughness"]),
        "map_enable_animation": bool(params["map_enable_animation"]),
        "map_distance_from_finish": bool(params["map_distance_from_finish"]),
        "process_gpx_verbose": bool(params["process_gpx_verbose"]),
    }


def apply_persisted_settings(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        state["params"][key] = value


def save_gui_settings(settings: dict[str, Any]) -> None:
    GUI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GUI_SETTINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)


def initialize_gui_settings() -> None:
    state["setup_form"] = dict(DEFAULT_PERSISTED_PARAMS)
    if not GUI_SETTINGS_PATH.exists():
        state["needs_setup"] = True
        return


    try:
        with GUI_SETTINGS_PATH.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise RuntimeError("settings file must contain a JSON object")
        settings = normalize_gui_settings(raw)
        apply_persisted_settings(settings)
        state["setup_form"] = dict(settings)
    except Exception as exc:
        state["needs_setup"] = True
        state["setup_error"] = str(exc)


def set_setup_field(key: str, value: Any) -> None:
    state["setup_form"][key] = value


def _default_profile_form() -> dict[str, Any]:
    params = state["params"]
    return {
        "CdA": _coerce_float(params.get("CdA"), DEFAULT_PERSISTED_PARAMS["CdA"], 0.05),
        "Cr": _coerce_float(params.get("Cr"), DEFAULT_PERSISTED_PARAMS["Cr"], 0.0001),
        "mass_kg": _coerce_float(params.get("mass_kg"), DEFAULT_PERSISTED_PARAMS["mass_kg"], 20.0),
        "profile_preset": str(params.get("profile_preset", "competitive")),
        "uphill_mode": str(params.get("uphill_mode", "realistic")),
        "downhill_mode": str(params.get("downhill_mode", "realistic")),
        "corner_mode": str(params.get("corner_mode", "realistic")),
        "behavior_thresholds": {},  # optional thresholds, empty by default
    }


def _sanitize_profile_form(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    base = raw or {}
    profile_preset = str(base.get("profile_preset", "competitive")).strip()
    if profile_preset not in PROFILE_PRESETS:
        profile_preset = "competitive"

    uphill = str(base.get("uphill_mode", "realistic")).strip()
    downhill = str(base.get("downhill_mode", "realistic")).strip()
    corner = str(base.get("corner_mode", "realistic")).strip()

    # Sanitize behavior thresholds (optional, can be empty)
    raw_thresholds = base.get("behavior_thresholds") or {}
    behavior_thresholds = {}
    if isinstance(raw_thresholds, dict):
        # Only keep valid threshold keys with float/int values
        valid_threshold_keys = {
            "uphill_facteur_forte", "uphill_facteur_moderee", "uphill_facteur_legere",
            "downhill_vitesse_max_absolue", "downhill_vitesse_reduction_factor", "downhill_vitesse_reduction_cap",
            "downhill_puissance_min", "downhill_facteur_forte", "downhill_facteur_legere",
            "downhill_corner_safety_factor",
            "corner_speed_straight", "corner_speed_slight", "corner_speed_moderate", "corner_speed_sharp", "corner_speed_hairpin"
        }
        for key, value in raw_thresholds.items():
            if key in valid_threshold_keys and value is not None and str(value).strip():
                try:
                    behavior_thresholds[key] = float(value)
                except (ValueError, TypeError):
                    pass  # Skip invalid values

    return {
        "CdA": _coerce_float(base.get("CdA"), DEFAULT_PERSISTED_PARAMS["CdA"], 0.05),
        "Cr": _coerce_float(base.get("Cr"), DEFAULT_PERSISTED_PARAMS["Cr"], 0.0001),
        "mass_kg": _coerce_float(base.get("mass_kg"), DEFAULT_PERSISTED_PARAMS["mass_kg"], 20.0),
        "profile_preset": profile_preset,
        "uphill_mode": uphill if uphill in MODE_LABEL_KEYS else "realistic",
        "downhill_mode": downhill if downhill in MODE_LABEL_KEYS else "realistic",
        "corner_mode": corner if corner in MODE_LABEL_KEYS else "realistic",
        "behavior_thresholds": behavior_thresholds,
    }


def set_profile_form_field(key: str, value: Any) -> None:
    profile_form = _sanitize_profile_form(state.get("profile_form"))
    profile_form[key] = value
    state["profile_form"] = _sanitize_profile_form(profile_form)


def set_profile_form_float_field(key: str, value: Any) -> None:
    if value is None:
        return
    set_profile_form_field(key, float(value))


def _set_behavior_threshold(key: str, value: Any) -> None:
    """Update a behavior threshold parameter in the profile_form."""
    profile_form = state.get("profile_form", {})
    if "behavior_thresholds" not in profile_form:
        profile_form["behavior_thresholds"] = {}
    
    if value is None or str(value).strip() == "":
        # Remove the key if empty
        profile_form["behavior_thresholds"].pop(key, None)
    else:
        try:
            profile_form["behavior_thresholds"][key] = float(value)
        except (ValueError, TypeError):
            pass
    
    state["profile_form"] = _sanitize_profile_form(profile_form)
    # Do NOT call refresh_all() here — it would close the expansion panel on every keystroke


def set_profile_form_preset(value: str) -> None:
    preset = value if value in PROFILE_PRESETS else "competitive"
    profile_form = _sanitize_profile_form(state.get("profile_form"))
    profile_form["profile_preset"] = preset
    modes = PRESET_TO_MODES[preset]
    profile_form["uphill_mode"] = modes["uphill"]
    profile_form["downhill_mode"] = modes["downhill"]
    profile_form["corner_mode"] = modes["corner"]
    state["profile_form"] = _sanitize_profile_form(profile_form)
    refresh_all()


def _cyclist_params_dir_path() -> Path:
    raw = str(state["params"].get("cyclist_params_dir", "")).strip()
    if not raw:
        raw = str(DEFAULT_PERSISTED_PARAMS["cyclist_params_dir"])
    return Path(raw).expanduser()


def _normalize_profile_filename(raw_name: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_name.strip())
    if not name:
        return ""
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    return name


def list_cyclist_profile_files() -> list[str]:
    try:
        directory = _cyclist_params_dir_path()
        if not directory.exists() or not directory.is_dir():
            return []
        return sorted(path.name for path in directory.glob("*.json") if path.is_file())
    except Exception:
        return []


def cyclist_profile_file_options() -> dict[str, str]:
    files = list_cyclist_profile_files()
    return {file_name: file_name for file_name in files}


def set_selected_cyclist_profile_file(value: Any) -> None:
    state["selected_cyclist_profile_file"] = str(value or "")


def _load_cyclist_profile_content(file_name: str) -> tuple[dict[str, Any], str]:
    path = _cyclist_params_dir_path() / file_name
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise RuntimeError("profile file must contain a JSON object")
    profile_form = _sanitize_profile_form(raw)
    profile_name = str(raw.get("name") or "").strip() or Path(file_name).stem
    return profile_form, profile_name


def _activate_loaded_profile(profile_form: dict[str, Any], profile_name: str, file_name: str) -> None:
    """Apply a loaded cyclist profile as the active profile for upcoming simulations."""
    clean_profile = _sanitize_profile_form(profile_form)
    params = state["params"]

    # Reset physics parameters for the next simulation.
    params["CdA"] = float(clean_profile["CdA"])
    params["Cr"] = float(clean_profile["Cr"])
    params["mass_kg"] = float(clean_profile["mass_kg"])

    # Reset behavior modes for the next simulation.
    params["profile_preset"] = str(clean_profile["profile_preset"])
    params["uphill_mode"] = str(clean_profile["uphill_mode"])
    params["downhill_mode"] = str(clean_profile["downhill_mode"])
    params["corner_mode"] = str(clean_profile["corner_mode"])

    # Keep active advanced thresholds separate from editor state.
    state["active_behavior_thresholds"] = dict(clean_profile.get("behavior_thresholds", {}))

    state["selected_cyclist_profile_file"] = file_name
    state["cyclist_profile_name"] = profile_name
    state["profile_form"] = dict(clean_profile)


def load_selected_profile_into_form(file_name: str | None = None) -> bool:
    selected = str(file_name or state.get("selected_cyclist_profile_file") or "").strip()
    if not selected:
        ui.notify(tr("cyclist_profile_select_first"), type="warning")
        return False

    path = _cyclist_params_dir_path() / selected
    if not path.exists():
        ui.notify(tr("cyclist_profile_not_found", name=selected), type="warning")
        return False

    try:
        profile_form, profile_name = _load_cyclist_profile_content(selected)
        state["profile_form"] = profile_form
        state["selected_cyclist_profile_file"] = selected
        state["cyclist_profile_name"] = profile_name
        ui.notify(tr("cyclist_profile_edit_loaded", name=selected), type="positive")
        refresh_all()
        return True
    except Exception as exc:
        ui.notify(tr("cyclist_profile_load_error", error=exc), type="negative")
        return False


def load_cyclist_profile_file(file_name: str | None = None) -> None:
    selected = str(file_name or state.get("selected_cyclist_profile_file") or "").strip()
    file_name = selected
    if not file_name:
        ui.notify(tr("cyclist_profile_select_first"), type="warning")
        return

    path = _cyclist_params_dir_path() / file_name
    if not path.exists():
        ui.notify(tr("cyclist_profile_not_found", name=file_name), type="warning")
        return

    try:
        profile_form, profile_name = _load_cyclist_profile_content(file_name)
        _activate_loaded_profile(profile_form, profile_name, file_name)
        persist_runtime_defaults()
        ui.notify(tr("cyclist_profile_loaded", name=file_name), type="positive")
        refresh_all()
    except Exception as exc:
        ui.notify(tr("cyclist_profile_load_error", error=exc), type="negative")


def save_cyclist_profile_file() -> None:
    raw_name = str(state.get("cyclist_profile_name") or "").strip()
    if not raw_name:
        ui.notify(tr("cyclist_profile_name_required"), type="warning")
        return

    file_name = _normalize_profile_filename(raw_name)
    if not file_name:
        ui.notify(tr("cyclist_profile_name_required"), type="warning")
        return

    profile_form = _sanitize_profile_form(state.get("profile_form"))
    data = {
        "name": raw_name,
        "CdA": float(profile_form["CdA"]),
        "Cr": float(profile_form["Cr"]),
        "mass_kg": float(profile_form["mass_kg"]),
        "profile_preset": str(profile_form["profile_preset"]),
        "uphill_mode": str(profile_form["uphill_mode"]),
        "downhill_mode": str(profile_form["downhill_mode"]),
        "corner_mode": str(profile_form["corner_mode"]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Only add behavior_thresholds if non-empty (optional)
    if profile_form.get("behavior_thresholds"):
        data["behavior_thresholds"] = dict(profile_form["behavior_thresholds"])

    try:
        directory = _cyclist_params_dir_path()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / file_name
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        state["selected_cyclist_profile_file"] = file_name
        ui.notify(tr("cyclist_profile_saved", name=file_name), type="positive")
        refresh_all()
    except Exception as exc:
        ui.notify(tr("cyclist_profile_save_error", error=exc), type="negative")


def delete_cyclist_profile_file(file_name: str) -> None:
    path = _cyclist_params_dir_path() / str(file_name)
    try:
        if not path.exists():
            ui.notify(tr("cyclist_profile_not_found", name=file_name), type="warning")
            return
        path.unlink()
        if str(state.get("selected_cyclist_profile_file") or "") == str(file_name):
            state["selected_cyclist_profile_file"] = ""
        ui.notify(tr("cyclist_profile_deleted", name=file_name), type="positive")
        refresh_all()
    except Exception as exc:
        ui.notify(tr("cyclist_profile_delete_error", error=exc), type="negative")


def open_delete_cyclist_profile_confirmation(file_name: str) -> None:
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(tr("cyclist_profile_delete_confirm_title")).classes("text-subtitle1 text-weight-medium")
        ui.label(file_name).classes("text-body2 text-grey-8")
        ui.label(tr("cyclist_profile_delete_confirm_text")).classes("text-body2 text-grey-7")

        def _confirm_delete() -> None:
            delete_cyclist_profile_file(file_name)
            dialog.close()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(tr("action_cancel"), on_click=dialog.close).props("flat")
            ui.button(tr("run_delete"), on_click=_confirm_delete).props("unelevated color=negative")

    dialog.open()


def open_profile_manager() -> None:
    state["profile_manager_open"] = True
    selected = str(state.get("selected_cyclist_profile_file") or "").strip()
    files = list_cyclist_profile_files()
    if selected and selected in files:
        load_selected_profile_into_form(selected)
        return
    if files:
        state["selected_cyclist_profile_file"] = files[0]
        load_selected_profile_into_form(files[0])
        return
    state["profile_form"] = _default_profile_form()
    state["cyclist_profile_name"] = ""
    refresh_all()


def close_profile_manager() -> None:
    state["profile_manager_open"] = False
    refresh_all()


def new_profile_form() -> None:
    state["selected_cyclist_profile_file"] = ""
    state["cyclist_profile_name"] = ""
    state["profile_form"] = _default_profile_form()
    refresh_all()


def persist_runtime_defaults() -> None:
    settings = normalize_gui_settings(collect_persisted_settings_from_state())
    save_gui_settings(settings)


def create_initial_settings() -> None:
    try:
        settings = normalize_gui_settings(state["setup_form"])
        save_gui_settings(settings)
        apply_persisted_settings(settings)
        state["setup_form"] = dict(settings)
        state["needs_setup"] = False
        state["setup_error"] = None
        ui.notify(tr("setup_saved"), type="positive")
        refresh_all()
    except Exception as exc:
        ui.notify(tr("setup_invalid", error=exc), type="negative")


def open_setup_editor() -> None:
    state["setup_error"] = None
    state["setup_form"] = normalize_gui_settings(collect_persisted_settings_from_state())
    if GUI_SETTINGS_PATH.exists():
        try:
            with GUI_SETTINGS_PATH.open("r", encoding="utf-8") as file:
                raw = json.load(file)
            if isinstance(raw, dict):
                state["setup_form"] = normalize_gui_settings(raw)
        except Exception as exc:
            state["setup_error"] = str(exc)
    state["needs_setup"] = True
    refresh_all()


def close_setup_editor() -> None:
    if GUI_SETTINGS_PATH.exists():
        state["needs_setup"] = False
        state["setup_error"] = None
        refresh_all()


def tr(key: str, **kwargs: Any) -> str:
    lang = state["language"]
    bundle = TRANSLATIONS.get(key, {})
    template = bundle.get(lang) or bundle.get("en") or key
    return template.format(**kwargs) if kwargs else template


def format_duration(duration: Any) -> str:
    if duration is None:
        return tr("not_available")
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def behavior_mode_options() -> dict[str, str]:
    return {mode: tr(label_key) for mode, label_key in MODE_LABEL_KEYS.items()}


def profile_preset_options() -> dict[str, str]:
    return {
        "conservative": tr("profile_conservative"),
        "competitive": tr("profile_competitive"),
        "pro": tr("profile_pro"),
    }


def apply_profile_preset(preset: str) -> None:
    state["params"]["profile_preset"] = preset
    modes = PRESET_TO_MODES[preset]
    state["params"]["uphill_mode"] = modes["uphill"]
    state["params"]["downhill_mode"] = modes["downhill"]
    state["params"]["corner_mode"] = modes["corner"]


def build_behavior() -> CyclistBehavior:
    params = state["params"]

    # Always rebuild behavior from modes to ensure clean reset before each simulation.
    behavior = CyclistBehavior(
        uphill=str(params.get("uphill_mode", "realistic")),
        downhill=str(params.get("downhill_mode", "realistic")),
        corner=str(params.get("corner_mode", "realistic")),
    )

    # Apply optional custom behavior thresholds from the active loaded profile.
    _apply_behavior_thresholds(behavior, state.get("active_behavior_thresholds", {}))
    # DEBUG JA 
    #behavior.display()
    return behavior


def _apply_behavior_thresholds(behavior: CyclistBehavior, thresholds: dict[str, float]) -> None:
    """Apply optional custom threshold parameters to CyclistBehavior instance."""
    if not thresholds:
        return
    
    # List of customizable threshold attributes with their valid ranges
    # JA "downhill_vitesse_reduction_factor" changed  to (0.Z, 10.0) to allow lower values
    # JA corner_speed_hairpin changed to (1.0, 14.0) to allow lower values
    # compliance xit create_pro_file of biwipy
    customizable_attrs = {
        "uphill_facteur_forte": (1.0, 10.0),
        "uphill_facteur_moderee": (1.0, 10.0),
        "uphill_facteur_legere": (1.0, 10.0),
        "downhill_vitesse_max_absolue": (5.0, 30.0),  # m/s
        "downhill_vitesse_reduction_factor": (0.3, 10.0),
        "downhill_vitesse_reduction_cap": (0.0, 1.0),
        "downhill_puissance_min": (1, 50),  # watts
        "downhill_facteur_forte": (1.0, 50.0),
        "downhill_facteur_legere": (1.0, 50.0),
        "downhill_corner_safety_factor": (0.5, 1.0),
        "corner_speed_straight": (5.0, 30.0),  # m/s
        "corner_speed_slight": (5.0, 30.0),
        "corner_speed_moderate": (5.0, 30.0),
        "corner_speed_sharp": (1.0, 20.0),
        "corner_speed_hairpin": (1.0, 14.0),
    }
    
    for key, value in thresholds.items():
        if key not in customizable_attrs or not hasattr(behavior, key):
            continue
        
        min_val, max_val = customizable_attrs[key]
        try:
            # Clamp value within valid range
            clamped_value = max(min_val, min(max_val, float(value)))
            setattr(behavior, key, clamped_value)
        except (ValueError, TypeError):
            pass  # Skip invalid values


def set_busy(message: str | None) -> None:
    state["is_busy"] = bool(message)
    state["busy_message"] = message or ""
    refresh_all()


async def run_replay() -> None:
    gpx_result = state.get("gpx_result")
    params = state["params"]
    if state.get("is_busy"):
        return
    if gpx_result is None:
        ui.notify(tr("replay_needs_gpx"), type="warning")
        return
    if not gpx_result.has_timestamps:
        ui.notify(tr("replay_needs_timestamps"), type="warning")
        return

    _clear_plots()
    set_busy(tr("busy_replay"))
    await asyncio.sleep(0)

    try:
        sim = Simulator(
            grib=None,
            behavior=build_behavior(),
            CdA=float(params["CdA"]),
            Cr=float(params["Cr"]),
            m=float(params["mass_kg"]),
        )
        result = await asyncio.to_thread(
            sim.simulate_replay,
            gpx_result.segments,
            t_start=gpx_result.t_start,
            velocity_smooth_window=REPLAY_SMOOTHING_WINDOW,
        )
    except Exception as exc:
        state["simulation_result"] = None
        state["simulation_result_run_id"] = None
        state["simulation_markdown"] = f"### {tr('replay_error')}\n\n`{exc}`"
    else:
        apply_replay_calibrated_p0_defaults(result)
        state["simulation_result"] = result
        state["simulation_markdown"] = "\n".join(
            [
                f"### {tr('replay_done')}",
                "",
                tr(
                    "params_used",
                    cda=float(params["CdA"]),
                    cr=float(params["Cr"]),
                    mass=float(params["mass_kg"]),
                ),
                f"- {tr('kpi_speed')}: **{result.speed.avg:.2f} km/h**",
                f"- {tr('kpi_power')}: **{result.power.avg:.1f} W**" if result.power else f"- {tr('kpi_power')}: {tr('not_available')}",
                f"- {tr('kpi_p0')}: **{result.power.P0_calibrated:.1f} W**" if result.power else f"- {tr('kpi_p0')}: {tr('not_available')}",
            ]
        )
        add_run_to_history(
            build_run_record(
                "replay_no_weather",
                result,
                gpx_result,
                weather_context={"enabled": False},
                target_context={"mode": "replay"},
            )
        )
        _cache_run_segments(str(state.get("selected_run_id") or ""), result)
        state["simulation_result_run_id"] = str(state.get("selected_run_id") or "")
    finally:
        set_busy(None)

    refresh_all()


def simulation_target_options() -> dict[str, str]:
    return {
        "p0": tr("sim_mode_p0"),
        "v0": tr("sim_mode_v0"),
    }


def get_future_target_kwargs() -> tuple[dict[str, float], str]:
    params = state["params"]
    mode = str(params["sim_target_mode"])
    if mode == "v0":
        v0_kmh = float(params["sim_v0_kmh"])
        v0_ms = v0_kmh / 3.6
        return {"v0": v0_ms}, tr("sim_target_used_v0", value=v0_kmh)

    p0 = float(params["sim_p0"])
    return {"P0": p0}, tr("sim_target_used_p0", value=p0)


def capture_summary_statistics(result: Any, label: str = "") -> str:
    """Run print_summary_statistics and capture its stdout as a string."""
    buffer = io.StringIO()
    previous_output_lang = os.environ.get("OUTPUT_LANG")
    ui_lang = str(state.get("language") or "en").strip().lower()
    os.environ["OUTPUT_LANG"] = "fr" if ui_lang.startswith("fr") else "en"
    try:
        with contextlib.redirect_stdout(buffer):
            print_summary_statistics(result, label=label or "Run")
    except Exception as exc:
        return f"[Erreur print_summary_statistics: {exc}]"
    finally:
        if previous_output_lang is None:
            os.environ.pop("OUTPUT_LANG", None)
        else:
            os.environ["OUTPUT_LANG"] = previous_output_lang
    return buffer.getvalue()


def get_active_run_statistics_text(active_run: dict[str, Any] | None) -> str:
    """Return detailed stats in the current UI language when source data is available."""
    if active_run is None:
        return ""

    run_id = str(active_run.get("id") or "")
    label = str(active_run.get("source_gpx") or "Run")

    simulation_result = state.get("simulation_result")
    simulation_result_run_id = str(state.get("simulation_result_run_id") or "")
    if simulation_result is not None and run_id and run_id == simulation_result_run_id:
        return capture_summary_statistics(simulation_result, label=label)

    cached_segments = state.get("segments_cache", {}).get(run_id)
    if cached_segments:
        return capture_summary_statistics(cached_segments, label=label)

    return str(active_run.get("summary_statistics") or "")


def _clear_plots() -> None:
    state["plot_wind_rose_img"] = None
    state["plot_elevation_img"] = None
    state["plot_speed_wind_img"] = None
    state["plots_gpx_name"] = None
    state["plot_compare_img"] = None


def fig_to_base64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _get_smoothed_segments() -> list | None:
    result = state.get("simulation_result")
    if result is None:
        return None
    try:
        raw = result.get_segments()
        window = int(state["params"].get("smoothing_window", 17))
        return smooth_segments(raw, window=window)
    except Exception:
        return None


def _get_active_run_segments() -> list | None:
    active_run = get_active_run()
    active_run_id = str(active_run.get("id")) if active_run is not None else ""

    if active_run_id:
        cached = state.get("segments_cache", {}).get(active_run_id)
        if cached:
            return cached

    result = state.get("simulation_result")
    if result is None:
        return None

    try:
        raw = result.get_segments()
        if active_run_id:
            state["segments_cache"][active_run_id] = raw
        return raw
    except Exception:
        return None


async def generate_interactive_map_for_active_run() -> None:
    active_run = get_active_run()
    if active_run is None:
        ui.notify(tr("interactive_map_no_run"), type="warning")
        return

    segments = await asyncio.to_thread(_get_active_run_segments)
    if not segments:
        ui.notify(tr("interactive_map_no_segments"), type="warning")
        return

    run_id = str(active_run.get("id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    source_gpx = str(active_run.get("source_gpx") or "route").strip() or "route"
    safe_gpx = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_gpx)
    filename = f"interactive_map_{safe_gpx}_{run_id}.html"
    output_path = GENERATED_MAPS_DIR / filename

    title = tr("interactive_map_title", name=source_gpx)

    try:
        enable_animation = bool(state["params"].get("map_enable_animation", False))
        distance_from_finish = bool(state["params"].get("map_distance_from_finish", False))
        await asyncio.to_thread(
            create_interactive_map,
            segments,
            str(output_path),
            title,
            enable_animation,
            distance_from_finish,
        )
    except Exception as exc:
        ui.notify(tr("interactive_map_error", error=exc), type="negative")
        return

    url = f"{GENERATED_MAPS_ROUTE}/{filename}"
    state["interactive_map_url"] = url
    state["interactive_map_run_id"] = run_id
    state["interactive_map_title"] = title

    ui.notify(tr("interactive_map_generated"), type="positive")
    _open_url_in_new_tab(url)
    refresh_all()


def _open_url_in_new_tab(url: str) -> None:
    ui.run_javascript(f"window.open({json.dumps(url)}, '_blank')")


async def _generate_plot(
    plot_key: str,
    plot_fn: Any,
    **kwargs: Any,
) -> None:
    segs = await asyncio.to_thread(_get_smoothed_segments)
    if segs is None:
        ui.notify(tr("plot_no_result"), type="warning")
        return

    gpx_name = state.get("gpx_name") or "Run"

    def _render() -> str | None:
        try:
            fig, _ = plot_fn(segs, **kwargs)
            if fig is None:
                return None
            return fig_to_base64(fig)
        except Exception as exc:
            return f"ERROR:{exc}"

    img = await asyncio.to_thread(_render)
    if img is None:
        ui.notify(tr("plot_unavailable"), type="warning")
        return
    if isinstance(img, str) and img.startswith("ERROR:"):
        ui.notify(tr("plot_error", error=img[6:]), type="negative")
        return
    state[plot_key] = img
    state["plots_gpx_name"] = gpx_name
    refresh_all()


async def generate_plot_wind_rose() -> None:
    gpx_name = state.get("gpx_name") or "Run"
    await _generate_plot(
        "plot_wind_rose_img",
        plot_wind_rose,
        title=tr("plot_wind_rose_title", name=gpx_name),
    )


async def generate_plot_elevation() -> None:
    gpx_name = state.get("gpx_name") or "Run"
    await _generate_plot(
        "plot_elevation_img",
        plot_elevation_profile,
        title=tr("plot_elevation_title", name=gpx_name),
    )


async def generate_plot_speed_wind() -> None:
    gpx_name = state.get("gpx_name") or "Run"
    await _generate_plot(
        "plot_speed_wind_img",
        plot_segments_evolution,
        attributes=["speed_m_s", "wind_along"],
        x_axis="distance",
        title=tr("plot_speed_wind_title", name=gpx_name),
    )


def get_compare_runs_compatibility() -> tuple[bool, str]:
    """Returns (can_plot, reason_key). Checks GPX compatibility and cache availability."""
    compare_ids: list[str] = list(state.get("compare_run_ids", []))
    if len(compare_ids) != 2:
        return False, "compare_need_two"
    left = get_run_by_id(compare_ids[0])
    right = get_run_by_id(compare_ids[1])
    if left is None or right is None:
        return False, "compare_need_two"
    gpx_a = str(left.get("source_gpx") or "").strip()
    gpx_b = str(right.get("source_gpx") or "").strip()
    if not gpx_a or not gpx_b or gpx_a != gpx_b:
        return False, "compare_plot_different_gpx"
    dist_a = _float_or_none(left.get("metrics", {}).get("distance_km"))
    dist_b = _float_or_none(right.get("metrics", {}).get("distance_km"))
    if dist_a is None or dist_b is None:
        return False, "compare_plot_no_distance"
    if abs(dist_a - dist_b) > 0.5:
        return False, "compare_plot_distance_mismatch"
    cache = state.get("segments_cache", {})
    missing = []
    if compare_ids[0] not in cache:
        missing.append("A")
    if compare_ids[1] not in cache:
        missing.append("B")
    if missing:
        return False, "compare_plot_segments_not_in_cache"
    return True, ""


async def generate_plot_compare() -> None:
    can_plot, reason_key = get_compare_runs_compatibility()
    if not can_plot:
        ui.notify(tr(reason_key), type="warning")
        return

    compare_ids = list(state.get("compare_run_ids", []))
    left = get_run_by_id(compare_ids[0])
    right = get_run_by_id(compare_ids[1])

    window = int(state["params"].get("smoothing_window", 17))
    label_a = get_run_title_with_weather(left) if left is not None else "A"
    label_b = get_run_title_with_weather(right) if right is not None else "B"
    title = tr("plot_compare_title", a=label_a[:30], b=label_b[:30])

    cache = state.get("segments_cache", {})
    segs_a = cache.get(compare_ids[0])
    segs_b = cache.get(compare_ids[1])

    def _render() -> str | None:
        try:
            smoothed_a = smooth_segments(segs_a, window=window)
            smoothed_b = smooth_segments(segs_b, window=window)
            fig, _ = compare_scenarios(
                [smoothed_a, smoothed_b],
                labels=[label_a, label_b],
                attribute="speed_m_s",
                x_axis="distance",
                title=title,
            )
            if fig is None:
                return None
            return fig_to_base64(fig)
        except Exception as exc:
            return f"ERROR:{exc}"

    img = await asyncio.to_thread(_render)
    if img is None:
        ui.notify(tr("plot_unavailable"), type="warning")
        return
    if isinstance(img, str) and img.startswith("ERROR:"):
        ui.notify(tr("plot_error", error=img[6:]), type="negative")
        return
    state["plot_compare_img"] = img
    refresh_all()


def apply_replay_calibrated_p0_defaults(result: Any) -> None:
    power = getattr(result, "power", None)
    if power is None:
        return

    p0_calibrated = _float_or_none(getattr(power, "P0_calibrated", None))
    if p0_calibrated is None or p0_calibrated <= 0:
        return

    # Replay calibration should become the default target for next weather simulation.
    state["params"]["sim_target_mode"] = "p0"
    state["params"]["sim_p0"] = p0_calibrated


def get_equivalent_target_text() -> str:
    params = state["params"]
    try:
        sim = Simulator(
            None,
            CdA=float(params["CdA"]),
            Cr=float(params["Cr"]),
            m=float(params["mass_kg"]),
        )
        mode = str(params["sim_target_mode"])
        if mode == "v0":
            p0 = sim.P0_from_v0(float(params["sim_v0_kmh"]) / 3.6)
            return tr("sim_equivalent_p0", value=p0)
        v0_kmh = sim.v0_from_P0(float(params["sim_p0"])) * 3.6
        return tr("sim_equivalent_v0", value=v0_kmh)
    except Exception:
        return tr("sim_equivalent_na")


def parse_start_datetime(raw_value: str) -> datetime:
    start_dt = datetime.fromisoformat(raw_value)
    if start_dt.tzinfo is None:
        # Treat naive datetime as local time, convert to UTC
        start_dt = start_dt.astimezone(timezone.utc)
    else:
        start_dt = start_dt.astimezone(timezone.utc)
    return start_dt


async def run_no_weather_simulation() -> None:
    gpx_result = state.get("gpx_result")
    params = state["params"]
    if state.get("is_busy"):
        return
    if gpx_result is None:
        ui.notify(tr("replay_needs_gpx"), type="warning")
        return

    _clear_plots()
    set_busy(tr("busy_future_no_weather"))
    await asyncio.sleep(0)

    try:
        start_dt = parse_start_datetime(str(params["sim_start"]))
        target_kwargs, target_label = get_future_target_kwargs()

        sim = Simulator(
            grib=None,
            behavior=build_behavior(),
            CdA=float(params["CdA"]),
            Cr=float(params["Cr"]),
            m=float(params["mass_kg"]),
        )
        result = await asyncio.to_thread(
            sim.simulate_future,
            gpx_result.segments,
            t_start=start_dt,
            **target_kwargs,
        )
    except Exception as exc:
        state["simulation_result"] = None
        state["simulation_result_run_id"] = None
        state["simulation_markdown"] = f"### {tr('future_no_weather_error')}\n\n`{exc}`"
    else:
        state["simulation_result"] = result
        state["simulation_markdown"] = "\n".join(
            [
                f"### {tr('future_no_weather_done')}",
                "",
                f"- {tr('weather_disabled')}",
                f"- {target_label}",
                tr(
                    "params_used",
                    cda=float(params["CdA"]),
                    cr=float(params["Cr"]),
                    mass=float(params["mass_kg"]),
                ),
                f"- {tr('kpi_speed')}: **{result.speed.avg:.2f} km/h**",
                f"- {tr('kpi_power')}: **{result.power.avg:.1f} W**" if result.power else f"- {tr('kpi_power')}: {tr('not_available')}",
                f"- {tr('kpi_p0')}: **{result.power.P0_calibrated:.1f} W**" if result.power else f"- {tr('kpi_p0')}: {tr('not_available')}",
            ]
        )
        add_run_to_history(
            build_run_record(
                "future_no_weather",
                result,
                gpx_result,
                weather_context={"enabled": False},
                target_context={
                    "mode": params["sim_target_mode"],
                    "p0": float(params["sim_p0"]),
                    "v0_kmh": float(params["sim_v0_kmh"]),
                    "label": target_label,
                },
            )
        )
        _cache_run_segments(str(state.get("selected_run_id") or ""), result)
        state["simulation_result_run_id"] = str(state.get("selected_run_id") or "")
    finally:
        set_busy(None)

    refresh_all()


async def run_weather_simulation() -> None:
    gpx_result = state.get("gpx_result")
    params = state["params"]
    if state.get("is_busy"):
        return
    if gpx_result is None:
        ui.notify(tr("replay_needs_gpx"), type="warning")
        return

    _clear_plots()
    set_busy(tr("busy_weather_future"))
    await asyncio.sleep(0)

    try:
        start_dt = parse_start_datetime(str(params["sim_start"]))
        duration_h = int(params["sim_duration_h"])
        pas_raw = int(params["weather_pas"])
        pas = cast(Literal[1, 3], pas_raw if pas_raw in (1, 3) else 1)
        model = str(params["weather_model"]).upper()
        grib_dir = str(params["general_grib_dir"]).strip() or "."
        target_kwargs, target_label = get_future_target_kwargs()

        gribs_list = await asyncio.to_thread(
            build_grib_list,
            grib_dir,
            start_dt,
            pas=pas,
            intervalle_h=duration_h,
            model=model,
        )
        if not gribs_list:
            raise RuntimeError(tr("weather_no_grib"))

        weather = await asyncio.to_thread(WeatherProvider, gribs_list, model=model)
        await asyncio.to_thread(_configure_roughness_provider, weather, gpx_result)
        roughness_raster_used = bool(getattr(weather.grib, "roughness_provider", None))
        sim = Simulator(
            weather.grib,
            behavior=build_behavior(),
            CdA=float(params["CdA"]),
            Cr=float(params["Cr"]),
            m=float(params["mass_kg"]),
        )
        result = await asyncio.to_thread(
            sim.simulate_future,
            gpx_result.segments,
            t_start=start_dt,
            **target_kwargs,
        )
    except Exception as exc:
        state["simulation_result"] = None
        state["simulation_result_run_id"] = None
        state["simulation_markdown"] = f"### {tr('weather_error')}\n\n`{exc}`"
    else:
        state["simulation_result"] = result
        state["simulation_markdown"] = "\n".join(
            [
                f"### {tr('weather_done')}",
                "",
                tr("weather_context", model=model, count=len(gribs_list), start=start_dt.isoformat(), duration=duration_h),
                f"- {target_label}",
                tr(
                    "params_used",
                    cda=float(params["CdA"]),
                    cr=float(params["Cr"]),
                    mass=float(params["mass_kg"]),
                ),
                f"- {tr('kpi_speed')}: **{result.speed.avg:.2f} km/h**",
                f"- {tr('kpi_power')}: **{result.power.avg:.1f} W**" if result.power else f"- {tr('kpi_power')}: {tr('not_available')}",
                f"- {tr('kpi_p0')}: **{result.power.P0_calibrated:.1f} W**" if result.power else f"- {tr('kpi_p0')}: {tr('not_available')}",
            ]
        )
        add_run_to_history(
            build_run_record(
                "future_weather",
                result,
                gpx_result,
                weather_context={
                    "enabled": True,
                    "model": model,
                    "roughness_raster": roughness_raster_used,
                    "pas": pas,
                    "grib_count": len(gribs_list),
                    "start_utc": start_dt.isoformat(),
                    "duration_h": duration_h,
                },
                target_context={
                    "mode": params["sim_target_mode"],
                    "p0": float(params["sim_p0"]),
                    "v0_kmh": float(params["sim_v0_kmh"]),
                    "label": target_label,
                },
            )
        )
        _cache_run_segments(str(state.get("selected_run_id") or ""), result)
        state["simulation_result_run_id"] = str(state.get("selected_run_id") or "")
    finally:
        set_busy(None)

    refresh_all()


def infer_interval_hours_from_gpx(gpx_result: Any) -> int:
    duration = getattr(gpx_result, "duration", None)
    if duration is None:
        return 4
    hours = duration.total_seconds() / 3600.0
    return max(1, int(math.ceil(hours)) + 1)


async def run_replay_with_weather() -> None:
    gpx_result = state.get("gpx_result")
    params = state["params"]

    if state.get("is_busy"):
        return
    if gpx_result is None:
        ui.notify(tr("replay_needs_gpx"), type="warning")
        return
    if not gpx_result.has_timestamps:
        ui.notify(tr("replay_needs_timestamps"), type="warning")
        return

    _clear_plots()
    set_busy(tr("busy_weather_replay"))
    await asyncio.sleep(0)

    try:
        model = str(params["weather_model"]).upper()
        pas_raw = int(params["weather_pas"])
        pas = cast(Literal[1, 3], pas_raw if pas_raw in (1, 3) else 1)
        grib_dir = str(params["general_grib_dir"]).strip() or "."
        start_dt = gpx_result.t_start
        interval_h = infer_interval_hours_from_gpx(gpx_result)

        gribs_list = await asyncio.to_thread(
            build_grib_list,
            grib_dir,
            start_dt,
            pas=pas,
            intervalle_h=interval_h,
            model=model,
        )
        if not gribs_list:
            raise RuntimeError(tr("weather_no_grib"))

        weather = await asyncio.to_thread(WeatherProvider, gribs_list, model=model)
        await asyncio.to_thread(_configure_roughness_provider, weather, gpx_result)
        roughness_raster_used = bool(getattr(weather.grib, "roughness_provider", None))
        sim = Simulator(
            weather.grib,
            behavior=build_behavior(),
            CdA=float(params["CdA"]),
            Cr=float(params["Cr"]),
            m=float(params["mass_kg"]),
        )
        result = await asyncio.to_thread(
            sim.simulate_replay,
            gpx_result.segments,
            t_start=gpx_result.t_start,
            velocity_smooth_window=REPLAY_SMOOTHING_WINDOW,
        )
    except Exception as exc:
        state["simulation_result"] = None
        state["simulation_result_run_id"] = None
        state["simulation_markdown"] = f"### {tr('weather_replay_error')}\n\n`{exc}`"
    else:
        apply_replay_calibrated_p0_defaults(result)
        state["simulation_result"] = result
        state["simulation_markdown"] = "\n".join(
            [
                f"### {tr('weather_replay_done')}",
                "",
                tr("weather_context", model=model, count=len(gribs_list), start=start_dt.isoformat(), duration=interval_h),
                tr(
                    "params_used",
                    cda=float(params["CdA"]),
                    cr=float(params["Cr"]),
                    mass=float(params["mass_kg"]),
                ),
                f"- {tr('kpi_speed')}: **{result.speed.avg:.2f} km/h**",
                f"- {tr('kpi_power')}: **{result.power.avg:.1f} W**" if result.power else f"- {tr('kpi_power')}: {tr('not_available')}",
                f"- {tr('kpi_p0')}: **{result.power.P0_calibrated:.1f} W**" if result.power else f"- {tr('kpi_p0')}: {tr('not_available')}",
            ]
        )
        add_run_to_history(
            build_run_record(
                "replay_weather",
                result,
                gpx_result,
                weather_context={
                    "enabled": True,
                    "model": model,
                    "roughness_raster": roughness_raster_used,
                    "pas": pas,
                    "grib_count": len(gribs_list),
                    "start_utc": start_dt.isoformat(),
                    "duration_h": interval_h,
                },
                target_context={"mode": "replay"},
            )
        )
        _cache_run_segments(str(state.get("selected_run_id") or ""), result)
        state["simulation_result_run_id"] = str(state.get("selected_run_id") or "")
    finally:
        set_busy(None)

    refresh_all()


def _configure_roughness_provider(weather: WeatherProvider, gpx_result: Any) -> None:
    """Attach an optional roughness provider to the active GRIB instance."""
    if not bool(state["params"].get("weather_use_raster_roughness", False)):
        weather.grib.roughness_provider = None
        return

    stats = getattr(gpx_result, "stats", {}) if gpx_result is not None else {}
    rect_sn = stats.get("rectangle_SN") if isinstance(stats, dict) else None
    rect_ew = stats.get("rectangle_EW") if isinstance(stats, dict) else None
    if not rect_sn or not rect_ew:
        weather.grib.roughness_provider = None
        return

    try:
        south, north = int(rect_sn[0]), int(rect_sn[1])
        west, east = int(rect_ew[0]), int(rect_ew[1])
        raster_root = Path(str(state["params"].get("general_grib_dir") or ".").strip() or ".") / "roughness_cache"
        provider = RoughnessProvider(raster_dir=str(raster_root))
        ready = provider.prepare(
            south=south,
            north=north,
            west=west,
            east=east,
            verbose=bool(state["params"].get("process_gpx_verbose", True)),
        )
        weather.grib.roughness_provider = provider if ready else None
    except Exception:
        weather.grib.roughness_provider = None


def _weather_enabled_for_actions() -> bool:
    model = str(state["params"].get("weather_model") or "").strip().upper()
    return model not in {"", "NONE"}


async def run_replay_action() -> None:
    if _weather_enabled_for_actions():
        await run_replay_with_weather()
        return
    await run_replay()


async def run_simulation_action() -> None:
    if _weather_enabled_for_actions():
        await run_weather_simulation()
        return
    await run_no_weather_simulation()


async def on_gpx_upload(event: events.UploadEventArguments) -> None:
    analyzer = RouteAnalyzer()
    temp_path = ""

    try:
        uploaded_bytes = await extract_uploaded_bytes(event)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as temp_file:
            temp_file.write(uploaded_bytes)
            temp_path = temp_file.name

        gpx_result = analyzer.process_gpx(
            temp_path,
            verbose=bool(state["params"].get("process_gpx_verbose", True)),
        )
        upload_name = extract_uploaded_name(event)

        state["gpx_result"] = gpx_result
        state["gpx_source_result"] = gpx_result
        state["gpx_source_name"] = upload_name
        state["gpx_name"] = upload_name
        state["simulation_result"] = None
        state["simulation_result_run_id"] = None
        state["simulation_markdown"] = ""
        state["status_message"] = tr("route_loaded", name=upload_name)

        ui.notify(tr("analysis_ok"), type="positive")
        refresh_all()
    except Exception as exc:
        state["status_message"] = tr("analysis_error", error=exc)
        state["simulation_result"] = None
        state["simulation_result_run_id"] = None
        state["simulation_markdown"] = f"### {tr('analysis_error_title')}\n\n`{exc}`"
        refresh_all()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def extract_uploaded_bytes(event: events.UploadEventArguments) -> bytes:
    content = getattr(event, "content", None)
    if content is not None:
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        if hasattr(content, "read"):
            maybe_bytes = content.read()
            if inspect.isawaitable(maybe_bytes):
                maybe_bytes = await maybe_bytes
            return maybe_bytes

    upload_file = getattr(event, "file", None)
    if upload_file is not None:
        if isinstance(upload_file, (bytes, bytearray)):
            return bytes(upload_file)
        if hasattr(upload_file, "read"):
            maybe_bytes = upload_file.read()
            if inspect.isawaitable(maybe_bytes):
                maybe_bytes = await maybe_bytes
            return maybe_bytes

    available = sorted(name for name in dir(event) if not name.startswith("_"))
    raise RuntimeError(tr("upload_format_error", available=available))


def extract_uploaded_name(event: events.UploadEventArguments) -> str:
    file_obj = getattr(event, "file", None)
    if file_obj is not None:
        file_name = getattr(file_obj, "name", None)
        if isinstance(file_name, str) and file_name.strip():
            return Path(file_name).name

    event_name = getattr(event, "name", None)
    if isinstance(event_name, str) and event_name.strip():
        return Path(event_name).name

    content = getattr(event, "content", None)
    if content is not None:
        content_name = getattr(content, "name", None)
        if isinstance(content_name, str) and content_name.strip():
            return Path(content_name).name

    return "upload.gpx"


def set_language(language: str) -> None:
    state["language"] = language
    if state.get("gpx_name"):
        state["status_message"] = tr("route_loaded", name=state["gpx_name"])
    refresh_all()


def set_param(key: str, value: Any) -> None:
    state["params"][key] = value


def set_float_param(key: str, value: Any) -> None:
    if value is None:
        return
    set_param(key, float(value))


def set_param_and_refresh(key: str, value: Any) -> None:
    set_param(key, value)
    refresh_all()


def split_sim_start_local(raw_value: str) -> tuple[str, str]:
    try:
        start_dt = datetime.fromisoformat(raw_value)
    except Exception:
        start_dt = datetime.now() + timedelta(hours=24)
    return start_dt.strftime("%Y-%m-%d"), start_dt.strftime("%H:%M")


def _distance_km_or_none(gpx_result: Any) -> float | None:
    if gpx_result is None:
        return None
    distance = _float_or_none(getattr(gpx_result, "distance_km", None))
    if distance is not None:
        return distance
    stats = getattr(gpx_result, "stats", {})
    if isinstance(stats, dict):
        return _float_or_none(stats.get("distance_km"))
    return None


def _parse_cut_bound(raw: Any) -> float | None:
    text = str(raw or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def set_cut_field(key: str, value: Any) -> None:
    state["cut_form"][key] = str(value or "")


def open_cut_editor() -> None:
    source = state.get("gpx_source_result") or state.get("gpx_result")
    if source is None:
        ui.notify(tr("gpx_cut_needs_gpx"), type="warning")
        return

    source_distance = _distance_km_or_none(source)
    source_name = str(state.get("gpx_source_name") or state.get("gpx_name") or tr("route_pending"))

    with ui.dialog() as dialog, ui.card().classes("w-[30rem]"):
        ui.label(tr("gpx_cut_title")).classes("text-subtitle1 text-weight-medium")
        ui.label(tr("gpx_cut_help")).classes("text-body2 text-grey-7")
        ui.label(tr("gpx_cut_source", name=source_name)).classes("text-caption text-grey-6")
        ui.label(
            tr("gpx_cut_source_distance", value=f"{source_distance:.2f}") if source_distance is not None else tr("not_available")
        ).classes("text-caption text-grey-6")

        with ui.row().classes("w-full gap-2"):
            ui.input(
                tr("gpx_cut_p1_label"),
                value=str(state.get("cut_form", {}).get("p1_km", "")),
                on_change=lambda e: set_cut_field("p1_km", e.value),
            ).classes("w-full")
            ui.input(
                tr("gpx_cut_p2_label"),
                value=str(state.get("cut_form", {}).get("p2_km", "")),
                on_change=lambda e: set_cut_field("p2_km", e.value),
            ).classes("w-full")

        def _apply_and_close() -> None:
            if apply_gpx_cut():
                dialog.close()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(tr("action_cancel"), on_click=dialog.close).props("flat")
            ui.button(tr("gpx_cut_apply"), on_click=_apply_and_close).props("unelevated color=primary")

    dialog.open()


def apply_gpx_cut() -> bool:
    source = state.get("gpx_source_result") or state.get("gpx_result")
    if source is None:
        ui.notify(tr("gpx_cut_needs_gpx"), type="warning")
        return False

    p1 = _parse_cut_bound(state.get("cut_form", {}).get("p1_km"))
    p2 = _parse_cut_bound(state.get("cut_form", {}).get("p2_km"))
    source_distance = _distance_km_or_none(source)

    if p1 is not None and p1 < 0:
        ui.notify(tr("gpx_cut_invalid_bounds"), type="warning")
        return False
    if p2 is not None and p2 < 0:
        ui.notify(tr("gpx_cut_invalid_bounds"), type="warning")
        return False
    if p1 is not None and p2 is not None and p2 <= p1:
        ui.notify(tr("gpx_cut_invalid_order"), type="warning")
        return False
    if source_distance is not None and p2 is not None and p2 > source_distance:
        ui.notify(tr("gpx_cut_overflow", max_km=f"{source_distance:.2f}"), type="warning")
        return False

    try:
        cut_result = source.cut(p1_km=p1, p2_km=p2)
    except Exception as exc:
        ui.notify(tr("gpx_cut_error", error=exc), type="negative")
        return False

    if cut_result is None or not getattr(cut_result, "segments", None):
        ui.notify(tr("gpx_cut_empty"), type="warning")
        return False

    source_name = str(state.get("gpx_source_name") or state.get("gpx_name") or "route.gpx")
    left = f"{p1:.2f}" if p1 is not None else "0.00"
    right = f"{p2:.2f}" if p2 is not None else (f"{source_distance:.2f}" if source_distance is not None else "end")

    state["gpx_result"] = cut_result
    state["gpx_name"] = f"{source_name} [cut {left}-{right} km]"
    state["status_message"] = tr("route_cut_loaded", p1=left, p2=right)
    state["simulation_result"] = None
    state["simulation_result_run_id"] = None
    state["simulation_markdown"] = ""
    _clear_plots()
    state["interactive_map_url"] = None
    state["interactive_map_run_id"] = None
    state["interactive_map_title"] = None
    ui.notify(tr("gpx_cut_done"), type="positive")
    refresh_all()
    return True


def reset_gpx_cut() -> None:
    source = state.get("gpx_source_result")
    if source is None:
        ui.notify(tr("gpx_cut_no_source"), type="warning")
        return

    state["gpx_result"] = source
    state["gpx_name"] = state.get("gpx_source_name")
    state["status_message"] = tr("route_cut_reset")
    state["simulation_result"] = None
    state["simulation_result_run_id"] = None
    state["simulation_markdown"] = ""
    _clear_plots()
    state["interactive_map_url"] = None
    state["interactive_map_run_id"] = None
    state["interactive_map_title"] = None
    ui.notify(tr("gpx_cut_reset"), type="positive")
    refresh_all()


def set_sim_start_date(value: Any) -> None:
    date_part = str(value or "").strip()
    if not date_part:
        return
    _, time_part = split_sim_start_local(str(state["params"].get("sim_start", "")))
    new_value = f"{date_part}T{time_part}"
    try:
        datetime.fromisoformat(new_value)
    except Exception:
        return
    set_param_and_refresh("sim_start", new_value)


def set_sim_start_time(value: Any) -> None:
    time_part = str(value or "").strip()
    if not time_part:
        return
    if len(time_part) >= 5:
        time_part = time_part[:5]
    date_part, _ = split_sim_start_local(str(state["params"].get("sim_start", "")))
    new_value = f"{date_part}T{time_part}"
    try:
        datetime.fromisoformat(new_value)
    except Exception:
        return
    set_param_and_refresh("sim_start", new_value)


def set_sim_start_hour(value: Any) -> None:
    hour_text = str(value or "").strip()
    if not hour_text:
        return
    date_part, time_part = split_sim_start_local(str(state["params"].get("sim_start", "")))
    minute_part = time_part[3:5] if len(time_part) >= 5 else "00"
    new_value = f"{date_part}T{hour_text[:2]}:{minute_part}"
    try:
        datetime.fromisoformat(new_value)
    except Exception:
        return
    set_param_and_refresh("sim_start", new_value)


def set_sim_start_minute(value: Any) -> None:
    minute_text = str(value or "").strip()
    if not minute_text:
        return
    date_part, time_part = split_sim_start_local(str(state["params"].get("sim_start", "")))
    hour_part = time_part[:2] if len(time_part) >= 2 else "00"
    new_value = f"{date_part}T{hour_part}:{minute_text[:2]}"
    try:
        datetime.fromisoformat(new_value)
    except Exception:
        return
    set_param_and_refresh("sim_start", new_value)


def sim_hour_options() -> dict[str, str]:
    return {f"{hour:02d}": f"{hour:02d}" for hour in range(24)}


def sim_minute_options() -> dict[str, str]:
    return {f"{minute:02d}": f"{minute:02d}" for minute in range(60)}


def set_profile_preset(value: str) -> None:
    apply_profile_preset(value)
    refresh_all()


def set_behavior_mode(key: str, value: str) -> None:
    set_param(key, value)
    refresh_all()


def weather_model_options() -> dict[str, str]:
    return {
        "NONE": tr("weather_model_none"),
        "GFS": "GFS",
        "IFS": "IFS",
    }


def weather_pas_options() -> dict[int, str]:
    return {
        1: "1h",
        3: "3h",
    }


@ui.refreshable
def header_panel() -> None:
    with ui.row().classes("w-full items-start justify-between gap-4"):
        with ui.column().classes("gap-1"):
            ui.label(tr("app_title")).classes("text-h4 text-weight-bold")
            ui.label(tr("app_subtitle")).classes("text-subtitle2 text-grey-7")
        with ui.row().classes("items-center gap-2"):
            if not state.get("needs_setup"):
                ui.button(icon="tune", on_click=open_setup_editor).props("flat round dense").tooltip(tr("setup_open_editor"))
            ui.select(
                {"fr": "FR", "en": "EN"},
                value=state["language"],
                label=tr("language"),
                on_change=lambda e: set_language(str(e.value)),
            ).classes("min-w-24")


@ui.refreshable
def left_panel() -> None:
    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            ui.label(tr("route_data")).classes("text-h6")
            ui.label(state.get("status_message") or tr("gpx_help")).classes("text-body2 text-grey-7")

            last_gpx_name = state.get("gpx_name")
            if last_gpx_name:
                with ui.row().classes("w-full items-center gap-2 bg-blue-1 text-primary rounded p-2"):
                    ui.icon("route")
                    ui.label(tr("last_gpx_loaded", name=last_gpx_name)).classes("text-body1 text-weight-medium")
            else:
                ui.label(tr("route_none")).classes("text-caption text-grey-6")

            if state.get("gpx_result") is not None:
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    current_distance = _distance_km_or_none(state.get("gpx_result"))
                    source_distance = _distance_km_or_none(state.get("gpx_source_result"))
                    if current_distance is not None:
                        if source_distance is not None and abs(current_distance - source_distance) > 1e-6:
                            ui.label(
                                tr("gpx_cut_distance_current_vs_source", current=f"{current_distance:.2f}", source=f"{source_distance:.2f}")
                            ).classes("text-caption text-grey-7")
                        else:
                            ui.label(tr("gpx_cut_distance_full", value=f"{current_distance:.2f}")).classes("text-caption text-grey-7")

                    with ui.row().classes("items-center gap-1"):
                        ui.button(icon="content_cut", on_click=open_cut_editor).props("flat round dense").tooltip(tr("gpx_cut_open"))
                        ui.button(icon="restart_alt", on_click=reset_gpx_cut).props("flat round dense").tooltip(tr("gpx_cut_reset"))

            ui.upload(on_upload=on_gpx_upload, auto_upload=True, label=tr("upload_gpx")).props("accept=.gpx")

        with ui.card().classes("w-full"):
            ui.label(tr("cyclist_params")).classes("text-h6")
            ui.label(tr("cyclist_params_dir", value=str(state["params"].get("cyclist_params_dir", "")))).classes("text-caption text-grey-6")

            file_options = cyclist_profile_file_options()
            with ui.row().classes("w-full items-end gap-2"):
                ui.select(
                    file_options,
                    value=state.get("selected_cyclist_profile_file", "") or None,
                    label=tr("cyclist_profile_file"),
                    on_change=lambda e: set_selected_cyclist_profile_file(e.value),
                ).classes("w-full")
                ui.button(tr("cyclist_profile_load"), on_click=load_cyclist_profile_file).props("outline")

            with ui.row().classes("w-full justify-end"):
                ui.button(tr("cyclist_profile_manage_open"), on_click=open_profile_manager).props("flat")

        with ui.card().classes("w-full"):
            ui.label(tr("weather_params")).classes("text-h6")
            with ui.column().classes("w-full gap-2"):
                ui.label(tr("weather_replay_info")).classes("text-caption text-grey-6")
                ui.select(
                    weather_model_options(),
                    value=state["params"]["weather_model"],
                    label=tr("weather_model"),
                    on_change=lambda e: set_param_and_refresh("weather_model", str(e.value)),
                ).classes("w-full")
                if _weather_enabled_for_actions():
                    ui.select(
                        weather_pas_options(),
                        value=state["params"]["weather_pas"],
                        label=tr("weather_pas"),
                        on_change=lambda e: set_param("weather_pas", int(e.value)),
                    ).classes("w-full")
                else:
                    ui.label(tr("weather_disabled")).classes("text-caption text-grey-7")
                ui.label(tr("weather_future_info")).classes("text-caption text-grey-6")

        with ui.card().classes("w-full"):
            ui.label(tr("simulation_params")).classes("text-h6")
            with ui.column().classes("w-full gap-2"):
                sim_start_date, sim_start_time = split_sim_start_local(str(state["params"]["sim_start"]))
                ui.select(
                    simulation_target_options(),
                    value=state["params"]["sim_target_mode"],
                    label=tr("simulation_target"),
                    on_change=lambda e: set_param_and_refresh("sim_target_mode", str(e.value)),
                ).classes("w-full")
                if state["params"]["sim_target_mode"] == "p0":
                    sim_p0_input = ui.number(
                        tr("sim_p0"),
                        value=state["params"]["sim_p0"],
                        min=80,
                        max=500,
                        step=5,
                        format="%.0f",
                        on_change=lambda e: set_param("sim_p0", float(e.value)) if e.value is not None else None,
                    ).classes("w-full")
                    sim_p0_input.on("blur", lambda _e: refresh_all())
                    sim_p0_input.on("keydown.enter", lambda _e: refresh_all())
                    ui.label(get_equivalent_target_text()).classes("text-caption text-grey-7")
                else:
                    sim_v0_input = ui.number(
                        tr("sim_v0_kmh"),
                        value=state["params"]["sim_v0_kmh"],
                        min=10,
                        max=60,
                        step=0.5,
                        format="%.1f",
                        on_change=lambda e: set_param("sim_v0_kmh", float(e.value)) if e.value is not None else None,
                    ).classes("w-full")
                    sim_v0_input.on("blur", lambda _e: refresh_all())
                    sim_v0_input.on("keydown.enter", lambda _e: refresh_all())
                    ui.label(get_equivalent_target_text()).classes("text-caption text-grey-7")
                ui.input(
                    tr("sim_start_date"),
                    value=sim_start_date,
                    on_change=lambda e: set_sim_start_date(e.value),
                ).classes("w-full").props('type="date"')
                with ui.row().classes("w-full items-end gap-2 flex-nowrap"):
                    ui.select(
                        sim_hour_options(),
                        value=(sim_start_time[:2] if len(sim_start_time) >= 2 else "00"),
                        label=tr("sim_start_hour_label"),
                        on_change=lambda e: set_sim_start_hour(e.value),
                    ).classes("w-28")
                    ui.select(
                        sim_minute_options(),
                        value=(sim_start_time[3:5] if len(sim_start_time) >= 5 else "00"),
                        label=tr("sim_start_minute_label"),
                        on_change=lambda e: set_sim_start_minute(e.value),
                    ).classes("w-28")
                ui.number(
                    tr("sim_duration_h"),
                    value=state["params"]["sim_duration_h"],
                    min=1,
                    max=24,
                    step=1,
                    format="%.0f",
                    on_change=lambda e: set_param_and_refresh("sim_duration_h", int(e.value) if e.value is not None else 4),
                ).classes("w-full")

        with ui.card().classes("w-full"):
            ui.label(tr("actions")).classes("text-h6")
            if state.get("is_busy"):
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label(state.get("busy_message") or tr("busy_running")).classes("text-caption text-grey-7")

            replay_enabled = (state.get("gpx_result") is not None and bool(state["gpx_result"].has_timestamps) and not state.get("is_busy"))
            simulation_enabled = state.get("gpx_result") is not None and not state.get("is_busy")
            replay_button = ui.button(tr("run_replay"), on_click=run_replay_action).props("unelevated color=primary")
            replay_button.classes("w-full")
            replay_button.set_enabled(replay_enabled)
            simulation_button = ui.button(tr("run_simulation"), on_click=run_simulation_action).props("unelevated color=primary")
            simulation_button.classes("w-full")
            simulation_button.set_enabled(simulation_enabled)


@ui.refreshable
def setup_panel() -> None:
    with ui.card().classes("w-full max-w-3xl mx-auto"):
        ui.label(tr("setup_title")).classes("text-h5 text-weight-bold")
        if GUI_SETTINGS_PATH.exists():
            ui.label(tr("setup_intro_edit")).classes("text-body2 text-grey-7")
        else:
            ui.label(tr("setup_intro")).classes("text-body2 text-grey-7")
        ui.label(str(GUI_SETTINGS_PATH)).classes("text-caption text-grey-6")

        if state.get("setup_error"):
            ui.label(tr("setup_load_error", error=state["setup_error"])).classes("text-caption text-red-7")

        with ui.column().classes("w-full gap-2"):
            ui.input(
                tr("general_grib_dir"),
                value=state["setup_form"]["general_grib_dir"],
                on_change=lambda e: set_setup_field("general_grib_dir", str(e.value or "")),
            )
            ui.input(
                tr("cyclist_params_dir_setup"),
                value=state["setup_form"]["cyclist_params_dir"],
                on_change=lambda e: set_setup_field("cyclist_params_dir", str(e.value or "")),
            )
            ui.number(
                "CdA",
                value=state["setup_form"]["CdA"],
                min=0.15,
                max=1.0,
                step=0.01,
                format="%.3f",
                on_change=lambda e: set_setup_field("CdA", float(e.value) if e.value is not None else DEFAULT_PERSISTED_PARAMS["CdA"]),
            )
            ui.number(
                "Cr",
                value=state["setup_form"]["Cr"],
                min=0.001,
                max=0.02,
                step=0.0005,
                format="%.4f",
                on_change=lambda e: set_setup_field("Cr", float(e.value) if e.value is not None else DEFAULT_PERSISTED_PARAMS["Cr"]),
            )
            ui.number(
                tr("mass_kg"),
                value=state["setup_form"]["mass_kg"],
                min=40.0,
                max=150.0,
                step=0.5,
                format="%.1f",
                on_change=lambda e: set_setup_field("mass_kg", float(e.value) if e.value is not None else DEFAULT_PERSISTED_PARAMS["mass_kg"]),
            )
            ui.select(
                weather_model_options(),
                value=state["setup_form"]["weather_model"],
                label=tr("weather_model"),
                on_change=lambda e: set_setup_field("weather_model", str(e.value)),
            )
            ui.select(
                weather_pas_options(),
                value=int(state["setup_form"]["weather_pas"]),
                label=tr("weather_pas"),
                on_change=lambda e: set_setup_field("weather_pas", int(e.value)),
            )
            ui.switch(
                tr("setup_weather_use_raster_roughness"),
                value=bool(state["setup_form"].get("weather_use_raster_roughness", False)),
                on_change=lambda e: set_setup_field("weather_use_raster_roughness", bool(e.value)),
            )
            ui.switch(
                tr("setup_map_enable_animation"),
                value=bool(state["setup_form"].get("map_enable_animation", False)),
                on_change=lambda e: set_setup_field("map_enable_animation", bool(e.value)),
            )
            ui.switch(
                tr("setup_map_distance_from_finish"),
                value=bool(state["setup_form"].get("map_distance_from_finish", False)),
                on_change=lambda e: set_setup_field("map_distance_from_finish", bool(e.value)),
            )
            ui.switch(
                tr("setup_process_gpx_verbose"),
                value=bool(state["setup_form"].get("process_gpx_verbose", True)),
                on_change=lambda e: set_setup_field("process_gpx_verbose", bool(e.value)),
            )

        with ui.row().classes("w-full justify-end gap-2"):
            if GUI_SETTINGS_PATH.exists():
                ui.button(tr("setup_back"), on_click=close_setup_editor).props("flat")
            ui.button(tr("setup_save"), on_click=create_initial_settings).props("unelevated color=primary")


@ui.refreshable
def profile_manager_panel() -> None:
    files = list_cyclist_profile_files()
    profile_form = _sanitize_profile_form(state.get("profile_form"))

    with ui.card().classes("w-full max-w-4xl mx-auto"):
        ui.label(tr("cyclist_profile_manager_title")).classes("text-h5 text-weight-bold")
        ui.label(tr("cyclist_profile_manager_intro")).classes("text-body2 text-grey-7")
        ui.label(tr("cyclist_params_dir", value=str(_cyclist_params_dir_path()))).classes("text-caption text-grey-6")

        with ui.row().classes("w-full justify-end"):
            ui.button(tr("cyclist_profile_manager_back"), on_click=close_profile_manager).props("flat")

        with ui.separator().classes("my-2"):
            pass

        with ui.column().classes("w-full gap-2"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(tr("cyclist_profile_manager_list_title")).classes("text-subtitle1 text-weight-medium")
                ui.button(tr("cyclist_profile_new"), on_click=new_profile_form).props("outline")

            if not files:
                ui.label(tr("cyclist_profile_manager_empty")).classes("text-body2 text-grey-7")
            else:
                for file_name in files:
                    with ui.row().classes("w-full items-center justify-between gap-2 p-2 border rounded"):
                        with ui.column().classes("gap-0"):
                            ui.label(file_name).classes("text-body2 text-weight-medium")
                            if str(state.get("selected_cyclist_profile_file") or "") == file_name:
                                ui.label(tr("run_selected")).classes("text-caption text-primary")
                        with ui.row().classes("items-center gap-1"):
                            ui.button(tr("cyclist_profile_edit"), on_click=lambda _e, n=file_name: load_selected_profile_into_form(n)).props("flat")
                            ui.button(tr("cyclist_profile_load_to_app"), on_click=lambda _e, n=file_name: load_cyclist_profile_file(n)).props("flat")
                            ui.button(tr("run_delete"), on_click=lambda _e, n=file_name: open_delete_cyclist_profile_confirmation(n)).props("flat color=negative")

        with ui.separator().classes("my-2"):
            pass

        ui.label(tr("cyclist_profile_create")).classes("text-subtitle1 text-weight-medium")
        ui.input(
            tr("cyclist_profile_name"),
            value=str(state.get("cyclist_profile_name", "")),
            on_change=lambda e: state.update({"cyclist_profile_name": str(e.value or "")}),
        ).classes("w-full")

        with ui.column().classes("w-full gap-2"):
            ui.number(
                "CdA",
                value=float(profile_form["CdA"]),
                min=0.15,
                max=1.0,
                step=0.01,
                format="%.3f",
                on_change=lambda e: set_profile_form_float_field("CdA", e.value),
            )
            ui.number(
                "Cr",
                value=float(profile_form["Cr"]),
                min=0.001,
                max=0.02,
                step=0.0005,
                format="%.4f",
                on_change=lambda e: set_profile_form_float_field("Cr", e.value),
            )
            ui.number(
                tr("mass_kg"),
                value=float(profile_form["mass_kg"]),
                min=40.0,
                max=150.0,
                step=0.5,
                format="%.1f",
                on_change=lambda e: set_profile_form_float_field("mass_kg", e.value),
            )

        ui.select(
            profile_preset_options(),
            value=str(profile_form["profile_preset"]),
            label=tr("profile_preset"),
            on_change=lambda e: set_profile_form_preset(str(e.value)),
        ).classes("w-full")

        with ui.expansion(tr("advanced_behavior"), icon="tune").classes("w-full"):
            mode_options = behavior_mode_options()
            ui.select(
                mode_options,
                value=str(profile_form["uphill_mode"]),
                label=tr("uphill_mode"),
                on_change=lambda e: set_profile_form_field("uphill_mode", str(e.value)),
            ).classes("w-full")
            ui.select(
                mode_options,
                value=str(profile_form["downhill_mode"]),
                label=tr("downhill_mode"),
                on_change=lambda e: set_profile_form_field("downhill_mode", str(e.value)),
            ).classes("w-full")
            ui.select(
                mode_options,
                value=str(profile_form["corner_mode"]),
                label=tr("corner_mode"),
                on_change=lambda e: set_profile_form_field("corner_mode", str(e.value)),
            ).classes("w-full")

        with ui.expansion(tr("behavior_thresholds_title"), icon="sliders").classes("w-full"):
            ui.label(tr("behavior_thresholds_help")).classes("text-caption text-grey-7")
            behavior_thresholds = profile_form.get("behavior_thresholds", {})
            
            def _get_threshold_value(key: str, default=None):
                """Safely get threshold value, returning None if empty."""
                val = behavior_thresholds.get(key)
                if val is None or val == "":
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            # Uphill thresholds
            ui.label("Montée / Uphill").classes("text-subtitle2 text-weight-bold mt-2")
            ui.number(
                tr("uphill_facteur_forte"),
                value=_get_threshold_value("uphill_facteur_forte"),
                on_change=lambda e: _set_behavior_threshold("uphill_facteur_forte", e.value),
            ).classes("w-full")
            ui.number(
                tr("uphill_facteur_moderee"),
                value=_get_threshold_value("uphill_facteur_moderee"),
                on_change=lambda e: _set_behavior_threshold("uphill_facteur_moderee", e.value),
            ).classes("w-full")
            ui.number(
                tr("uphill_facteur_legere"),
                value=_get_threshold_value("uphill_facteur_legere"),
                on_change=lambda e: _set_behavior_threshold("uphill_facteur_legere", e.value),
            ).classes("w-full")

            # Downhill thresholds
            ui.label("Descente / Downhill").classes("text-subtitle2 text-weight-bold mt-2")
            ui.number(
                tr("downhill_vitesse_max_absolue"),
                value=_get_threshold_value("downhill_vitesse_max_absolue"),
                on_change=lambda e: _set_behavior_threshold("downhill_vitesse_max_absolue", e.value),
            ).classes("w-full")
            ui.number(
                tr("downhill_vitesse_reduction_factor"),
                value=_get_threshold_value("downhill_vitesse_reduction_factor"),
                on_change=lambda e: _set_behavior_threshold("downhill_vitesse_reduction_factor", e.value),
            ).classes("w-full")
            ui.number(
                tr("downhill_vitesse_reduction_cap"),
                value=_get_threshold_value("downhill_vitesse_reduction_cap"),
                on_change=lambda e: _set_behavior_threshold("downhill_vitesse_reduction_cap", e.value),
            ).classes("w-full")
            ui.number(
                tr("downhill_puissance_min"),
                value=_get_threshold_value("downhill_puissance_min"),
                on_change=lambda e: _set_behavior_threshold("downhill_puissance_min", e.value),
            ).classes("w-full")
            ui.number(
                tr("downhill_facteur_forte"),
                value=_get_threshold_value("downhill_facteur_forte"),
                on_change=lambda e: _set_behavior_threshold("downhill_facteur_forte", e.value),
            ).classes("w-full")
            ui.number(
                tr("downhill_facteur_legere"),
                value=_get_threshold_value("downhill_facteur_legere"),
                on_change=lambda e: _set_behavior_threshold("downhill_facteur_legere", e.value),
            ).classes("w-full")
            ui.number(
                tr("downhill_corner_safety_factor"),
                value=_get_threshold_value("downhill_corner_safety_factor"),
                on_change=lambda e: _set_behavior_threshold("downhill_corner_safety_factor", e.value),
            ).classes("w-full")

            # Corner thresholds
            ui.label("Virages / Corners").classes("text-subtitle2 text-weight-bold mt-2")
            ui.number(
                tr("corner_speed_straight"),
                value=_get_threshold_value("corner_speed_straight"),
                on_change=lambda e: _set_behavior_threshold("corner_speed_straight", e.value),
            ).classes("w-full")
            ui.number(
                tr("corner_speed_slight"),
                value=_get_threshold_value("corner_speed_slight"),
                on_change=lambda e: _set_behavior_threshold("corner_speed_slight", e.value),
            ).classes("w-full")
            ui.number(
                tr("corner_speed_moderate"),
                value=_get_threshold_value("corner_speed_moderate"),
                on_change=lambda e: _set_behavior_threshold("corner_speed_moderate", e.value),
            ).classes("w-full")
            ui.number(
                tr("corner_speed_sharp"),
                value=_get_threshold_value("corner_speed_sharp"),
                on_change=lambda e: _set_behavior_threshold("corner_speed_sharp", e.value),
            ).classes("w-full")
            ui.number(
                tr("corner_speed_hairpin"),
                value=_get_threshold_value("corner_speed_hairpin"),
                on_change=lambda e: _set_behavior_threshold("corner_speed_hairpin", e.value),
            ).classes("w-full")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(tr("cyclist_profile_new"), on_click=new_profile_form).props("flat")
            ui.button(tr("cyclist_profile_save"), on_click=save_cyclist_profile_file).props("unelevated color=primary")


@ui.refreshable
def main_content_panel() -> None:
    if state.get("needs_setup"):
        setup_panel()
        return

    if state.get("profile_manager_open"):
        profile_manager_panel()
        return

    with ui.row().classes("w-full items-start gap-6"):
        with ui.column().classes("w-full lg:w-80 gap-4"):
            left_panel()
        with ui.column().classes("w-full flex-1 gap-4"):
            results_panel()


def render_kpi(title: str, value: str) -> None:
    with ui.card().classes("min-w-40 bg-slate-50"):
        ui.label(title).classes("text-caption text-grey-7")
        ui.label(value).classes("text-h6")


def compare_runs_markdown() -> str:
    compare_ids: list[str] = list(state.get("compare_run_ids", []))
    if len(compare_ids) != 2:
        return tr("compare_need_two")

    left = get_run_by_id(compare_ids[0])
    right = get_run_by_id(compare_ids[1])
    if left is None or right is None:
        return tr("compare_need_two")

    lm = left.get("metrics", {})
    rm = right.get("metrics", {})

    speed_a = _float_or_none(lm.get("speed_kmh"))
    speed_b = _float_or_none(rm.get("speed_kmh"))
    power_a = _float_or_none(lm.get("power_w"))
    power_b = _float_or_none(rm.get("power_w"))
    dist_a = _float_or_none(lm.get("distance_km"))
    dist_b = _float_or_none(rm.get("distance_km"))
    dur_a = _float_or_none(lm.get("duration_s"))
    dur_b = _float_or_none(rm.get("duration_s"))

    lines = [
        f"### {tr('compare_title')}",
        "",
        f"- A: **{get_run_title_with_weather(left)}** ({get_run_reference_time_text(left)})",
        f"- B: **{get_run_title_with_weather(right)}** ({get_run_reference_time_text(right)})",
        "",
        f"| {tr('compare_metric')} | A | B | {tr('compare_delta_b_minus_a')} |",
        "|---|---:|---:|---:|",
        f"| {tr('kpi_distance')} | {_format_metric(dist_a, 'km', 2)} | {_format_metric(dist_b, 'km', 2)} | {_format_delta(dist_a, dist_b, 'km', 2)} |",
        f"| {tr('kpi_speed')} | {_format_metric(speed_a, 'km/h', 2)} | {_format_metric(speed_b, 'km/h', 2)} | {_format_delta(speed_a, speed_b, 'km/h', 2)} |",
        f"| {tr('kpi_power')} | {_format_metric(power_a, 'W', 1)} | {_format_metric(power_b, 'W', 1)} | {_format_delta(power_a, power_b, 'W', 1)} |",
        f"| {tr('kpi_duration')} | {format_duration_seconds(dur_a)} | {format_duration_seconds(dur_b)} | {_format_delta(dur_a, dur_b, 's', 0)} |",
        f"| {tr('kpi_windscore')} | {lm.get('windscore_grade') or tr('not_available')} | {rm.get('windscore_grade') or tr('not_available')} | {_windscore_delta(lm.get('windscore_grade'), rm.get('windscore_grade'))} |",
    ]
    return "\n".join(lines)


@ui.refreshable
def results_panel() -> None:
    active_run = get_active_run()
    gpx_result = state.get("gpx_result")
    route_status = state.get("gpx_name") or tr("route_pending")

    if active_run is None:
        simulation_status = tr("simulation_pending")
        run_label = tr("simulation_pending")
    else:
        simulation_status = tr(run_kind_label_key(str(active_run.get("kind", ""))) )
        run_label = format_history_time(str(active_run.get("created_at", "")))

    with ui.column().classes("w-full gap-4"):
        with ui.row().classes("w-full gap-4"):
            render_kpi(tr("status_label"), route_status)
            render_kpi(tr("tab_simulation"), simulation_status)
            render_kpi(tr("active_run"), run_label)

        with ui.tabs(
            value=state.get("results_active_tab", "summary"),
            on_change=lambda e: set_results_active_tab(str(e.value)),
        ).classes("w-full") as tabs:
            ui.tab("summary", label=tr("tab_summary"))
            ui.tab("details", label=tr("tab_details"))
            ui.tab("history", label=tr("tab_history"))
        with ui.tab_panels(tabs, value=state.get("results_active_tab", "summary")).classes("w-full"):
            with ui.tab_panel("summary"):
                with ui.card().classes("w-full"):
                    if active_run is None:
                        ui.markdown(tr("summary_ready") if gpx_result else tr("summary_waiting"))
                        ui.label(tr("result_placeholder")).classes("text-body2 text-grey-7")
                    else:
                        source_gpx = str(active_run.get("source_gpx", "")).strip() or tr("route_pending")
                        ui.label(f"{tr('gpx_file')}: {source_gpx}").classes("text-caption text-grey-7")
                        metrics = active_run.get("metrics", {})
                        with ui.row().classes("w-full gap-4"):
                            speed = _float_or_none(metrics.get("speed_kmh"))
                            power = _float_or_none(metrics.get("power_w"))
                            p0 = _float_or_none(metrics.get("p0_w"))
                            dist = _float_or_none(metrics.get("distance_km"))
                            duration_s = _float_or_none(metrics.get("duration_s"))
                            windscore = str(metrics.get("windscore_grade") or tr("not_available"))
                            render_kpi(tr("kpi_speed"), f"{speed:.2f} km/h" if speed is not None else tr("not_available"))
                            render_kpi(tr("kpi_power"), f"{power:.1f} W" if power is not None else tr("not_available"))
                            render_kpi(tr("kpi_p0"), f"{p0:.1f} W" if p0 is not None else tr("not_available"))
                            render_kpi(tr("kpi_distance"), f"{dist:.2f} km" if dist is not None else tr("not_available"))
                            render_kpi(tr("kpi_duration"), format_duration_seconds(duration_s))
                            render_kpi(tr("kpi_windscore"), windscore)

                        summary_md = active_run.get("summary_markdown") or tr("result_placeholder")
                        ui.markdown(summary_md)

            with ui.tab_panel("details"):
                with ui.column().classes("w-full gap-3"):
                    if active_run is None:
                        ui.label(tr("details_none")).classes("text-body2 text-grey-7")
                    else:
                        with ui.card().classes("w-full"):
                            ui.label(tr("details_simulation")).classes("text-subtitle1 text-weight-medium")
                            params = active_run.get("params", {})
                            target = active_run.get("target", {})
                            target_mode = str(target.get("mode") or params.get("sim_target_mode", ""))
                            target_line = target.get("label")
                            if not target_line:
                                if target_mode == "v0":
                                    target_line = tr("sim_target_used_v0", value=float(params.get("sim_v0_kmh", 0.0)))
                                elif target_mode == "p0":
                                    target_line = tr("sim_target_used_p0", value=float(params.get("sim_p0", 0.0)))
                                else:
                                    target_line = tr("not_available")
                            ui.markdown(
                                "\n".join(
                                    [
                                        f"- {tr('run_kind')}: **{tr(run_kind_label_key(str(active_run.get('kind', ''))))}**",
                                        f"- {tr('run_datetime')}: **{get_run_reference_time_text(active_run)}**",
                                        f"- {tr('run_created_at')}: **{format_history_time(str(active_run.get('created_at', '')))}**",
                                        f"- {tr('simulation_target')}: **{target_line}**",
                                        f"- {tr('kpi_windscore')}: **{active_run.get('metrics', {}).get('windscore_grade', tr('not_available'))}**",
                                        f"- {tr('windscore_reason')}: **{active_run.get('metrics', {}).get('windscore_reason', tr('not_available'))}**",
                                    ]
                                )
                            )

                        with ui.card().classes("w-full"):
                            ui.label(tr("details_weather")).classes("text-subtitle1 text-weight-medium")
                            weather = active_run.get("weather", {})
                            enabled = bool(weather.get("enabled"))
                            if not enabled:
                                ui.label(tr("weather_disabled")).classes("text-body2 text-grey-7")
                            else:
                                ui.markdown(
                                    "\n".join(
                                        [
                                            f"- {tr('weather_model')}: **{weather.get('model', tr('not_available'))}**",
                                            f"- {tr('weather_pas')}: **{weather.get('pas', tr('not_available'))}**",
                                            f"- {tr('grib_files')}: **{weather.get('grib_count', tr('not_available'))}**",
                                            f"- {tr('start')}: **{weather.get('start_utc', tr('not_available'))}**",
                                            f"- {tr('sim_duration_h')}: **{weather.get('duration_h', tr('not_available'))}**",
                                        ]
                                    )
                                )

                        with ui.card().classes("w-full"):
                            ui.label(tr("details_segments")).classes("text-subtitle1 text-weight-medium")
                            route = active_run.get("route", {})
                            source_gpx = str(active_run.get("source_gpx", "")).strip() or tr("route_pending")
                            ui.markdown(
                                "\n".join(
                                    [
                                        f"- {tr('gpx_file')}: **{source_gpx}**",
                                        f"- {tr('segments')}: **{route.get('segments', tr('not_available'))}**",
                                        f"- {tr('kpi_distance')}: **{route.get('distance_km', tr('not_available'))} km**" if route.get("distance_km") is not None else f"- {tr('kpi_distance')}: {tr('not_available')}",
                                        f"- {tr('timestamps')}: **{tr('yes') if route.get('has_timestamps') else tr('no')}**",
                                        f"- {tr('start')}: **{route.get('start', tr('not_available'))}**",
                                        f"- {tr('end')}: **{route.get('end', tr('not_available'))}**",
                                        f"- {tr('kpi_duration')}: **{format_duration_seconds(_float_or_none(route.get('duration_s')))}**",
                                    ]
                                )
                            )

                        with ui.card().classes("w-full"):
                            ui.label(tr("details_params")).classes("text-subtitle1 text-weight-medium")
                            p = active_run.get("params", {})
                            ui.markdown(
                                "\n".join(
                                    [
                                        f"- CdA: **{p.get('CdA', tr('not_available'))}**",
                                        f"- Cr: **{p.get('Cr', tr('not_available'))}**",
                                        f"- {tr('mass_kg')}: **{p.get('mass_kg', tr('not_available'))}**",
                                        f"- {tr('profile_preset')}: **{p.get('profile_preset', tr('not_available'))}**",
                                        f"- {tr('uphill_mode')}: **{p.get('uphill_mode', tr('not_available'))}**",
                                        f"- {tr('downhill_mode')}: **{p.get('downhill_mode', tr('not_available'))}**",
                                        f"- {tr('corner_mode')}: **{p.get('corner_mode', tr('not_available'))}**",
                                    ]
                                )
                            )

                        stats_text = get_active_run_statistics_text(active_run)
                        with ui.card().classes("w-full"):
                            ui.label(tr("details_statistics")).classes("text-subtitle1 text-weight-medium")
                            if stats_text.strip():
                                ui.code(stats_text, language="text").classes("w-full text-xs")
                            else:
                                ui.label(tr("details_statistics_none")).classes("text-body2 text-grey-7")

                        with ui.card().classes("w-full"):
                            ui.label(tr("details_plots")).classes("text-subtitle1 text-weight-medium")
                            has_result = state.get("simulation_result") is not None
                            if not has_result:
                                ui.label(tr("details_plots_need_result")).classes("text-body2 text-grey-7")
                            else:
                                plots_src = state.get("plots_gpx_name")
                                if plots_src:
                                    ui.label(tr("plots_source", name=plots_src)).classes("text-caption text-grey-6")

                                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                    ui.number(
                                        tr("smoothing_window"),
                                        value=state["params"].get("smoothing_window", 17),
                                        min=3, max=51, step=2,
                                        format="%.0f",
                                        on_change=lambda e: set_param("smoothing_window", int(e.value) if e.value else 17),
                                    ).classes("w-36")
                                    ui.button(tr("plot_wind_rose"), icon="air", on_click=generate_plot_wind_rose).props("outline")
                                    ui.button(tr("plot_elevation"), icon="terrain", on_click=generate_plot_elevation).props("outline")
                                    ui.button(tr("plot_speed_wind"), icon="speed", on_click=generate_plot_speed_wind).props("outline")
                                    any_plot = any([
                                        state.get("plot_wind_rose_img"),
                                        state.get("plot_elevation_img"),
                                        state.get("plot_speed_wind_img"),
                                    ])
                                    if any_plot:
                                        ui.button(tr("plots_clear"), icon="delete_outline", on_click=lambda: (_clear_plots(), refresh_all())).props("flat color=grey")

                                for img_key, label_key in [
                                    ("plot_wind_rose_img", "plot_wind_rose"),
                                    ("plot_elevation_img", "plot_elevation"),
                                    ("plot_speed_wind_img", "plot_speed_wind"),
                                ]:
                                    img_data = state.get(img_key)
                                    if img_data:
                                        with ui.expansion(tr(label_key), icon="image").classes("w-full"):
                                            ui.image(img_data).classes("w-full")

                        with ui.card().classes("w-full"):
                            ui.label(tr("details_interactive_map")).classes("text-subtitle1 text-weight-medium")

                            active_run_id = str(active_run.get("id")) if active_run is not None else ""
                            map_url = state.get("interactive_map_url")
                            map_run_id = str(state.get("interactive_map_run_id") or "")

                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                ui.button(
                                    tr("interactive_map_generate"),
                                    icon="map",
                                    on_click=generate_interactive_map_for_active_run,
                                ).props("outline")

                                if map_url and map_run_id == active_run_id:
                                    ui.button(
                                        tr("interactive_map_open_new_tab"),
                                        icon="open_in_new",
                                        on_click=lambda url=map_url: _open_url_in_new_tab(str(url)),
                                    ).props("flat")

                            if map_url and map_run_id == active_run_id:
                                ui.label(tr("interactive_map_preview")).classes("text-caption text-grey-6")

            with ui.tab_panel("history"):
                with ui.column().classes("w-full gap-3"):
                    with ui.card().classes("w-full"):
                        with ui.row().classes("w-full items-center justify-between gap-2"):
                            ui.label(tr("run_history_title")).classes("text-subtitle1 text-weight-medium")
                            ui.button(tr("run_clear_history"), on_click=open_clear_history_confirmation).props("flat color=negative")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.select(
                                history_kind_filter_options(),
                                value=state.get("history_kind_filter", "all"),
                                label=tr("history_filter_label"),
                                on_change=lambda e: set_history_kind_filter(str(e.value)),
                            ).classes("w-full")
                            ui.select(
                                history_sort_options(),
                                value=state.get("history_sort_order", "newest"),
                                label=tr("history_sort_label"),
                                on_change=lambda e: set_history_sort_order(str(e.value)),
                            ).classes("w-full")

                        history: list[dict[str, Any]] = get_filtered_history()
                        active_run = get_active_run()
                        active_run_id = str(active_run.get("id")) if active_run is not None else ""
                        if not history:
                            ui.label(tr("run_no_history")).classes("text-body2 text-grey-7")
                        else:
                            for record in history:
                                run_id = str(record.get("id", ""))
                                checked = run_id in state.get("compare_run_ids", [])
                                with ui.row().classes("w-full items-center justify-between gap-2 p-2 border rounded"):
                                    with ui.column().classes("gap-0 w-full"):
                                        with ui.row().classes("items-center gap-2"):
                                            run_type_label = tr(run_kind_label_key(str(record.get("kind", ""))) )
                                            run_date_label = get_run_reference_time_text(record)
                                            ui.chip(run_type_label).props(f"dense square color={run_kind_chip_color(str(record.get('kind', '')))} text-color=white")
                                            ui.chip(get_run_weather_model_label(record)).props(
                                                f"dense square color={run_weather_chip_color(record)} text-color=white"
                                            )
                                            ui.label(tr("run_history_line", run_date=run_date_label)).classes("text-body2 text-weight-medium")
                                            if run_id == active_run_id:
                                                ui.chip(tr("run_selected")).props("dense outline color=primary")
                                            if run_id in state.get("segments_cache", {}):
                                                ui.chip(tr("run_segments_cached")).props("dense square color=green-6 text-color=white")
                                        ui.label(str(record.get("source_gpx", "")).strip() or tr("route_pending")).classes("text-body2")
                                        metrics = record.get("metrics", {})
                                        speed = _float_or_none(metrics.get("speed_kmh"))
                                        power = _float_or_none(metrics.get("power_w"))
                                        windscore = metrics.get("windscore_grade") or tr("not_available")
                                        ui.label(
                                            tr(
                                                "run_history_compact",
                                                run_date=get_run_reference_time_text(record),
                                                saved_date=format_history_time(str(record.get("created_at", ""))),
                                                speed=f"{speed:.1f}" if speed is not None else tr("not_available"),
                                                power=f"{power:.0f}" if power is not None else tr("not_available"),
                                                windscore=windscore,
                                            )
                                        ).classes("text-caption text-grey-6")
                                    with ui.row().classes("items-center gap-2"):
                                        compare_box = ui.checkbox(tr("run_compare"), value=checked, on_change=lambda e, rid=run_id: toggle_compare_run(rid, bool(e.value)))
                                        compare_box.set_enabled(can_select_for_compare(run_id))
                                        ui.button(tr("run_reopen"), on_click=lambda _e, rid=run_id: select_run(rid)).props("flat")
                                        ui.button(tr("run_delete"), on_click=lambda _e, rid=run_id: open_delete_run_confirmation(rid)).props("flat color=negative")

                    with ui.card().classes("w-full"):
                        ui.label(tr("compare_title")).classes("text-subtitle1 text-weight-medium")
                        ui.label(tr("run_compare_help")).classes("text-caption text-grey-6")
                        ui.markdown(compare_runs_markdown())

                        compare_ids_now = list(state.get("compare_run_ids", []))
                        if len(compare_ids_now) == 2:
                            can_plot, reason_key = get_compare_runs_compatibility()
                            with ui.row().classes("w-full items-center gap-2 mt-2"):
                                btn = ui.button(tr("plot_compare"), icon="show_chart", on_click=generate_plot_compare).props("outline")
                                btn.set_enabled(can_plot)
                                if not can_plot:
                                    ui.label(tr(reason_key)).classes("text-caption text-grey-6")
                                elif state.get("plot_compare_img"):
                                    ui.button(tr("plots_clear"), icon="delete_outline",
                                              on_click=lambda: (state.update({"plot_compare_img": None}), refresh_all())).props("flat color=grey")
                            if state.get("plot_compare_img"):
                                with ui.expansion(tr("plot_compare"), icon="image").classes("w-full"):
                                    ui.image(state["plot_compare_img"]).classes("w-full")


def refresh_all() -> None:
    header_panel.refresh()
    main_content_panel.refresh()


ui.page_title(tr("app_title"))
ui.colors(primary="#1d4ed8", secondary="#0f766e", accent="#ea580c")

initialize_gui_settings()
load_run_history()

with ui.column().classes("w-full max-w-screen-2xl mx-auto p-6 gap-6"):
    header_panel()
    main_content_panel()




ui.run(title=tr("app_title"), port=8081)
