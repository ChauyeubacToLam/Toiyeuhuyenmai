from __future__ import annotations

import json
import shutil
import zipfile
import copy
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


APP_VERSION = "0.3.0"


def new_project_state(name: str = "Dự án PLS chưa đặt tên") -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "app_version": APP_VERSION,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "workspace": "",
        "data_path": "",
        "model": {"nodes": [], "connections": []},
        "data_files": [],
        "models": [],
        "active_model_id": "",
        "active_data_id": "",
        "results_history": [],
        "notes": "",
    }


def save_project(path: str, state: dict[str, Any]) -> None:
    state = normalize_project_state(copy.deepcopy(state))
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_project(path: str) -> dict[str, Any]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    base = new_project_state(state.get("name", "Dự án PLS chưa đặt tên"))
    base.update(state)
    return normalize_project_state(base)


def normalize_project_state(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate 0.2 single-model projects to the multi-model 0.3 schema."""
    state.setdefault("data_files", [])
    state.setdefault("models", [])
    state.setdefault("active_model_id", "")
    state.setdefault("active_data_id", "")

    if not state["data_files"] and state.get("data_path"):
        data_path = str(state["data_path"])
        data_id = str(uuid.uuid4())
        state["data_files"].append(
            {
                "id": data_id,
                "name": Path(data_path).stem or "Data",
                "path": data_path,
            }
        )
        state["active_data_id"] = data_id

    legacy_model = state.get("model") or {"nodes": [], "connections": []}
    if not state["models"] and (legacy_model.get("nodes") or state.get("model_name")):
        model_id = str(uuid.uuid4())
        state["models"].append(
            {
                "id": model_id,
                "name": state.get("model_name", "Path Model"),
                "data_file_id": state.get("active_data_id", ""),
                "model": legacy_model,
            }
        )
        state["active_model_id"] = model_id

    for entry in state["data_files"]:
        entry.setdefault("id", str(uuid.uuid4()))
        entry.setdefault("name", Path(str(entry.get("path", "Data"))).stem or "Data")
        entry.setdefault("path", "")
    data_ids = {entry.get("id") for entry in state["data_files"]}
    for model in state["models"]:
        model.setdefault("id", str(uuid.uuid4()))
        model.setdefault("name", "Path Model")
        model.setdefault("data_file_id", "")
        model.setdefault("model", {"nodes": [], "connections": []})
        if model["data_file_id"] not in data_ids:
            model["data_file_id"] = ""

    model_ids = {entry.get("id") for entry in state["models"]}
    if state.get("active_model_id") not in model_ids:
        state["active_model_id"] = state["models"][0]["id"] if state["models"] else ""
    data_ids = {entry.get("id") for entry in state["data_files"]}
    if state.get("active_data_id") not in data_ids:
        state["active_data_id"] = state["data_files"][0]["id"] if state["data_files"] else ""

    active_model = next((entry for entry in state["models"] if entry["id"] == state["active_model_id"]), None)
    if active_model:
        state["model"] = active_model["model"]
        state["model_name"] = active_model["name"]
        if active_model.get("data_file_id"):
            state["active_data_id"] = active_model["data_file_id"]
    active_data = next((entry for entry in state["data_files"] if entry["id"] == state["active_data_id"]), None)
    state["data_path"] = active_data.get("path", "") if active_data else ""
    return state


def export_project_zip(path: str, state: dict[str, Any], data_path: str | None = None) -> None:
    target = Path(path)
    payload = normalize_project_state(copy.deepcopy(state))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        for entry in payload.get("data_files", []):
            data_file = Path(str(entry.get("path", "")))
            if not data_file.exists():
                continue
            name = data_file.name
            counter = 2
            while name.lower() in used_names:
                name = f"{data_file.stem}_{counter}{data_file.suffix}"
                counter += 1
            used_names.add(name.lower())
            entry["path"] = f"data/{name}"
            archive.write(data_file, f"data/{name}")
        active = next((item for item in payload.get("data_files", []) if item.get("id") == payload.get("active_data_id")), None)
        payload["data_path"] = active.get("path", "") if active else ""
        archive.writestr("project.json", json.dumps(payload, indent=2, ensure_ascii=False))


def import_project_zip(zip_path: str, target_dir: str) -> str:
    destination = Path(target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    return str(destination / "project.json")


def duplicate_project_file(source_path: str, destination_path: str) -> None:
    shutil.copy2(source_path, destination_path)
