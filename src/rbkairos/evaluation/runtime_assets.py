from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evaluation.parameters import (
    FIXED_EFFORT_LIMIT_NM,
    FIXED_KD,
    FIXED_MU_HIGH,
    FIXED_MU_LOW,
    EvaluationParameters,
)


WHEEL_JOINTS = {
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
}
WHEEL_CONTACT_PAIRS = {"fric_lf", "fric_rf", "fric_lb", "fric_rb"}
MJCF_ASSET_FILE_TAGS = {"mesh", "texture", "hfield"}


@dataclass
class RuntimeAssets:
    asset_dir: Path
    urdf_path: Path | None
    mujoco_model_path: Path | None
    traction_mujoco_model_paths: dict[str, Path]
    config_path: Path | None
    fingerprint: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_dir": str(self.asset_dir),
            "urdf_path": str(self.urdf_path) if self.urdf_path else None,
            "mujoco_model_path": str(self.mujoco_model_path) if self.mujoco_model_path else None,
            "traction_mujoco_model_paths": {
                axis: str(path)
                for axis, path in sorted(self.traction_mujoco_model_paths.items())
            },
            "config_path": str(self.config_path) if self.config_path else None,
            "fingerprint": self.fingerprint,
            "notes": list(self.notes),
        }


def _write_parameterized_urdf(
    source: Path,
    target: Path,
    parameters: EvaluationParameters,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    for joint in root.findall(".//joint"):
        if joint.get("name") not in WHEEL_JOINTS:
            continue
        limit = joint.find("limit")
        if limit is not None:
            limit.set("effort", str(FIXED_EFFORT_LIMIT_NM))
    for joint in root.findall(".//ros2_control/joint"):
        if joint.get("name") not in WHEEL_JOINTS:
            continue
        for param in joint.findall("param"):
            if param.get("name") == "velocity_kp":
                param.text = str(float(parameters.kp))
            elif param.get("name") == "velocity_ki":
                param.text = str(float(parameters.ki))
            elif param.get("name") == "velocity_i_max":
                param.text = str(float(parameters.i_max))
            elif param.get("name") == "velocity_i_min":
                param.text = str(-float(parameters.i_max))
            elif param.get("name") == "velocity_kd":
                param.text = str(FIXED_KD)
    tree.write(target, encoding="unicode", xml_declaration=True)


def _rewrite_relative_mjcf_asset_paths(root: ET.Element, source_dir: Path) -> None:
    compiler = root.find("./compiler")
    mesh_base = source_dir
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir:
            meshdir_path = Path(meshdir)
            mesh_base = meshdir_path if meshdir_path.is_absolute() else source_dir / meshdir_path

    for element in root.iter():
        raw_file = element.get("file")
        if (
            element.tag not in MJCF_ASSET_FILE_TAGS
            or raw_file is None
            or not raw_file.strip()
            or "://" in raw_file
        ):
            continue
        path = Path(raw_file)
        if path.is_absolute():
            continue
        base_dir = mesh_base if element.tag == "mesh" else source_dir
        element.set("file", str((base_dir / path).resolve()))


def _write_parameterized_mjcf(
    source: Path,
    target: Path,
    parameters: EvaluationParameters,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    _rewrite_relative_mjcf_asset_paths(root, source.parent)
    for pair in root.findall(".//contact/pair"):
        if pair.get("name") not in WHEEL_CONTACT_PAIRS:
            continue
        raw = pair.get("friction", "")
        values = raw.split()
        if len(values) >= 2:
            values[0] = str(FIXED_MU_LOW)
            values[1] = str(FIXED_MU_HIGH)
        elif len(values) == 1:
            values[0] = str(FIXED_MU_LOW)
            values.append(str(FIXED_MU_HIGH))
        else:
            values = [str(FIXED_MU_LOW), str(FIXED_MU_HIGH)]
        pair.set("friction", " ".join(values))
    tree.write(target, encoding="unicode", xml_declaration=True)


def _write_parameterized_traction_scene(
    source: Path,
    target: Path,
    base_model_path: Path,
    source_model_name: str,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    for include in root.findall(".//include"):
        raw_file = include.get("file")
        if raw_file is None or not raw_file.strip():
            continue
        include_path = Path(raw_file)
        if include_path.name == source_model_name:
            include.set("file", base_model_path.name)
        elif not include_path.is_absolute():
            include.set("file", str((source.parent / include_path).resolve()))
    tree.write(target, encoding="unicode", xml_declaration=True)


def _write_parameterized_config(source: Path, target: Path, model_path: Path | None) -> None:
    with source.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    rewritten = copy.deepcopy(data)
    if model_path is not None:
        rewritten.setdefault("mujoco_sim_ros2_node", {}).setdefault(
            "ros__parameters", {}
        )["model_file"] = str(model_path)
    with target.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(rewritten, file_obj, sort_keys=False)


def _fingerprint(paths: list[Path], parameters: EvaluationParameters) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(parameters.to_dict(), sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def create_runtime_assets(
    parameters: EvaluationParameters,
    asset_dir: Path,
    source_urdf: Path = Path("src/rbkairos/urdf/rbkairos.urdf"),
    source_mjcf: Path = Path("src/rbkairos/mujoco/rbkairos.xml"),
    source_traction_scenes: dict[str, Path] | None = None,
    source_config: Path = Path("src/rbkairos/config/mujoco_mecanum_drive.yaml"),
) -> RuntimeAssets:
    parameters.validate()
    asset_dir.mkdir(parents=True, exist_ok=True)
    notes = [
        "Generated candidate-local assets; source URDF/MJCF/config files were not modified.",
        "Candidate variables are Kp, Ki, and i_max; i_min=-i_max.",
        "Fixed runtime values: mu=(0.01, 0.7), Kd=0, effort_limit=100 Nm.",
        "Managed launch consumes these assets through urdf_file, config_file, "
        "and model_file arguments.",
    ]

    urdf_path = asset_dir / "rbkairos.parameterized.urdf"
    mjcf_path = asset_dir / "rbkairos.parameterized.xml"
    if source_traction_scenes is None:
        source_traction_scenes = {
            axis: Path(f"src/rbkairos/mujoco/rbkairos_scene_tendon_test_{axis}.xml")
            for axis in ("x", "y")
        }
    traction_mjcf_paths: dict[str, Path] = {}
    config_path = asset_dir / "mujoco_mecanum_drive.parameterized.yaml"

    _write_parameterized_urdf(source_urdf, urdf_path, parameters)
    _write_parameterized_mjcf(source_mjcf, mjcf_path, parameters)
    for axis in ("x", "y"):
        source_scene = source_traction_scenes.get(axis)
        if source_scene is None or not source_scene.exists():
            notes.append(f"No candidate-local Traction {axis.upper()} scene was generated.")
            continue
        target_scene = asset_dir / f"rbkairos_scene_tendon_test_{axis}.parameterized.xml"
        _write_parameterized_traction_scene(
            source_scene,
            target_scene,
            mjcf_path,
            source_mjcf.name,
        )
        traction_mjcf_paths[axis] = target_scene
    _write_parameterized_config(source_config, config_path, mjcf_path)

    fingerprint_paths = [urdf_path, mjcf_path, config_path]
    fingerprint_paths.extend(traction_mjcf_paths[axis] for axis in sorted(traction_mjcf_paths))
    fingerprint = _fingerprint(fingerprint_paths, parameters)
    return RuntimeAssets(
        asset_dir=asset_dir,
        urdf_path=urdf_path,
        mujoco_model_path=mjcf_path,
        traction_mujoco_model_paths=traction_mjcf_paths,
        config_path=config_path,
        fingerprint=fingerprint,
        notes=notes,
    )
