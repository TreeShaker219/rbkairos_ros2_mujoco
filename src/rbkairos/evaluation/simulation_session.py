from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SimulationSession:
    """Process wrapper for future managed MuJoCo execution."""

    command: list[str]
    log_dir: Path
    process: subprocess.Popen[Any] | None = field(default=None, init=False)
    stdout_path: Path | None = field(default=None, init=False)
    stderr_path: Path | None = field(default=None, init=False)
    start_error: str | None = field(default=None, init=False)
    process_group_id: int | None = field(default=None, init=False)
    stop_error: str | None = field(default=None, init=False)
    sent_sigterm_to_group: bool = field(default=False, init=False)
    sent_sigkill_to_group: bool = field(default=False, init=False)
    _stdout_handle: Any | None = field(default=None, init=False, repr=False)
    _stderr_handle: Any | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.log_dir / "mujoco_stdout.log"
        self.stderr_path = self.log_dir / "mujoco_stderr.log"
        self._stdout_handle = self.stdout_path.open("w", encoding="utf-8")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                start_new_session=True,
            )
            try:
                self.process_group_id = os.getpgid(self.process.pid)
            except OSError as exc:
                self.stop_error = f"process group lookup failed: {exc}"
        except OSError as exc:
            self.start_error = str(exc)
            self._close_logs()
            raise

    def stop(self, timeout_s: float = 10.0) -> None:
        try:
            if self.process is None:
                return
            if self.process_group_id is not None:
                self.sent_sigterm_to_group = self._signal_process_group(signal.SIGTERM)
            elif self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.force_kill(timeout_s=timeout_s)
                self.process.wait(timeout=timeout_s)
        finally:
            self._close_logs()

    def force_kill(self, timeout_s: float = 1.0) -> None:
        if self.process is None:
            return
        if self.process_group_id is not None:
            self.sent_sigkill_to_group = self._signal_process_group(signal.SIGKILL)
        elif self.process.poll() is None:
            self.process.kill()
        if self.process.poll() is None:
            self.process.wait(timeout=timeout_s)

    def _signal_process_group(self, signal_number: int) -> bool:
        if self.process_group_id is None:
            return False
        try:
            os.killpg(self.process_group_id, signal_number)
        except ProcessLookupError:
            return False
        except OSError as exc:
            self.stop_error = str(exc)
            return False
        return True

    def _close_logs(self) -> None:
        for handle in (self._stdout_handle, self._stderr_handle):
            if handle is not None and not handle.closed:
                handle.close()
        self._stdout_handle = None
        self._stderr_handle = None

    def returncode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait(self, timeout_s: float | None = None) -> int | None:
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout_s)
        finally:
            self._close_logs()

    def __enter__(self) -> "SimulationSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "log_dir": str(self.log_dir),
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "pid": self.process.pid if self.process else None,
            "process_group_id": self.process_group_id,
            "returncode": self.process.poll() if self.process else None,
            "start_error": self.start_error,
            "stop_error": self.stop_error,
            "sent_sigterm_to_group": self.sent_sigterm_to_group,
            "sent_sigkill_to_group": self.sent_sigkill_to_group,
            "running": self.running(),
        }


def build_default_launch_command(
    model_file: str = "mujoco/rbkairos.xml",
    urdf_file: str = "urdf/rbkairos.urdf",
    config_file: str = "config/mujoco_mecanum_drive.yaml",
    model_package: str = "rbkairos",
    traction_tendon_enabled: bool = False,
    traction_tendon_name: str = "traction_cable_x",
    traction_tendon_sensor_name: str = "traction_cable_x_limit_force",
    traction_tendon_attach_site_name: str = "traction_attach_x",
    traction_tendon_anchor_site_name: str = "traction_anchor_x",
    traction_tendon_publish_rate_hz: float = 100.0,
    traction_tendon_state_topic: str = "/traction_tendon/state",
    traction_tendon_wrench_topic: str = "/traction_tendon/wrench",
    traction_tendon_frame_id: str = "traction_attach_x",
) -> list[str]:
    enabled = "true" if traction_tendon_enabled else "false"
    return [
        "ros2",
        "launch",
        "rbkairos",
        "mujoco_mecanum_drive.launch.py",
        f"urdf_file:={urdf_file}",
        f"config_file:={config_file}",
        f"model_package:={model_package}",
        f"model_file:={model_file}",
        f"traction_tendon_enabled:={enabled}",
        f"traction_tendon_name:={traction_tendon_name}",
        f"traction_tendon_sensor_name:={traction_tendon_sensor_name}",
        f"traction_tendon_attach_site_name:={traction_tendon_attach_site_name}",
        f"traction_tendon_anchor_site_name:={traction_tendon_anchor_site_name}",
        f"traction_tendon_publish_rate_hz:={traction_tendon_publish_rate_hz}",
        f"traction_tendon_state_topic:={traction_tendon_state_topic}",
        f"traction_tendon_wrench_topic:={traction_tendon_wrench_topic}",
        f"traction_tendon_frame_id:={traction_tendon_frame_id}",
    ]
