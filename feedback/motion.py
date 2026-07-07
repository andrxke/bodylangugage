"""
Pepper joint motion driver.

Provides an abstract interface for controlling Pepper's joints, with
two implementations:

  - **SimulatedMotionDriver**: Logs to console (no robot needed).
  - **PepperMotionDriver**: Sends commands to a bridge script running
    on Pepper via SSH. Uses ``qi.Session()`` on the robot side — exactly
    like existing Pepper scripts.

The factory function ``create_motion_driver`` selects the appropriate
implementation based on the ``simulate`` flag.

Usage:
    driver = create_motion_driver(ip="192.168.10.196", port=9559, simulate=False)
    driver.set_pose({"HeadYaw": 1.0, "LElbowRoll": -1.4}, speed=0.2)
    driver.go_neutral()
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from feedback.gestures import POSE_NEUTRAL
from feedback.ssh_utils import DEFAULT_PEPPER_PASSWORD, ssh_base_args, ssh_env

logger = logging.getLogger(__name__)

# Path to the bridge script that runs on Pepper.
_BRIDGE_SCRIPT = Path(__file__).parent / "pepper_bridge.py"


class MotionDriver(ABC):
    """Abstract base class for Pepper joint motion control."""

    @abstractmethod
    def set_pose(self, angles: dict[str, float], speed: float = 0.2) -> None:
        """Move Pepper's joints to the specified angles.

        Args:
            angles: Dictionary of joint_name → angle (radians).
            speed:  Motion speed fraction [0.0, 1.0]. Lower = smoother.
        """

    @abstractmethod
    def go_neutral(self, speed: float = 0.2) -> None:
        """Return Pepper to the neutral rest pose.

        Args:
            speed: Motion speed fraction [0.0, 1.0].
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources and put the robot to rest."""


class SimulatedMotionDriver(MotionDriver):
    """Simulated motion driver that logs joint angles to the console.

    Used for development and testing without a physical Pepper robot.
    """

    def set_pose(self, angles: dict[str, float], speed: float = 0.2) -> None:
        """Log the target pose to console.

        Args:
            angles: Dictionary of joint_name → angle (radians).
            speed:  Motion speed (logged but not used).
        """
        angle_strs = [f"{k}={v:+.2f}" for k, v in sorted(angles.items())]
        logger.info(
            "[SIMULATED MOTION] Setting pose (speed=%.2f): %s",
            speed,
            ", ".join(angle_strs),
        )
        # Simulate the time it takes for joints to move.
        move_time = (1.0 - speed) * 1.5  # Slower speed = longer wait.
        time.sleep(move_time)

    def go_neutral(self, speed: float = 0.2) -> None:
        """Log return to neutral pose.

        Args:
            speed: Motion speed (logged but not used).
        """
        logger.info("[SIMULATED MOTION] Returning to neutral pose.")
        self.set_pose(POSE_NEUTRAL, speed)

    def close(self) -> None:
        """No resources to release in simulated mode."""
        logger.info("[SIMULATED MOTION] Driver closed.")


class PepperMotionDriver(MotionDriver):
    """Real Pepper motion driver using the bridge script over SSH.

    Deploys ``pepper_bridge.py`` to the robot and communicates with
    it via stdin/stdout over an SSH session. The bridge uses
    ``qi.Session()`` on Pepper (where the qi module is available),
    matching the exact pattern used in existing Pepper scripts.

    Args:
        ip:   Pepper's IP address.
        port: NAOqi port (default: 9559).
    """

    def __init__(self, ip: str, port: int = 9559, password: str = DEFAULT_PEPPER_PASSWORD) -> None:
        """Deploy the bridge script and start an SSH session to Pepper.

        Args:
            ip:       Pepper robot IP address.
            port:     NAOqi port number.
            password: SSH password for the robot.

        Raises:
            ConnectionError: If unable to connect to the robot.
        """
        self._ip = ip
        self._port = port
        self._password = password
        self._bridge_proc: subprocess.Popen | None = None

        self._start_bridge()

    def _start_bridge(self) -> None:
        """Deploy and start the bridge script on Pepper via SSH.

        Raises:
            ConnectionError: If the bridge fails to start or connect.
        """
        # First, deploy the bridge script to Pepper.
        bridge_content = _BRIDGE_SCRIPT.read_text()

        try:
            # Copy the bridge script to Pepper via SSH + cat.
            deploy_cmd = ssh_base_args() + [
                f"nao@{self._ip}",
                "cat > /home/nao/pepper_bridge.py",
            ]
            deploy_result = subprocess.run(
                deploy_cmd,
                input=bridge_content,
                capture_output=True,
                text=True,
                timeout=60.0,
                env=ssh_env(self._password),
            )

            if deploy_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to deploy bridge: {deploy_result.stderr}"
                )

            logger.info("Bridge script deployed to Pepper at %s.", self._ip)

        except subprocess.TimeoutExpired:
            raise ConnectionError(
                f"SSH connection to Pepper at {self._ip} timed out. "
                f"Make sure SSH keys are set up or use ssh-copy-id."
            )
        except FileNotFoundError:
            raise ConnectionError("SSH client not found on this system.")

        # Start the bridge script on Pepper as a persistent process.
        ssh_cmd = ssh_base_args() + [
            f"nao@{self._ip}",
            f"source /etc/profile; source ~/.profile; python /home/nao/pepper_bridge.py "
            f"--ip 127.0.0.1 --port {self._port}",
        ]

        self._bridge_proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=ssh_env(self._password),
        )

        # Wait for the bridge to connect and respond.
        response = self._read_response(timeout=10.0)
        if response is None or response.get("status") != "ok":
            msg = response.get("message", "unknown error") if response else "no response"
            raise ConnectionError(
                f"Bridge failed to connect to NAOqi: {msg}"
            )

        logger.info(
            "Connected to Pepper via bridge at %s:%d", self._ip, self._port,
        )

        # Wake up the robot.
        self._send_command({"cmd": "wakeup"})
        logger.info("Pepper is awake.")

    def _send_command(self, command: dict) -> dict | None:
        """Send a JSON command to the bridge and read the response.

        Args:
            command: Command dictionary to send.

        Returns:
            Response dictionary, or None on failure.
        """
        if self._bridge_proc is None or self._bridge_proc.poll() is not None:
            logger.error("Bridge process is not running.")
            return None

        try:
            line = json.dumps(command) + "\n"
            self._bridge_proc.stdin.write(line)
            self._bridge_proc.stdin.flush()
            return self._read_response(timeout=30.0)
        except (BrokenPipeError, OSError) as exc:
            logger.error("Bridge communication error: %s", exc)
            return None

    def _read_response(self, timeout: float = 10.0) -> dict | None:
        """Read a single JSON response from the bridge.

        Skips extraneous stdout lines (like NAOqi SDK warnings).

        Args:
            timeout: Max seconds to wait.

        Returns:
            Parsed response dict, or None on timeout/error.
        """
        import select

        if self._bridge_proc is None:
            return None

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning("Bridge response timed out after %.1fs.", timeout)
                return None

            ready, _, _ = select.select(
                [self._bridge_proc.stdout], [], [], timeout - elapsed,
            )
            if not ready:
                continue

            line = self._bridge_proc.stdout.readline().strip()
            if not line:
                return None

            if line.startswith("{"):
                try:
                    return json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            logger.debug("Bridge stdout (ignored): %s", line)

    def set_pose(self, angles: dict[str, float], speed: float = 0.2) -> None:
        """Move Pepper's joints to the specified angles.

        Args:
            angles: Dictionary of joint_name → angle (radians).
            speed:  Motion speed fraction [0.0, 1.0].
        """
        logger.info(
            "[PEPPER MOTION] Setting %d joints at speed=%.2f",
            len(angles),
            speed,
        )

        response = self._send_command({
            "cmd": "set_angles",
            "names": list(angles.keys()),
            "angles": list(angles.values()),
            "speed": speed,
        })

        if response and response.get("status") != "ok":
            logger.error(
                "[PEPPER MOTION] set_angles failed: %s",
                response.get("message"),
            )

    def go_neutral(self, speed: float = 0.2) -> None:
        """Return Pepper to the neutral rest pose.

        Args:
            speed: Motion speed fraction [0.0, 1.0].
        """
        logger.info("[PEPPER MOTION] Returning to neutral pose.")
        self.set_pose(POSE_NEUTRAL, speed)

    def close(self) -> None:
        """Send rest command and shut down the bridge."""
        if self._bridge_proc is not None and self._bridge_proc.poll() is None:
            try:
                self._send_command({"cmd": "rest"})
                self._send_command({"cmd": "exit"})
                self._bridge_proc.wait(timeout=5.0)
            except Exception:
                self._bridge_proc.kill()
            finally:
                self._bridge_proc = None

        logger.info("Pepper motion driver closed.")


def create_motion_driver(
    ip: str = "127.0.0.1",
    port: int = 9559,
    simulate: bool = True,
    password: str = DEFAULT_PEPPER_PASSWORD,
) -> MotionDriver:
    """Factory function to create the appropriate motion driver.

    Args:
        ip:       Pepper robot IP address (ignored if simulate=True).
        port:     NAOqi port (ignored if simulate=True).
        simulate: If True, return a simulated driver that logs to console.
        password: SSH password for the robot.

    Returns:
        A MotionDriver instance.
    """
    if simulate:
        logger.info("Using simulated motion driver.")
        return SimulatedMotionDriver()

    logger.info("Connecting to Pepper motion at %s:%d ...", ip, port)
    return PepperMotionDriver(ip, port, password)
