"""
Pepper text-to-speech driver.

Provides an abstract interface for making Pepper speak, with two
implementations:

  - **SimulatedSpeechDriver**: Logs to console (no robot needed).
  - **PepperSpeechDriver**: Sends commands to a bridge script running
    on Pepper via SSH. Uses ``qi.Session()`` on the robot side — exactly
    like existing Pepper scripts.

The factory function ``create_speech_driver`` selects the appropriate
implementation based on the ``simulate`` flag.

Usage:
    driver = create_speech_driver(ip="192.168.10.196", port=9559, simulate=False)
    driver.say("I noticed you crossed your arms 3 times.")
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the bridge script that runs on Pepper.
_BRIDGE_SCRIPT = Path(__file__).parent / "pepper_bridge.py"


class SpeechDriver(ABC):
    """Abstract base class for Pepper text-to-speech."""

    @abstractmethod
    def say(self, text: str) -> None:
        """Speak the given text.

        This call should block until speech is complete (or a
        reasonable approximation for simulated mode).

        Args:
            text: The text for Pepper to speak.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the driver."""


class SimulatedSpeechDriver(SpeechDriver):
    """Simulated speech driver that logs text to the console.

    Used for development and testing without a physical Pepper robot.
    Estimates speech duration based on word count for realistic timing.
    """

    # Approximate speaking rate: words per second.
    _WORDS_PER_SECOND = 2.5

    def say(self, text: str) -> None:
        """Log the speech text and pause for estimated duration.

        Args:
            text: The text Pepper would speak.
        """
        word_count = len(text.split())
        duration = word_count / self._WORDS_PER_SECOND

        logger.info("[SIMULATED SPEECH] Pepper says: \"%s\"", text)
        logger.debug(
            "[SIMULATED SPEECH] Estimated duration: %.1fs (%d words)",
            duration,
            word_count,
        )
        time.sleep(duration)

    def close(self) -> None:
        """No resources to release in simulated mode."""


class PepperSpeechDriver(SpeechDriver):
    """Real Pepper speech driver using the bridge script over SSH.

    Deploys ``pepper_bridge.py`` to the robot and communicates with
    it via stdin/stdout over an SSH session. The bridge uses
    ``qi.Session()`` on Pepper (where the qi module is available),
    matching the exact pattern used in existing Pepper scripts.

    Supports both ``ALTextToSpeech.say`` and ``ALAnimatedSpeech.say``
    (which adds contextual body language to the speech).

    Args:
        ip:   Pepper's IP address.
        port: NAOqi port (default: 9559).
        use_animated: If True, use ALAnimatedSpeech for body language
                      during speech (like the existing Pepper scripts).
    """

    def __init__(
        self,
        ip: str,
        port: int = 9559,
        use_animated: bool = True,
    ) -> None:
        """Deploy the bridge script and start an SSH session to Pepper.

        Args:
            ip:           Pepper robot IP address.
            port:         NAOqi port number.
            use_animated: Use ALAnimatedSpeech (adds gestures to speech).

        Raises:
            ConnectionError: If unable to connect to the robot.
        """
        self._ip = ip
        self._port = port
        self._use_animated = use_animated
        self._bridge_proc: subprocess.Popen | None = None

        self._start_bridge()

    def _start_bridge(self) -> None:
        """Deploy and start the bridge script on Pepper via SSH.

        Raises:
            ConnectionError: If the bridge fails to start or connect.
        """
        bridge_content = _BRIDGE_SCRIPT.read_text()

        try:
            # Copy the bridge script to Pepper via SSH + cat.
            deploy_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                f"nao@{self._ip}",
                "cat > /home/nao/pepper_bridge.py",
            ]
            deploy_result = subprocess.run(
                deploy_cmd,
                input=bridge_content,
                capture_output=True,
                text=True,
                timeout=60.0,
            )

            if deploy_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to deploy bridge: {deploy_result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise ConnectionError(
                f"SSH connection to Pepper at {self._ip} timed out. "
                f"Make sure SSH keys are set up or use ssh-copy-id."
            )
        except FileNotFoundError:
            raise ConnectionError("SSH client not found on this system.")

        # Start the bridge script on Pepper.
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
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
        )

        # Wait for the bridge to connect.
        response = self._read_response(timeout=10.0)
        if response is None or response.get("status") != "ok":
            msg = response.get("message", "unknown error") if response else "no response"
            raise ConnectionError(
                f"Speech bridge failed to connect to NAOqi: {msg}"
            )

        logger.info(
            "Connected to Pepper speech via bridge at %s:%d",
            self._ip,
            self._port,
        )

    def _send_command(self, command: dict) -> dict | None:
        """Send a JSON command to the bridge and read the response.

        Args:
            command: Command dictionary to send.

        Returns:
            Response dictionary, or None on failure.
        """
        if self._bridge_proc is None or self._bridge_proc.poll() is not None:
            logger.error("Speech bridge process is not running.")
            return None

        try:
            line = json.dumps(command) + "\n"
            self._bridge_proc.stdin.write(line)
            self._bridge_proc.stdin.flush()
            return self._read_response(timeout=30.0)
        except (BrokenPipeError, OSError) as exc:
            logger.error("Speech bridge communication error: %s", exc)
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

    def say(self, text: str) -> None:
        """Speak the given text on Pepper.

        Uses ALAnimatedSpeech (with body language) or plain
        ALTextToSpeech depending on configuration.

        Args:
            text: The text for Pepper to speak.
        """
        logger.info("[PEPPER SPEECH] Speaking: \"%s\"", text)

        cmd_type = "animated_say" if self._use_animated else "say"
        response = self._send_command({
            "cmd": cmd_type,
            "text": text,
        })

        if response and response.get("status") != "ok":
            logger.error(
                "[PEPPER SPEECH] say failed: %s", response.get("message"),
            )

    def close(self) -> None:
        """Shut down the bridge."""
        if self._bridge_proc is not None and self._bridge_proc.poll() is None:
            try:
                self._send_command({"cmd": "exit"})
                self._bridge_proc.wait(timeout=5.0)
            except Exception:
                self._bridge_proc.kill()
            finally:
                self._bridge_proc = None

        logger.info("Pepper speech driver closed.")


def create_speech_driver(
    ip: str = "127.0.0.1",
    port: int = 9559,
    simulate: bool = True,
) -> SpeechDriver:
    """Factory function to create the appropriate speech driver.

    Args:
        ip:       Pepper robot IP address (ignored if simulate=True).
        port:     NAOqi port (ignored if simulate=True).
        simulate: If True, return a simulated driver that logs to console.

    Returns:
        A SpeechDriver instance.
    """
    if simulate:
        logger.info("Using simulated speech driver.")
        return SimulatedSpeechDriver()

    logger.info("Connecting to Pepper speech at %s:%d ...", ip, port)
    return PepperSpeechDriver(ip, port)
