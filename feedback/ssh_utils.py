"""
SSH helper utilities for Pepper robot communication.

Wraps SSH and SCP commands so that the robot password is fed
automatically via the ``SSH_ASKPASS`` mechanism, removing the need
for manual password entry on every connection.

The default password for Pepper/NAO robots is ``"nao"``.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Default password for Pepper/NAO robots.
DEFAULT_PEPPER_PASSWORD = "nao"


def _create_askpass_script(password: str) -> str:
    """Create a temporary script that echoes the SSH password.

    The script is used with ``SSH_ASKPASS`` so that ``ssh`` never prompts
    the user interactively.

    Args:
        password: The password to embed in the script.

    Returns:
        Absolute path to the temporary askpass script.
    """
    fd, path = tempfile.mkstemp(prefix="pepper_askpass_", suffix=".sh")
    with os.fdopen(fd, "w") as f:
        f.write(f"#!/bin/sh\necho '{password}'\n")
    os.chmod(path, stat.S_IRWXU)  # Owner read/write/execute.
    return path


def ssh_env(password: str) -> dict[str, str]:
    """Build an environment dict that auto-feeds the SSH password.

    Uses ``SSH_ASKPASS`` + ``SSH_ASKPASS_REQUIRE=force`` so that the
    ``ssh`` client never prompts interactively — even when there is a
    TTY attached.

    Args:
        password: SSH password for the robot.

    Returns:
        Environment dictionary suitable for ``subprocess.run/Popen``.
    """
    askpass_script = _create_askpass_script(password)

    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass_script
    env["SSH_ASKPASS_REQUIRE"] = "force"
    # DISPLAY must be set for SSH_ASKPASS to trigger on some systems.
    env.setdefault("DISPLAY", ":0")
    return env


def ssh_base_args() -> list[str]:
    """Return common SSH options used for all Pepper connections.

    Returns:
        List of SSH option arguments.
    """
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]
