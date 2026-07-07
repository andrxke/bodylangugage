#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
"""
Pepper feedback bridge — runs ON the robot.

This script runs on Pepper (where the qi module is natively available)
and listens for commands on stdin. The main pipeline (Python 3) sends
commands via SSH, and this script executes them using the NAOqi API.

Commands are single-line JSON objects:
    {"cmd": "say", "text": "Hello"}
    {"cmd": "set_angles", "names": [...], "angles": [...], "speed": 0.2}
    {"cmd": "wakeup"}
    {"cmd": "rest"}
    {"cmd": "animated_say", "text": "Hello"}
    {"cmd": "exit"}

Responds with single-line JSON:
    {"status": "ok"}
    {"status": "error", "message": "..."}

Usage (deployed to Pepper, run via SSH):
    ssh nao@<pepper-ip> python /home/nao/pepper_bridge.py

Standalone test (on Pepper):
    echo '{"cmd": "say", "text": "Hello"}' | python pepper_bridge.py
"""

import json
import os
import sys
import time

# Add common NAOqi Python paths for non-interactive SSH sessions where
# PYTHONPATH might not be fully populated.
_NAOQI_PATHS = [
    "/opt/aldebaran/lib/python2.7/site-packages",
    "/usr/lib/python2.7/site-packages",
    "/home/nao/.local/lib/python2.7/site-packages",
]
for p in _NAOQI_PATHS:
    if p not in sys.path:
        sys.path.append(p)

try:
    import qi
    # Suppress C++ SDK warnings from printing to stdout and corrupting JSON.
    try:
        qi.logging.setLevel(qi.logging.FATAL)
    except Exception:
        pass
except ImportError:
    print(json.dumps({"status": "error", "message": "qi module not found. sys.path: " + str(sys.path)}))
    sys.exit(1)


def respond(status, message=None):
    """Write a JSON response to stdout."""
    resp = {"status": status}
    if message is not None:
        resp["message"] = message
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def main(ip, port):
    """Connect to NAOqi and process commands from stdin."""
    session = qi.Session()
    try:
        session.connect("tcp://" + ip + ":" + str(port))
    except RuntimeError:
        respond("error", "Cannot connect to NAOqi at %s:%d" % (ip, port))
        sys.exit(1)

    motion = session.service("ALMotion")
    tts = session.service("ALTextToSpeech")

    # Try to get ALAnimatedSpeech (may not be available on all versions).
    try:
        animated_speech = session.service("ALAnimatedSpeech")
    except Exception:
        animated_speech = None

    respond("ok", "connected")

    # Process commands from stdin.
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except (ValueError, TypeError):
            respond("error", "Invalid JSON: " + line)
            continue

        try:
            action = cmd.get("cmd", "")

            if action == "say":
                tts.say(str(cmd["text"]))
                respond("ok")

            elif action == "animated_say":
                if animated_speech is not None:
                    animated_speech.say(str(cmd["text"]))
                else:
                    tts.say(str(cmd["text"]))
                respond("ok")

            elif action == "set_angles":
                names = cmd["names"]
                angles = [float(a) for a in cmd["angles"]]
                speed = float(cmd.get("speed", 0.2))
                motion.setAngles(names, angles, speed)
                respond("ok")

            elif action == "wakeup":
                motion.wakeUp()
                respond("ok")

            elif action == "rest":
                motion.rest()
                respond("ok")

            elif action == "exit":
                respond("ok", "goodbye")
                break

            else:
                respond("error", "Unknown command: " + action)

        except Exception as e:
            respond("error", str(e))

    session.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP (use 127.0.0.1 when running on robot)")
    parser.add_argument("--port", type=int, default=9559)
    args = parser.parse_args()
    main(args.ip, args.port)
