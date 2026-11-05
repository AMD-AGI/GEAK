"""Optional trusted Docker runtime setup shared by workers and cold verification."""
import json
import os
from pathlib import Path
import subprocess


def runtime_options(name, config_path=None):
    path = config_path or os.environ.get("WORKLOAD_DOCKER_RUNTIME")
    if not path:
        return [], []
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Docker runtime configuration must be an object")
    values = {}
    for key in ("prepare", "docker_args", "entry_prefix"):
        args = config.get(key, [])
        if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
            raise ValueError("Docker runtime %s must be an argv list" % key)
        values[key] = [x.replace("{container}", name) for x in args]
    if values["prepare"]:
        subprocess.run(values["prepare"], check=True)
    return values["docker_args"], values["entry_prefix"]
