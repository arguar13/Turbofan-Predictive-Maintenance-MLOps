import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from config_schema import AppConfig

# A whole value of the form "${VAR}"
_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}^{]+)\}$")


# Defaults for local runs (docker-compose.yml). In Kubernetes every variable is
# injected by kubernetes/base/configmap.yaml, so these only apply on a laptop.
#
# The value must be a URI WITH a scheme: MLflow rejects a bare host:port with
# UnsupportedModelRegistryStoreURIException, which would break `make smoke-test`
# and `make train-toy`.
_LOCAL_DEFAULTS = {
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
}


def _resolve_env_placeholders(value: Any) -> Any:
    """Recursively replace "${VAR}" placeholders with environment variables.

    Works on the parsed values, not on YAML nodes, so it does not matter whether
    the value was quoted: PyYAML never applies implicit resolvers to quoted
    scalars, so an `add_implicit_resolver` approach would never fire for
    `mlflow_tracking_uri: "${MLFLOW_TRACKING_URI}"`.
    """
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.match(value)
        if match is None:
            return value
        env_var = match.group(1)
        resolved = os.environ.get(env_var)
        if resolved is not None:
            return resolved
        if env_var not in _LOCAL_DEFAULTS:
            # A misspelled placeholder must fail loudly, not resolve silently
            # to some arbitrary default.
            raise ValueError(
                f"Environment variable {env_var!r}, referenced in config.yaml, is not set "
                f"and has no local default. Variables with a default: "
                f"{sorted(_LOCAL_DEFAULTS)}."
            )
        return _LOCAL_DEFAULTS[env_var]
    return value


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load config.yaml, resolve environment placeholders and validate it.

    A missing key, wrong type or out-of-range value fails at startup against
    `AppConfig`, instead of surfacing later as a cryptic KeyError in the middle
    of a request or a training run.
    """
    base_dir = Path(__file__).resolve().parent.parent
    full_path = os.path.join(base_dir, config_path)

    with open(full_path) as file:
        raw_config = yaml.safe_load(file)

    resolved_config = _resolve_env_placeholders(raw_config)

    try:
        validated = AppConfig.model_validate(resolved_config)
    except ValidationError as exc:
        raise ValueError(f"Invalid config.yaml ({full_path}):\n{exc}") from exc

    return validated.model_dump()
