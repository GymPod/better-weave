"""Better Weave on Modal.

Architecture:
  - sync.py runs locally (has access to internal W&B) → pushes data to Modal volume
  - This Modal app reads from volume + calls Bedrock directly (AWS is public)

Deploy:  modal deploy modal_app.py
Sync:    python sync.py
"""

import os
import configparser

import modal

app = modal.App("better-weave")
volume = modal.Volume.from_name("better-weave-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "boto3",
        "modal",
        "wandb",
        "requests",
    )
    .add_local_dir(".", "/app/better_weave", ignore=[
        "__pycache__", "*.pyc", ".git", "sync.py", "modal_app.py", "run.sh",
    ])
)

# AWS credentials for Bedrock
_secret_env: dict[str, str] = {}
# Try ~/.aws/credentials
try:
    config = configparser.ConfigParser()
    aws_creds_file = os.path.expanduser("~/.aws/credentials")
    if os.path.exists(aws_creds_file):
        config.read(aws_creds_file)
        for profile_name in ["agi-emerge-dev", "default"]:
            if profile_name in config:
                for key in ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]:
                    if key in config[profile_name]:
                        _secret_env[key.upper()] = config[profile_name][key]
                break
except Exception:
    pass

# Also try env vars
for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
    val = os.environ.get(key)
    if val:
        _secret_env[key] = val

# Bearer token for Bedrock (used by some corp setups)
bearer = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
if bearer:
    _secret_env["AWS_BEARER_TOKEN_BEDROCK"] = bearer

_secret_env.setdefault("AWS_DEFAULT_REGION", "us-east-1")

secrets_list = [modal.Secret.from_dict(_secret_env)] if _secret_env else []


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400,
    min_containers=1,
    secrets=secrets_list,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8421, startup_timeout=30)
def web():
    import subprocess

    env = dict(os.environ)
    env["BETTER_WEAVE_DATA"] = "/data/better_weave"

    subprocess.Popen(
        ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8421"],
        cwd="/app/better_weave",
        env=env,
    )
