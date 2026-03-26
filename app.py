"""Better Weave — Live trajectory viewer for W&B runs.

Fetches data from W&B on demand. Caches run detail, history, and traces
locally in /tmp to avoid redundant API calls.

AI assistant powered by Bedrock Claude.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import boto3
import requests as _requests
import wandb as _wandb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Better Weave")

CACHE_DIR = Path(os.environ.get("BETTER_WEAVE_CACHE", "/tmp/better_weave_cache"))
WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://wandb.agi.amazon.dev")
DEFAULT_ENTITY = os.environ.get("BETTER_WEAVE_ENTITY", "autonomy")
DEFAULT_PROJECT = os.environ.get("BETTER_WEAVE_PROJECT", "slime-agent")


# ---------- Cache helpers ----------


def _cache_path(entity: str, project: str, *parts: str) -> Path:
    return CACHE_DIR / f"{entity}_{project}" / Path(*parts)


def _read_cache(entity: str, project: str, *parts: str) -> Any:
    p = _cache_path(entity, project, *parts)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _write_cache(entity: str, project: str, data: Any, *parts: str) -> None:
    p = _cache_path(entity, project, *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, default=str))


# ---------- W&B helpers ----------


_wandb_reachable_cache: list = []  # [timestamp, bool]


def _wandb_reachable() -> bool:
    now = time.time()
    if _wandb_reachable_cache and now - _wandb_reachable_cache[0] < 600:
        return _wandb_reachable_cache[1]
    try:
        resp = _requests.get(f"{WANDB_BASE_URL}/healthz", timeout=2)
        ok = resp.status_code < 500
    except Exception:
        ok = False
    _wandb_reachable_cache.clear()
    _wandb_reachable_cache.extend([now, ok])
    return ok


def _get_wandb_auth() -> dict[str, str]:
    import base64
    import netrc
    try:
        n = netrc.netrc()
        host = WANDB_BASE_URL.replace("https://", "").replace("http://", "")
        creds = n.authenticators(host)
        if creds:
            auth_b64 = base64.b64encode(f"api:{creds[2]}".encode()).decode()
            return {"Authorization": f"Basic {auth_b64}", "Content-Type": "application/json"}
    except Exception:
        pass
    return {}


# ---------- W&B fetch functions ----------


def _fetch_runs(entity: str, project: str, limit: int, state: str | None, search: str | None) -> list[dict[str, Any]]:
    import re
    headers = _get_wandb_auth()
    filters: dict[str, Any] = {}
    if state:
        filters["state"] = state
    if search:
        filters["display_name"] = {"$regex": f".*{re.escape(search)}.*"}

    query = """
    query Runs($project: String!, $entity: String!, $limit: Int!, $filters: JSONString) {
      project(name: $project, entityName: $entity) {
        runs(first: $limit, order: "-createdAt", filters: $filters) {
          edges { node { name displayName state historyLineCount createdAt heartbeatAt tags } }
        }
      }
    }
    """
    resp = _requests.post(
        f"{WANDB_BASE_URL}/graphql",
        json={"query": query, "variables": {
            "project": project, "entity": entity, "limit": limit,
            "filters": json.dumps(filters) if filters else None,
        }},
        headers=headers, timeout=10,
    )
    resp.raise_for_status()
    edges = resp.json().get("data", {}).get("project", {}).get("runs", {}).get("edges", [])
    return [
        {
            "id": e["node"]["name"],
            "display_name": e["node"]["displayName"],
            "state": e["node"]["state"],
            "history_count": e["node"]["historyLineCount"],
            "created_at": e["node"]["createdAt"],
            "heartbeat_at": e["node"].get("heartbeatAt"),
            "tags": e["node"].get("tags", []),
        }
        for e in edges
    ]


def _fetch_run_detail(entity: str, project: str, run_id: str) -> dict[str, Any]:
    api = _wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    config = dict(run.config) if run.config else {}
    summary: dict[str, Any] = {}
    try:
        for k, v in dict(run.summary).items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                summary[k] = v
            except (TypeError, ValueError):
                summary[k] = str(v)
    except Exception:
        pass
    return {
        "id": run.id, "name": run.name, "display_name": run.display_name,
        "state": run.state, "config": config, "summary": summary,
        "url": run.url, "created_at": str(run.created_at),
    }


def _fetch_history(entity: str, project: str, run_id: str, samples: int = 500) -> dict[str, Any]:
    headers = _get_wandb_auth()
    query = """
    query RunHistory($project: String!, $entity: String!, $name: String!, $samples: Int!) {
      project(name: $project, entityName: $entity) {
        run(name: $name) { history(samples: $samples) }
      }
    }
    """
    resp = _requests.post(
        f"{WANDB_BASE_URL}/graphql",
        json={"query": query, "variables": {
            "project": project, "entity": entity, "name": run_id, "samples": samples,
        }},
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    history = resp.json().get("data", {}).get("project", {}).get("run", {}).get("history", [])
    rows = [json.loads(r) if isinstance(r, str) else r for r in history]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    return {"rows": rows, "keys": sorted(all_keys), "count": len(rows)}


def _list_trace_versions(entity: str, project: str, run_id: str) -> list[dict[str, Any]]:
    """List available trajectory artifact versions (fast — no download)."""
    api = _wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    all_arts = list(run.logged_artifacts())
    traj_arts = [a for a in all_arts if "TrajectoryDetails" in a.name]
    traj_arts.sort(key=lambda a: a.version)
    return [{"version": a.version, "name": a.name} for a in traj_arts]


def _parse_trace_artifact(art, entity: str, project: str, run_id: str) -> list[dict]:
    """Download and parse a single trajectory artifact into trace rows."""
    dl_path = art.download(f"/tmp/better_weave_artifacts/{entity}_{project}/{run_id}/{art.version}")
    table_files = list(Path(dl_path).glob("*.table.json"))
    if not table_files:
        return []
    with open(table_files[0]) as f:
        table = json.load(f)
    columns = table.get("columns", [])
    traces = []
    for row in table.get("data", []):
        row_dict = dict(zip(columns, row))
        traces.append({
            "id": f"{run_id}-v{art.version}-{len(traces)}",
            "step": row_dict.get("step"),
            "called": row_dict.get("called", ""),
            "dataset_name": row_dict.get("dataset_name", ""),
            "status": row_dict.get("status", ""),
            "prompt": str(row_dict.get("prompt", ""))[:500],
            "label": str(row_dict.get("label", ""))[:500] if row_dict.get("label") else "",
            "trajectory": row_dict.get("trajectory", ""),
            "reward": row_dict.get("reward"),
            "env_reward": row_dict.get("env_reward"),
            "failure_reason": row_dict.get("failure_reason", ""),
            "error": row_dict.get("error", ""),
            "n_turns": row_dict.get("num_steps", 0),
            "num_tool_calls": row_dict.get("num_tool_calls", 0),
            "response_length": row_dict.get("response_length", 0),
            "verifier_outcome": row_dict.get("verifier_outcome", ""),
            "grader_responses": row_dict.get("grader_responses", ""),
            "duration": row_dict.get("duration", 0),
            "time_env_create": row_dict.get("time_env_create", 0),
            "time_agent_invoke": row_dict.get("time_agent_invoke", 0),
            "time_verify": row_dict.get("time_verify", 0),
            "time_cleanup": row_dict.get("time_cleanup", 0),
            "artifact_version": art.version,
        })
    return traces


def _fetch_trace_versions(entity: str, project: str, run_id: str, versions: list[int]) -> dict[str, Any]:
    """Fetch specific trajectory artifact versions."""
    api = _wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    all_arts = list(run.logged_artifacts())
    traj_arts = [a for a in all_arts if "TrajectoryDetails" in a.name]
    traj_arts.sort(key=lambda a: a.version)
    version_set = set(versions)
    selected = [a for a in traj_arts if a.version in version_set]
    logger.info(f"Fetching versions {versions} ({len(selected)} found) for {run_id}")

    traces: list[dict] = []
    for art in selected:
        try:
            rows = _parse_trace_artifact(art, entity, project, run_id)
            traces.extend(rows)
            logger.info(f"  v{art.version}: {len(rows)} rows")
        except Exception as e:
            logger.warning(f"  Failed to fetch artifact v{art.version}: {e}")

    return {"traces": traces, "count": len(traces), "source": "artifacts"}


# ---------- API Routes ----------


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "entity": DEFAULT_ENTITY,
        "project": DEFAULT_PROJECT,
        "wandb_base_url": WANDB_BASE_URL,
    }


@app.get("/api/runs")
def list_runs(
    limit: int = Query(default=50, le=200),
    state: str | None = None,
    search: str | None = None,
    entity: str | None = None,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """List runs — always live from W&B."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT
    try:
        return _fetch_runs(ent, proj, limit, state, search)
    except Exception as e:
        logger.warning(f"Fetch runs failed: {e}")
        return []


@app.get("/api/runs/{run_id}")
def get_run(
    run_id: str,
    entity: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Get run detail. Cached after first fetch."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT

    cached = _read_cache(ent, proj, f"{run_id}.json")
    if cached:
        return cached

    try:
        data = _fetch_run_detail(ent, proj, run_id)
        _write_cache(ent, proj, data, f"{run_id}.json")
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found: {e}")


@app.get("/api/runs/{run_id}/history")
def get_run_history(
    run_id: str,
    entity: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Get run history. Cached after first fetch."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT

    cached = _read_cache(ent, proj, f"{run_id}_history.json")
    if cached:
        return cached

    try:
        data = _fetch_history(ent, proj, run_id)
        _write_cache(ent, proj, data, f"{run_id}_history.json")
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"History for {run_id} not available: {e}")


_traces_loading: dict[str, float] = {}  # "run_id" or "run_id:v3" -> start timestamp


def _bg_list_versions(ent: str, proj: str, run_id: str) -> None:
    """Background: list available versions, then auto-fetch first + last."""
    key = run_id
    try:
        versions = _list_trace_versions(ent, proj, run_id)
        meta = {"versions": versions, "loaded_versions": [], "source": "versions_listed"}
        _write_cache(ent, proj, meta, f"{run_id}_traces_meta.json")
        logger.info(f"Background: {run_id} has {len(versions)} trace versions")

        if versions:
            # Auto-fetch first (step 0) and last version
            to_fetch = [versions[0]["version"]]
            if len(versions) > 1:
                to_fetch.append(versions[-1]["version"])
            data = _fetch_trace_versions(ent, proj, run_id, to_fetch)
            data["versions"] = versions
            data["loaded_versions"] = to_fetch
            _write_cache(ent, proj, data, f"{run_id}_traces.json")
            logger.info(f"Background: loaded v{to_fetch} for {run_id} ({data['count']} traces)")
        else:
            _write_cache(ent, proj,
                         {"traces": [], "count": 0, "source": "no_artifacts", "versions": [], "loaded_versions": []},
                         f"{run_id}_traces.json")
    except Exception as e:
        logger.warning(f"Background list versions failed for {run_id}: {e}")
        _write_cache(ent, proj,
                     {"traces": [], "count": 0, "source": "error", "versions": [], "loaded_versions": []},
                     f"{run_id}_traces.json")
    finally:
        _traces_loading.pop(key, None)


def _bg_fetch_version(ent: str, proj: str, run_id: str, version: int) -> None:
    """Background: fetch a single version and merge into cached traces."""
    key = f"{run_id}:v{version}"
    try:
        data = _fetch_trace_versions(ent, proj, run_id, [version])
        # Merge into existing cache
        cached = _read_cache(ent, proj, f"{run_id}_traces.json") or {
            "traces": [], "count": 0, "versions": [], "loaded_versions": []
        }
        cached["traces"].extend(data["traces"])
        cached["count"] = len(cached["traces"])
        loaded = cached.get("loaded_versions", [])
        if version not in loaded:
            loaded.append(version)
            loaded.sort()
        cached["loaded_versions"] = loaded
        cached["source"] = "artifacts"
        _write_cache(ent, proj, cached, f"{run_id}_traces.json")
        logger.info(f"Background: loaded v{version} for {run_id} (+{data['count']} traces)")
    except Exception as e:
        logger.warning(f"Background fetch v{version} failed for {run_id}: {e}")
    finally:
        _traces_loading.pop(key, None)


@app.get("/api/runs/{run_id}/traces")
def get_run_traces(
    run_id: str,
    entity: str | None = None,
    project: str | None = None,
    version: int | None = None,
) -> dict[str, Any]:
    """Get traces. First call lists versions + loads first/last. Use ?version=N to load more."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT

    # Request to load a specific version
    if version is not None:
        vkey = f"{run_id}:v{version}"
        cached = _read_cache(ent, proj, f"{run_id}_traces.json")
        if cached and version in cached.get("loaded_versions", []):
            return cached  # already loaded

        if vkey in _traces_loading:
            if time.time() - _traces_loading[vkey] < 120:
                return cached or {"traces": [], "count": 0, "source": "loading"}
            _traces_loading.pop(vkey, None)

        if _wandb_reachable():
            _traces_loading[vkey] = time.time()
            threading.Thread(target=_bg_fetch_version, args=(ent, proj, run_id, version), daemon=True).start()
        return cached or {"traces": [], "count": 0, "source": "loading"}

    # Default: return cached data or start initial load
    cached = _read_cache(ent, proj, f"{run_id}_traces.json")
    if cached and cached.get("source") not in (None, "loading"):
        return cached

    key = run_id
    if key in _traces_loading:
        if time.time() - _traces_loading[key] < 120:
            return {"traces": [], "count": 0, "source": "loading", "versions": [], "loaded_versions": []}
        _traces_loading.pop(key, None)

    if _wandb_reachable():
        _traces_loading[key] = time.time()
        threading.Thread(target=_bg_list_versions, args=(ent, proj, run_id), daemon=True).start()
        return {"traces": [], "count": 0, "source": "loading", "versions": [], "loaded_versions": []}

    return {"traces": [], "count": 0, "source": "not_available", "versions": [], "loaded_versions": []}


# ---------- AI Assistant ----------


class AskRequest(BaseModel):
    question: str
    run_id: str | None = None
    context: str | None = None
    entity: str | None = None
    project: str | None = None


class AskResponse(BaseModel):
    answer: str


@app.post("/api/ask")
def ask_assistant(req: AskRequest) -> AskResponse:
    ent = req.entity or DEFAULT_ENTITY
    proj = req.project or DEFAULT_PROJECT
    context_parts: list[str] = []

    if req.run_id:
        try:
            run_data = get_run(req.run_id, entity=ent, project=proj)
            context_parts.append(f"Run: {run_data.get('display_name', run_data.get('id'))} (state={run_data.get('state')})")
            config = run_data.get("config", {})
            important_keys = ["load", "lr", "min_lr", "sft", "bf16", "seed", "rm_type"]
            config_str = ", ".join(f"{k}={config[k]}" for k in important_keys if k in config)
            if config_str:
                context_parts.append(f"Config: {config_str}")
        except HTTPException:
            pass

        try:
            hist = get_run_history(req.run_id, entity=ent, project=proj)
            if hist.get("rows"):
                last_row = hist["rows"][-1]
                eval_metrics = {k: v for k, v in last_row.items() if "eval" in k and v is not None}
                if eval_metrics:
                    context_parts.append(f"Latest eval metrics: {json.dumps(eval_metrics, default=str)}")
                rollout_metrics = {k: v for k, v in last_row.items() if "rollout" in k and v is not None}
                if rollout_metrics:
                    context_parts.append(f"Latest rollout metrics: {json.dumps(rollout_metrics, default=str)}")
        except HTTPException:
            pass

    if req.context:
        context_parts.append(f"Additional context: {req.context}")

    system_prompt = """You are an AI assistant for analyzing ML training runs, specifically SLIME RL training for language models.
You help users understand their training metrics, diagnose issues, and interpret trajectories.

Key concepts:
- SLIME is an RL training framework for LLMs (GRPO, PPO, etc.)
- Rollouts are agent interactions with environments (terminal tasks, web tasks, coding tasks)
- Eval metrics show model performance on benchmark datasets
- reward = success rate, higher is better
- failure_reason breakdown shows why rollouts fail (max_iterations, agent_timeout, container_timeout, etc.)
- verifier_outcome shows Correct/Incorrect/ParseError/Truncated/VerifierError

When analyzing trajectories:
- You have access to the FULL trajectory log with all turns and tool calls
- Analyze the agent's actual behavior step by step
- Identify inefficiencies: unnecessary retries, wrong approaches, wasted turns
- Point out good patterns (systematic debugging, verification) and bad patterns (brute force, ignoring errors)
- For failed trajectories, pinpoint exactly where and why the agent went wrong

Be concise and actionable."""

    user_msg = req.question
    if context_parts:
        user_msg = "Context:\n" + "\n".join(context_parts) + "\n\nQuestion: " + req.question

    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        response = client.converse(
            modelId="us.anthropic.claude-opus-4-6-v1",
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": 4096, "temperature": 0.3},
        )
        return AskResponse(answer=response["output"]["message"]["content"][0]["text"])
    except Exception as e:
        logger.error(f"Bedrock call failed: {traceback.format_exc()}")
        return AskResponse(answer=f"AI assistant error: {e}")


# ---------- Static Files ----------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index() -> HTMLResponse:
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(content=f.read())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8421)
