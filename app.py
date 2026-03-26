"""Better Weave - Modern trajectory viewer for W&B runs.

Supports two modes:
- Local mode (python app.py): fetches from W&B on demand, works for any entity/project
- Modal mode (modal deploy): serves cached data from volume, synced via sync.py

AI assistant calls Bedrock directly (public AWS endpoint).
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import boto3
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Try to import wandb for live/on-demand mode
try:
    import requests as _requests
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Better Weave")

DATA_DIR = Path(os.environ.get("BETTER_WEAVE_DATA", "/tmp/better_weave_sync"))
WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://wandb.agi.amazon.dev")
DEFAULT_ENTITY = os.environ.get("BETTER_WEAVE_ENTITY", "autonomy")
DEFAULT_PROJECT = os.environ.get("BETTER_WEAVE_PROJECT", "slime-agent")


def _read_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))


def _project_data_dir(entity: str, project: str) -> Path:
    """Data directory for a specific entity/project."""
    if entity == DEFAULT_ENTITY and project == DEFAULT_PROJECT:
        return DATA_DIR
    return DATA_DIR / "projects" / f"{entity}_{project}"


# ---------- W&B Live Fetch Functions ----------


_wandb_reachable_result: list = []  # [timestamp, bool]

def _wandb_reachable() -> bool:
    """Quick check if W&B server is reachable. Caches result for 10 minutes."""
    import time
    now = time.time()
    if _wandb_reachable_result and now - _wandb_reachable_result[0] < 600:
        return _wandb_reachable_result[1]
    try:
        resp = _requests.get(f"{WANDB_BASE_URL}/healthz", timeout=2)
        ok = resp.status_code < 500
    except Exception:
        ok = False
    _wandb_reachable_result.clear()
    _wandb_reachable_result.extend([now, ok])
    if not ok:
        logger.info(f"W&B at {WANDB_BASE_URL} not reachable, skipping live fetch")
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


def _live_fetch_runs(entity: str, project: str, limit: int, state: str | None, search: str | None) -> list[dict[str, Any]] | None:
    """Fetch runs from W&B GraphQL. Returns None if W&B is unreachable."""
    if not _WANDB_AVAILABLE or not _wandb_reachable():
        return None
    try:
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
              edges {
                node { name displayName state historyLineCount createdAt heartbeatAt tags }
              }
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
    except Exception as e:
        logger.warning(f"Live fetch runs failed: {e}")
        return None


def _live_fetch_run_detail(entity: str, project: str, run_id: str) -> dict[str, Any] | None:
    """Fetch run detail from W&B SDK. Returns None if unavailable."""
    if not _WANDB_AVAILABLE or not _wandb_reachable():
        return None
    try:
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
    except Exception as e:
        logger.warning(f"Live fetch run detail failed: {e}")
        return None


def _live_fetch_history(entity: str, project: str, run_id: str, samples: int = 500) -> dict[str, Any] | None:
    """Fetch run history from W&B GraphQL. Returns None if unavailable."""
    if not _WANDB_AVAILABLE or not _wandb_reachable():
        return None
    try:
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
    except Exception as e:
        logger.warning(f"Live fetch history failed: {e}")
        return None


def _live_fetch_traces(entity: str, project: str, run_id: str, max_versions: int = 10) -> dict[str, Any] | None:
    """Fetch trajectory artifacts from W&B on demand. Returns None if unavailable."""
    if not _WANDB_AVAILABLE or not _wandb_reachable():
        return None
    try:
        api = _wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        all_arts = list(run.logged_artifacts())
        traj_arts = [a for a in all_arts if "TrajectoryDetails" in a.name]
        if not traj_arts:
            return {"traces": [], "count": 0, "source": "no_artifacts"}

        traj_arts.sort(key=lambda a: a.version)
        selected = traj_arts[-max_versions:]
        logger.info(f"Fetching {len(selected)}/{len(traj_arts)} trajectory artifacts for {run_id}")

        traces: list[dict] = []
        for art in selected:
            try:
                dl_path = art.download(f"/tmp/better_weave_artifacts/{entity}_{project}/{run_id}/{art.version}")
                table_files = list(Path(dl_path).glob("*.table.json"))
                if not table_files:
                    continue
                with open(table_files[0]) as f:
                    table = json.load(f)
                columns = table.get("columns", [])
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
                logger.info(f"  v{art.version}: {len(table.get('data', []))} rows")
            except Exception as e:
                logger.warning(f"  Failed to fetch artifact v{art.version}: {e}")

        return {"traces": traces, "count": len(traces), "source": "artifacts"}
    except Exception as e:
        logger.warning(f"Live fetch traces failed: {e}")
        return None


# ---------- API Routes ----------


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Return current configuration."""
    return {
        "entity": DEFAULT_ENTITY,
        "project": DEFAULT_PROJECT,
        "wandb_base_url": WANDB_BASE_URL,
        "live_mode": _WANDB_AVAILABLE,
    }


@app.get("/api/runs")
def list_runs(
    limit: int = Query(default=50, le=200),
    state: str | None = None,
    search: str | None = None,
    entity: str | None = None,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """List runs. Tries live W&B fetch first, falls back to cache."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT

    # Try live fetch first
    live_runs = _live_fetch_runs(ent, proj, limit, state, search)
    if live_runs is not None:
        return live_runs

    # Fall back to cached data
    data_dir = _project_data_dir(ent, proj)
    data = _read_json(data_dir / "runs.json")
    if not data:
        return []
    runs = data if isinstance(data, list) else data.get("runs", [])
    if state:
        runs = [r for r in runs if r.get("state") == state]
    if search:
        s = search.lower()
        runs = [r for r in runs if s in r.get("display_name", "").lower() or s in r.get("id", "").lower()]
    return runs[:limit]


@app.get("/api/runs/{run_id}")
def get_run(
    run_id: str,
    entity: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Get run details. Tries cache first, then live fetch."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT
    data_dir = _project_data_dir(ent, proj)

    # Try cache
    data = _read_json(data_dir / "runs" / f"{run_id}.json")
    if data:
        return data

    # Try live fetch
    data = _live_fetch_run_detail(ent, proj, run_id)
    if data:
        _write_json(data_dir / "runs" / f"{run_id}.json", data)
        return data

    raise HTTPException(status_code=404, detail=f"Run {run_id} not found. Run sync.py locally or ensure W&B is accessible.")


@app.get("/api/runs/{run_id}/history")
def get_run_history(
    run_id: str,
    entity: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Get run history. Tries cache first, then live fetch."""
    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT
    data_dir = _project_data_dir(ent, proj)

    # Try cache
    data = _read_json(data_dir / "runs" / f"{run_id}_history.json")
    if data:
        return data

    # Try live fetch
    data = _live_fetch_history(ent, proj, run_id)
    if data:
        _write_json(data_dir / "runs" / f"{run_id}_history.json", data)
        return data

    raise HTTPException(status_code=404, detail=f"History for {run_id} not available.")


_traces_loading: dict[str, float] = {}  # run_id -> start timestamp


def _bg_fetch_traces(ent: str, proj: str, run_id: str) -> None:
    """Background thread to fetch traces from W&B artifacts."""
    try:
        logger.info(f"Background: fetching traces for {run_id}")
        data = _live_fetch_traces(ent, proj, run_id)
        if data:
            data_dir = _project_data_dir(ent, proj)
            _write_json(data_dir / "runs" / f"{run_id}_traces.json", data)
            logger.info(f"Background: cached {data.get('count', 0)} traces for {run_id}")
        else:
            # Write empty result so we don't keep retrying
            data_dir = _project_data_dir(ent, proj)
            _write_json(data_dir / "runs" / f"{run_id}_traces.json",
                        {"traces": [], "count": 0, "source": "no_artifacts"})
            logger.info(f"Background: no traces found for {run_id}")
    except Exception as e:
        logger.warning(f"Background trace fetch failed for {run_id}: {e}")
        # Write error result
        try:
            data_dir = _project_data_dir(ent, proj)
            _write_json(data_dir / "runs" / f"{run_id}_traces.json",
                        {"traces": [], "count": 0, "source": "error"})
        except Exception:
            pass
    finally:
        _traces_loading.pop(run_id, None)


@app.get("/api/runs/{run_id}/traces")
def get_run_traces(
    run_id: str,
    entity: str | None = None,
    project: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Get traces. Returns cache immediately, fetches live in background if missing."""
    import threading

    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT
    data_dir = _project_data_dir(ent, proj)

    # Try cache first
    data = _read_json(data_dir / "runs" / f"{run_id}_traces.json")
    if data and data.get("count", 0) > 0 and not refresh:
        return data

    # If already fetching in background, tell frontend to poll (with 120s timeout)
    import time as _time
    if run_id in _traces_loading:
        if _time.time() - _traces_loading[run_id] < 120:
            return {"traces": [], "count": 0, "source": "loading"}
        else:
            _traces_loading.pop(run_id, None)  # stale, retry

    # Start background fetch
    if _WANDB_AVAILABLE and _wandb_reachable():
        _traces_loading[run_id] = _time.time()
        t = threading.Thread(target=_bg_fetch_traces, args=(ent, proj, run_id), daemon=True)
        t.start()
        return {"traces": [], "count": 0, "source": "loading"}

    return {"traces": [], "count": 0, "source": "not_available"}


@app.get("/api/sync_status")
def sync_status() -> dict[str, Any]:
    """Check when data was last synced."""
    meta = _read_json(DATA_DIR / "sync_meta.json")
    if not meta:
        return {"synced": False, "message": "No data synced yet. Run: python sync.py", "live_mode": _WANDB_AVAILABLE}
    meta["live_mode"] = _WANDB_AVAILABLE
    return meta


@app.post("/api/trigger_sync")
def trigger_sync(
    limit: int = Query(default=50, le=200),
    detail_top: int = Query(default=15, le=50),
    entity: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Trigger a manual sync — live-fetches from W&B and caches locally."""
    from datetime import datetime, timezone

    ent = entity or DEFAULT_ENTITY
    proj = project or DEFAULT_PROJECT

    if not _WANDB_AVAILABLE or not _wandb_reachable():
        return {"status": "error", "error": "W&B is not reachable"}

    data_dir = _project_data_dir(ent, proj)
    synced_runs = 0
    synced_details = 0

    runs = _live_fetch_runs(ent, proj, limit, None, None)
    if runs:
        _write_json(data_dir / "runs.json", runs)
        synced_runs = len(runs)

        for run in runs[:detail_top]:
            rid = run["id"]
            try:
                detail = _live_fetch_run_detail(ent, proj, rid)
                if detail:
                    _write_json(data_dir / "runs" / f"{rid}.json", detail)
                    synced_details += 1
            except Exception:
                pass
            try:
                hist = _live_fetch_history(ent, proj, rid)
                if hist:
                    _write_json(data_dir / "runs" / f"{rid}_history.json", hist)
            except Exception:
                pass
            try:
                traces = _live_fetch_traces(ent, proj, rid)
                if traces:
                    _write_json(data_dir / "runs" / f"{rid}_traces.json", traces)
            except Exception:
                pass

    meta = {
        "synced": True,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "num_runs": synced_runs,
        "num_details": synced_details,
        "message": "Data synced successfully",
    }
    _write_json(data_dir / "sync_meta.json", meta)
    return {"status": "synced", "sync_meta": meta}


# ---------- AI Assistant (calls Bedrock directly — AWS is public) ----------


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
    """AI assistant powered by Bedrock Claude."""
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
            context_parts.append(f"(Run {req.run_id} not synced)")

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
- Analyze the agent's actual behavior step by step — what commands it ran, what it found, how it reasoned
- Identify inefficiencies: unnecessary retries, wrong approaches, wasted turns
- Assess whether the solution is correct and complete, not just whether reward=1
- Point out good patterns (systematic debugging, verification steps) and bad patterns (brute force, ignoring errors)
- For failed trajectories, pinpoint exactly where and why the agent went wrong

Be concise and actionable. Use specific evidence from the trajectory to support your analysis."""

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
        answer = response["output"]["message"]["content"][0]["text"]
        return AskResponse(answer=answer)
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
