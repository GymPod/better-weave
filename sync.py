#!/usr/bin/env python3
"""Sync W&B data to Modal volume for Better Weave.

Run locally (needs access to wandb.agi.amazon.dev):
    python sync.py                    # sync latest 30 runs
    python sync.py --limit 100        # sync more runs
    python sync.py --run-id abc123    # sync specific run detail
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import requests
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://wandb.agi.amazon.dev")
os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
ENTITY = os.environ.get("BETTER_WEAVE_ENTITY", "autonomy")
PROJECT = os.environ.get("BETTER_WEAVE_PROJECT", "slime-agent")


def get_wandb_auth() -> dict[str, str]:
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


def fetch_runs(limit: int = 30, state: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
    """Fetch run list from W&B GraphQL."""
    headers = get_wandb_auth()
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
            node {
              name
              displayName
              state
              historyLineCount
              createdAt
              heartbeatAt
              tags
            }
          }
        }
      }
    }
    """
    resp = requests.post(
        f"{WANDB_BASE_URL}/graphql",
        json={"query": query, "variables": {
            "project": PROJECT, "entity": ENTITY, "limit": limit,
            "filters": json.dumps(filters) if filters else None,
        }},
        headers=headers, timeout=15,
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


def fetch_run_detail(run_id: str) -> dict[str, Any]:
    """Fetch run config + summary via wandb SDK."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
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


def fetch_run_history(run_id: str, samples: int = 500) -> dict[str, Any]:
    """Fetch metric history via GraphQL."""
    headers = get_wandb_auth()
    query = """
    query RunHistory($project: String!, $entity: String!, $name: String!, $samples: Int!) {
      project(name: $project, entityName: $entity) {
        run(name: $name) { history(samples: $samples) }
      }
    }
    """
    resp = requests.post(
        f"{WANDB_BASE_URL}/graphql",
        json={"query": query, "variables": {
            "project": PROJECT, "entity": ENTITY, "name": run_id, "samples": samples,
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


def fetch_run_traces(run_id: str, max_versions: int = 10) -> dict[str, Any]:
    """Fetch trajectory data from W&B Table artifacts (TrajectoryDetails).

    Each training step logs a TrajectoryDetails artifact with ~32 rollout rows.
    We fetch the latest N versions and combine them.

    Args:
        run_id: W&B run ID
        max_versions: Max artifact versions to fetch (latest N)
    """
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Find trajectory artifacts
    all_arts = list(run.logged_artifacts())
    traj_arts = [a for a in all_arts if "TrajectoryDetails" in a.name]
    if not traj_arts:
        logger.info(f"  No TrajectoryDetails artifacts for {run_id}")
        return {"traces": [], "count": 0, "source": "no_artifacts"}

    # Sort by version number, take latest N
    traj_arts.sort(key=lambda a: a.version)
    selected = traj_arts[-max_versions:]
    logger.info(f"  Found {len(traj_arts)} trajectory artifacts, fetching latest {len(selected)}")

    # Table columns from gta_trajectory_table.py
    TABLE_COLUMNS = [
        "called", "step", "dataset_name", "verifier_type", "status", "remove_sample",
        "prompt", "label", "trajectory", "reward", "env_reward", "failure_reason",
        "error", "num_steps", "num_tool_calls", "response_length", "verifier_outcome",
        "grader_responses", "duration", "time_env_create", "time_load_tools",
        "time_agent_invoke", "time_verify", "time_cleanup",
    ]

    traces: list[dict] = []
    for art in selected:
        try:
            dl_path = art.download(f"/tmp/better_weave_artifacts/{run_id}/{art.version}")
            # Find the table json file
            table_files = list(Path(dl_path).glob("*.table.json"))
            if not table_files:
                continue
            with open(table_files[0]) as f:
                table = json.load(f)
            columns = table.get("columns", TABLE_COLUMNS)
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
            logger.info(f"  v{art.version}: {len(table.get('data', []))} rows ({art.name})")
        except Exception as e:
            logger.warning(f"  Failed to fetch artifact {art.name} v{art.version}: {e}")

    return {"traces": traces, "count": len(traces), "source": "artifacts"}


def sync_to_volume(data_dir: Path) -> None:
    """Upload local data_dir to Modal volume."""
    vol = modal.Volume.from_name("better-weave-data", create_if_missing=True)
    logger.info("Uploading to Modal volume...")

    files = sorted(data_dir.rglob("*.json"))
    with vol.batch_upload(force=True) as batch:
        for fpath in files:
            rel = fpath.relative_to(data_dir)
            remote_path = f"/better_weave/{rel}"
            batch.put_file(fpath, remote_path)
            logger.info(f"  queued {rel}")

    logger.info(f"Uploaded {len(files)} files to volume.")


S3_BUCKET = os.environ.get("BETTER_WEAVE_S3_BUCKET", "agi-emerge-data-us-east-1")
S3_PREFIX = os.environ.get("BETTER_WEAVE_S3_PREFIX", "better_weave")


def sync_to_s3(data_dir: Path) -> None:
    """Upload local data_dir to S3 (for Vercel/Circuit deployment)."""
    import boto3

    s3 = boto3.client("s3")
    files = sorted(data_dir.rglob("*.json"))
    logger.info(f"Uploading {len(files)} files to s3://{S3_BUCKET}/{S3_PREFIX}/...")

    for fpath in files:
        rel = fpath.relative_to(data_dir)
        key = f"{S3_PREFIX}/{rel}"
        s3.upload_file(str(fpath), S3_BUCKET, key, ExtraArgs={"ContentType": "application/json"})
        logger.info(f"  uploaded {rel}")

    logger.info(f"Uploaded {len(files)} files to S3.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync W&B data to Modal volume")
    parser.add_argument("--limit", type=int, default=30, help="Number of runs to fetch")
    parser.add_argument("--run-id", type=str, help="Sync a specific run's detail+history")
    parser.add_argument("--detail-top", type=int, default=10, help="Fetch detail for top N runs")
    parser.add_argument("--local-only", action="store_true", help="Save locally, don't upload anywhere")
    parser.add_argument("--s3", action="store_true", help="Upload to S3 (for Vercel/Circuit deployment)")
    parser.add_argument("--modal", action="store_true", help="Upload to Modal volume (default if neither --s3 nor --local-only)")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (default: from env or 'autonomy')")
    parser.add_argument("--project", type=str, default=None, help="W&B project (default: from env or 'slime-agent')")
    args = parser.parse_args()

    global ENTITY, PROJECT
    if args.entity:
        ENTITY = args.entity
    if args.project:
        PROJECT = args.project

    tmp_dir = Path("/tmp/better_weave_sync")
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        # Sync single run
        logger.info(f"Fetching detail for {args.run_id}...")
        detail = fetch_run_detail(args.run_id)
        (runs_dir / f"{args.run_id}.json").write_text(json.dumps(detail, default=str))
        logger.info(f"Fetching history for {args.run_id}...")
        hist = fetch_run_history(args.run_id)
        (runs_dir / f"{args.run_id}_history.json").write_text(json.dumps(hist, default=str))
        logger.info(f"Fetching traces for {args.run_id}...")
        traces = fetch_run_traces(args.run_id)
        (runs_dir / f"{args.run_id}_traces.json").write_text(json.dumps(traces, default=str))
    else:
        # Sync run list + top N details
        logger.info(f"Fetching {args.limit} runs...")
        runs = fetch_runs(limit=args.limit)
        (tmp_dir / "runs.json").write_text(json.dumps(runs, default=str))
        logger.info(f"Fetched {len(runs)} runs")

        for i, run in enumerate(runs[:args.detail_top]):
            rid = run["id"]
            logger.info(f"[{i+1}/{min(len(runs), args.detail_top)}] Fetching detail for {run['display_name']} ({rid})...")
            try:
                detail = fetch_run_detail(rid)
                (runs_dir / f"{rid}.json").write_text(json.dumps(detail, default=str))
            except Exception as e:
                logger.warning(f"  Failed to fetch detail: {e}")

            if run.get("history_count", 0) > 0:
                logger.info(f"  Fetching history...")
                try:
                    hist = fetch_run_history(rid)
                    (runs_dir / f"{rid}_history.json").write_text(json.dumps(hist, default=str))
                except Exception as e:
                    logger.warning(f"  Failed to fetch history: {e}")

            logger.info(f"  Fetching traces...")
            try:
                traces = fetch_run_traces(rid)
                (runs_dir / f"{rid}_traces.json").write_text(json.dumps(traces, default=str))
            except Exception as e:
                logger.warning(f"  Failed to fetch traces: {e}")

    # Write sync metadata
    meta = {
        "synced": True,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "num_runs": len(list(runs_dir.glob("*.json"))) // 3,
        "message": "Data synced successfully",
    }
    (tmp_dir / "sync_meta.json").write_text(json.dumps(meta))

    if args.local_only:
        logger.info(f"Data saved to {tmp_dir}")
    elif args.s3:
        sync_to_s3(tmp_dir)
        logger.info("Done! Data available on S3.")
    else:
        # Default: upload to Modal (use --modal explicitly or as default)
        sync_to_volume(tmp_dir)
        logger.info("Done! Data available on Modal volume.")


if __name__ == "__main__":
    main()
