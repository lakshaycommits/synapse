from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Callable

from .client import GitHubClient
from .formatter import format_issue, format_pr, write_temp_file
from ..utils.logger import get_logger

logger = get_logger(__name__)


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def handle_webhook_event(
    event_type: str,
    payload: dict,
    qdrant,
    embeddings,
    ingest_fn: Callable,
) -> None:
    repo_name = payload.get("repository", {}).get("full_name", "")
    token = os.getenv("GITHUB_TOKEN", "")
    upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp"))
    slug = repo_name.replace("/", "_")

    try:
        client = GitHubClient(token, repo_name)
    except Exception:
        logger.exception("Webhook: cannot connect to repo %r", repo_name)
        return

    if event_type == "issues":
        action = payload.get("action", "")
        if action not in ("opened", "edited", "closed", "reopened"):
            return
        number = payload["issue"]["number"]
        try:
            issue = client.get_issue(number)
            content = format_issue(issue)
            name = f"gh_{slug}_issue_{number}.md"
            path = write_temp_file(content, ".md", upload_dir)
            ingest_fn(path, qdrant, embeddings, name)
            logger.info("Webhook: ingested issue #%d from %s", number, repo_name)
        except Exception:
            logger.exception("Webhook: failed to ingest issue #%d", number)

    elif event_type == "pull_request":
        action = payload.get("action", "")
        if action not in ("opened", "edited", "closed", "synchronize"):
            return
        number = payload["pull_request"]["number"]
        try:
            pr = client.get_pull_request(number)
            content = format_pr(pr)
            name = f"gh_{slug}_pr_{number}.md"
            path = write_temp_file(content, ".md", upload_dir)
            ingest_fn(path, qdrant, embeddings, name)
            logger.info("Webhook: ingested PR #%d from %s", number, repo_name)
        except Exception:
            logger.exception("Webhook: failed to ingest PR #%d", number)

    elif event_type == "push":
        ref = payload.get("ref", "")
        branch = ref.split("/")[-1] if "/" in ref else ref
        changed_paths = {
            f["filename"]
            for commit in payload.get("commits", [])
            for f in commit.get("added", []) + commit.get("modified", [])
            if isinstance(f, str)
        }
        if not changed_paths:
            return
        try:
            all_files = client.get_code_files(branch)
            for cf in all_files:
                if cf.path not in changed_paths:
                    continue
                from .formatter import format_code_file
                content = format_code_file(cf)
                safe_path = cf.path.replace("/", "_")
                name = f"gh_{slug}_{safe_path}.md"
                path = write_temp_file(content, ".md", upload_dir)
                ingest_fn(path, qdrant, embeddings, name)
                logger.info("Webhook: ingested changed file %s", cf.path)
        except Exception:
            logger.exception("Webhook: failed to process push event")
