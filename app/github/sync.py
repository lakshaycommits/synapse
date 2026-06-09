from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .client import GitHubClient
from .formatter import format_issue, format_pr, format_code_file, write_temp_file
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "_")


def sync_repository(
    repo: str,
    branch: str,
    qdrant,
    embeddings,
    ingest_fn: Callable,
) -> None:
    token = os.getenv("GITHUB_TOKEN", "")
    upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp"))
    counts = {"issues": 0, "prs": 0, "files": 0}
    slug = _repo_slug(repo)

    logger.info("GitHub sync started: %s (branch: %s)", repo, branch)

    try:
        client = GitHubClient(token, repo)
    except Exception:
        logger.exception("Cannot connect to GitHub repo %r", repo)
        return

    for issue in client.get_issues():
        try:
            content = format_issue(issue)
            name = f"gh_{slug}_issue_{issue.number}.md"
            path = write_temp_file(content, ".md", upload_dir)
            ingest_fn(path, qdrant, embeddings, name)
            counts["issues"] += 1
        except Exception:
            logger.exception("Failed to sync issue #%d", issue.number)

    for pr in client.get_pull_requests():
        try:
            content = format_pr(pr)
            name = f"gh_{slug}_pr_{pr.number}.md"
            path = write_temp_file(content, ".md", upload_dir)
            ingest_fn(path, qdrant, embeddings, name)
            counts["prs"] += 1
        except Exception:
            logger.exception("Failed to sync PR #%d", pr.number)

    for cf in client.get_code_files(branch):
        try:
            content = format_code_file(cf)
            safe_path = cf.path.replace("/", "_")
            name = f"gh_{slug}_{safe_path}.md"
            path = write_temp_file(content, ".md", upload_dir)
            ingest_fn(path, qdrant, embeddings, name)
            counts["files"] += 1
        except Exception:
            logger.exception("Failed to sync file %s", cf.path)

    logger.info("GitHub sync complete: %s → %s", repo, counts)
