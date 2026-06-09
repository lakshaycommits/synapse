from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def format_issue(issue) -> str:
    labels = ", ".join(label.name for label in issue.labels) or "none"
    lines = [
        f"# Issue #{issue.number}: {issue.title}",
        "",
        f"State: {issue.state} | Labels: {labels}",
        "",
    ]
    if issue.body:
        lines += ["## Description", issue.body, ""]
    comments = list(issue.get_comments())
    if comments:
        lines.append("## Comments")
        for c in comments:
            lines += [f"**{c.user.login}:** {c.body}", ""]
    return "\n".join(lines)


def format_pr(pr) -> str:
    lines = [
        f"# PR #{pr.number}: {pr.title}",
        "",
        f"State: {pr.state} | Branch: {pr.head.ref} → {pr.base.ref}",
        "",
    ]
    if pr.body:
        lines += ["## Description", pr.body, ""]
    review_comments = list(pr.get_review_comments())
    if review_comments:
        lines.append("## Review Comments")
        for c in review_comments:
            lines += [f"**{c.user.login} on {c.path}:{c.position}:** {c.body}", ""]
        lines.append("")
    files = list(pr.get_files())
    if files:
        lines.append("## Changed Files")
        for f in files:
            lines.append(f"- {f.filename} (+{f.additions} -{f.deletions})")
    return "\n".join(lines)


def format_code_file(content_file) -> str:
    try:
        content = content_file.decoded_content.decode("utf-8", errors="replace")
    except Exception:
        content = "[undecodable content]"
    return f"# File: {content_file.path}\n\n{content}"


def write_temp_file(content: str, suffix: str, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile(
        delete=False, suffix=suffix, dir=upload_dir, mode="w", encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)
