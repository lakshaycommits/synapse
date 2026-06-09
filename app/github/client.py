from __future__ import annotations

import os
from github import Github, GithubException
from ..utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".yaml", ".yml"}
MAX_FILE_BYTES = 500_000


class GitHubClient:
    def __init__(self, token: str, repo_name: str):
        self._gh = Github(token) if token else Github()
        self._repo = self._gh.get_repo(repo_name)
        logger.info("Connected to GitHub repo: %s", repo_name)

    def get_issues(self) -> list:
        try:
            return list(self._repo.get_issues(state="all"))
        except GithubException as e:
            logger.error("Failed to fetch issues: %s", e)
            return []

    def get_pull_requests(self) -> list:
        try:
            return list(self._repo.get_pulls(state="all"))
        except GithubException as e:
            logger.error("Failed to fetch PRs: %s", e)
            return []

    def get_code_files(self, branch: str = "main") -> list:
        results: list = []
        try:
            self._collect_files(
                self._repo.get_contents("", ref=branch), branch, results
            )
        except GithubException as e:
            logger.error("Failed to fetch code files: %s", e)
        return results

    def get_issue(self, number: int):
        return self._repo.get_issue(number)

    def get_pull_request(self, number: int):
        return self._repo.get_pull(number)

    def _collect_files(self, contents, branch: str, results: list) -> None:
        for item in contents:
            if item.type == "dir":
                try:
                    self._collect_files(
                        self._repo.get_contents(item.path, ref=branch),
                        branch,
                        results,
                    )
                except GithubException:
                    pass
            elif (
                any(item.name.endswith(ext) for ext in ALLOWED_EXTENSIONS)
                and item.size <= MAX_FILE_BYTES
            ):
                results.append(item)
