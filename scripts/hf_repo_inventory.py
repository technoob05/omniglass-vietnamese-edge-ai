#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_ids", nargs="+")
    parser.add_argument("--repo-type", default="model")
    args = parser.parse_args()
    api = HfApi()
    reports = []
    for repo_id in args.repo_ids:
        info = api.model_info(repo_id, files_metadata=True)
        files = []
        for sibling in info.siblings or []:
            size = int(sibling.size or 0)
            files.append({"path": sibling.rfilename, "size": size})
        files.sort(key=lambda row: row["size"], reverse=True)
        reports.append(
            {
                "repo_id": repo_id,
                "sha": info.sha,
                "private": info.private,
                "gated": info.gated,
                "total_bytes": sum(row["size"] for row in files),
                "files": files,
            }
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
