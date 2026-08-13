"""Run the local AIBOX-eye coordinator without installing system packages."""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

from .config import load_config
from .orchestrator import EyeOrchestrator
from .server import make_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIBOX-eye local coordinator")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config" / "production.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    app_root = Path(__file__).parents[1]
    app = EyeOrchestrator(config, app_root)
    server = make_server(app, str(config["api"]["host"]), int(config["api"]["port"]))

    def shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    app.start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
