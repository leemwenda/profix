"""Start the webhook.  Requires WEBHOOK_SECRET. Dry-run unless WEBHOOK_LIVE=1.
   export WEBHOOK_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
   python scripts/run_webhook.py --host 127.0.0.1 --port 8000
Put it behind HTTPS (reverse proxy / tunnel). Never expose plain HTTP to the internet.
"""
import argparse
import logging

import _common  # noqa: F401
import uvicorn

from profx.webhook.app import create_app
from profx.webhook.executor import DryRunExecutor, MT5Executor
from profx.webhook.settings import Settings

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1"), ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")
s = Settings.from_env()
ex = DryRunExecutor() if s.dry_run else MT5Executor(s)
logging.getLogger("profx.webhook").info('{"event":"startup","dry_run":%s}', str(s.dry_run).lower())
uvicorn.run(create_app(s, ex), host=a.host, port=a.port)
