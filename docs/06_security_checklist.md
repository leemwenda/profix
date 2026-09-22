# 6. Security checklist

- [ ] `WEBHOOK_SECRET` is a random string ≥ 32 chars (`python -c "import secrets;print(secrets.token_urlsafe(32))"`),
      stored in an environment variable / secrets manager — **never** committed to source control,
      never in `config/strategy.toml`, never in the Pine script when shared publicly.
- [ ] The webhook binds to `127.0.0.1` and is only reachable via a **reverse proxy terminating
      TLS** (or an SSH/VPN tunnel) — never expose plain HTTP to the internet.
- [ ] `WEBHOOK_LIVE` is unset (dry-run) until the demo checklist in `08_demo_testing_procedure.md`
      is complete.
- [ ] `WEBHOOK_ALLOW_REAL` stays unset until you have deliberately decided to go live; the
      `MT5Executor` refuses to trade a non-demo account otherwise.
- [ ] MT5 EA: `InpMode=TRADE` **and** `InpConfirmLive=true` are both required to send real orders;
      either alone leaves the EA in dry-run (`ProFX.mq5::OnInit`).
- [ ] MT5 terminal: "Allow algorithmic trading" only enabled when you intend it to trade;
      "Allow WebRequest" list should NOT include the webhook unless you specifically need MT5 to
      call out (this project's default flow is MT5-as-EA, not MT5-calling-webhooks).
- [ ] The webhook never logs or echoes the secret/passphrase (verified by
      `test_secret_never_echoed_or_logged`); rotate `WEBHOOK_SECRET` if you suspect exposure.
- [ ] Duplicate/replay protection is backed by SQLite so it survives a service restart
      (`test_replay_survives_restart`); back up or monitor `webhook_state.sqlite3` if you need an
      audit trail longer than `retention_days`.
- [ ] Kill switch: creating the file named by `WEBHOOK_KILL_FILE` (default `PROFX_KILL.flag`)
      immediately rejects all new signals (`503`) without restarting the process.
- [ ] Rate limiting (`max_signals_per_minute`) and body-size limits (`max_body_bytes`) are on by
      default; don't disable them.
- [ ] `.gitignore` should include `*.sqlite3`, `.env`, `PROFX_KILL.flag`, and any exported MT5
      history/credentials.
- [ ] Least privilege: run the webhook and MT5 terminal under an account that can't touch
      unrelated systems; do not run the webhook as root/Administrator.
