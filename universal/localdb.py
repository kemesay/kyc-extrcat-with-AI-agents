"""
Local PostgreSQL + pgvector for the universal engine — one command, no Docker,
no Homebrew, no root:

    python -m universal.localdb setup       install, initialise, secure, start, enable pgvector
    python -m universal.localdb use URL     use a PostgreSQL you already run (checks pgvector)
    python -m universal.localdb start | stop | status
    python -m universal.localdb url         the connection URL the engine uses
    python -m universal.localdb psql        an interactive SQL shell
    python -m universal.localdb backup      compressed pg_dump into var/universal/backups/
    python -m universal.localdb autostart on|off     start with the Mac at login (launchd)

One database holds the job queue, the results, the identity-signal store and
its pgvector index — the same engine on this laptop as on a bank's server,
so moving to production is a connection-string change.

  binaries   PostgreSQL + pgvector from conda-forge in their own conda env
             (UNIVERSAL_PG_CONDA_ENV, default kyc-pgvector), kept apart from
             the application env
  data       var/universal/pgdata (0700), inside the repo's ignored var/
  network    127.0.0.1 only, port UNIVERSAL_PG_PORT (default 5433, clear of a
             system PostgreSQL on 5432); no Unix socket
  auth       scram-sha-256; a random password generated at setup, stored
             only in var/universal/database.env (0600)

The engine reads database.env automatically (universal.jobqueue.database_url)
and starts the server if it isn't running, so after `setup` nothing else
needs configuring.
"""

from __future__ import annotations

import argparse
import gzip
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ENV_NAME = os.environ.get("UNIVERSAL_PG_CONDA_ENV", "kyc-pgvector")
PORT = int(os.environ.get("UNIVERSAL_PG_PORT", "5433"))
USER = "kyc"
DATABASE = "kyc"
LAUNCHD_LABEL = "com.kyc.universal.pgvector"


def _data_root() -> Path:
    from .jobqueue import data_dir
    return data_dir()


def pgdata() -> Path:
    return _data_root() / "pgdata"


def conf_file() -> Path:
    return _data_root() / "database.env"


def _conda() -> Optional[str]:
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda"),
                  str(Path.home() / "miniconda3/bin/conda"), str(Path.home() / "anaconda3/bin/conda"),
                  str(Path.home() / "miniforge3/bin/conda")]
    return next((c for c in candidates if c and Path(c).exists()), None)


def bin_dir() -> Optional[Path]:
    """Where pg_ctl lives: UNIVERSAL_PG_BIN, the conda env, or PATH."""
    explicit = os.environ.get("UNIVERSAL_PG_BIN")
    if explicit and (Path(explicit) / "pg_ctl").exists():
        return Path(explicit)
    conda = _conda()
    if conda:
        env_bin = Path(conda).resolve().parent.parent / "envs" / ENV_NAME / "bin"
        if (env_bin / "pg_ctl").exists():
            return env_bin
    on_path = shutil.which("pg_ctl")
    return Path(on_path).parent if on_path else None


def _run(args: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, **kw)


def managed_url() -> str:
    """The URL written by `setup`, or '' when there is no managed database."""
    path = conf_file()
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith("UNIVERSAL_DATABASE_URL="):
            return line.split("=", 1)[1].strip()
    return ""


def is_running() -> bool:
    b = bin_dir()
    if b is None or not pgdata().exists():
        return False
    return _run([b / "pg_ctl", "-D", pgdata(), "status"]).returncode == 0


def start(quiet: bool = False) -> None:
    b = _require_bin()
    if is_running():
        if not quiet:
            print(f"PostgreSQL already running on 127.0.0.1:{PORT}")
        return
    log = _data_root() / "postgres.log"
    r = _run([b / "pg_ctl", "-D", pgdata(), "-l", log, "-w", "-t", "60", "start"])
    if r.returncode != 0:
        raise RuntimeError(f"PostgreSQL did not start (see {log}):\n{r.stdout}{r.stderr}")
    if not quiet:
        print(f"PostgreSQL started on 127.0.0.1:{PORT}")


def stop() -> None:
    b = _require_bin()
    if not is_running():
        print("PostgreSQL is not running")
        return
    _run([b / "pg_ctl", "-D", pgdata(), "-m", "fast", "-w", "stop"])
    print("PostgreSQL stopped")


def _is_local_instance(url: str) -> bool:
    from urllib.parse import urlparse
    u = urlparse(url)
    return u.hostname in ("127.0.0.1", "localhost") and (u.port or 5432) == PORT


def ensure_running() -> None:
    """Called by the engine before connecting: starts the LOCAL instance if it
    is down. A database.env pointing at an existing server is left alone."""
    if _is_local_instance(managed_url()) and pgdata().exists() and not is_running():
        start(quiet=True)


def _require_bin() -> Path:
    b = bin_dir()
    if b is None:
        raise RuntimeError("PostgreSQL binaries not found — run: python -m universal.localdb setup")
    return b


def install() -> Path:
    conda = _conda()
    if conda is None:
        raise RuntimeError("conda not found. Install Miniconda, or set UNIVERSAL_PG_BIN to a PostgreSQL bin/ "
                           "directory that has the pgvector extension.")
    print(f"installing PostgreSQL + pgvector into conda env '{ENV_NAME}' (conda-forge)...")
    r = subprocess.run([conda, "create", "-y", "-q", "-n", ENV_NAME, "-c", "conda-forge", "postgresql", "pgvector"])
    if r.returncode != 0 or bin_dir() is None:
        raise RuntimeError("conda could not install postgresql + pgvector")
    return bin_dir()


def _port_free(port: int) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def setup() -> str:
    b = bin_dir() or install()
    data = pgdata()
    if not data.exists():
        if not _port_free(PORT):
            raise RuntimeError(f"port {PORT} is in use — set UNIVERSAL_PG_PORT to a free port and re-run")
        password = secrets.token_urlsafe(24)
        pwfile = _data_root() / ".pgpass.tmp"
        fd = os.open(pwfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(password)
        try:
            r = _run([b / "initdb", "-D", data, "-U", USER, "--auth=scram-sha-256", f"--pwfile={pwfile}",
                      "-E", "UTF8", "--locale=C"])
        finally:
            pwfile.unlink(missing_ok=True)
        if r.returncode != 0:
            raise RuntimeError(f"initdb failed:\n{r.stdout}{r.stderr}")
        with open(data / "postgresql.conf", "a") as f:
            f.write("\n# universal engine (universal/localdb.py)\n"
                    "listen_addresses = '127.0.0.1'\n"
                    f"port = {PORT}\n"
                    "unix_socket_directories = ''\n"
                    "password_encryption = 'scram-sha-256'\n"
                    "max_connections = 100\n"
                    "shared_buffers = 256MB\n")
        url = f"postgresql://{USER}:{password}@127.0.0.1:{PORT}/{DATABASE}"
        _write_conf(url, "Written by `python -m universal.localdb setup`")
        print(f"initialised {data}")
    start()
    url = managed_url()
    import psycopg
    admin = url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,)).fetchone():
            conn.execute(f'CREATE DATABASE "{DATABASE}"')
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        server = conn.execute("SHOW server_version").fetchone()[0]
    # Create the engine's tables now, so the first request doesn't.
    from .jobqueue import PostgresJobQueue
    from .signals import PostgresSignalStore
    PostgresJobQueue(url)
    PostgresSignalStore(url)
    print(f"ready: PostgreSQL {server} + pgvector {version} on 127.0.0.1:{PORT}, database '{DATABASE}'")
    print(f"the engine picks it up automatically from {conf_file()}")
    return url


def _write_conf(url: str, note: str) -> None:
    fd = os.open(conf_file(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"# {note} — contains the database password.\nUNIVERSAL_DATABASE_URL={url}\n")


def use(url: str) -> None:
    """Point the engine at an existing PostgreSQL: verify it, enable pgvector,
    create the engine's tables, and record the URL in database.env."""
    import psycopg
    from urllib.parse import urlparse
    if not url.startswith(("postgres://", "postgresql://")):
        raise RuntimeError("expected a postgresql://user:password@host:port/database URL")
    try:
        conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
    except Exception as e:
        raise RuntimeError(f"cannot connect: {e}")
    with conn:
        server = conn.execute("SHOW server_version").fetchone()[0]
        available = conn.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'").fetchone()
        if not available:
            raise RuntimeError(f"PostgreSQL {server} is reachable but the pgvector extension is not installed on "
                               "that server. Ask its administrator to install pgvector (package postgresql-XX-pgvector, "
                               "or build from github.com/pgvector/pgvector), then re-run.")
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as e:
            raise RuntimeError(f"pgvector is available but could not be enabled ({e}). A database owner or "
                               "superuser must run once: CREATE EXTENSION vector;")
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        ssl = conn.pgconn.ssl_in_use
    from .jobqueue import PostgresJobQueue
    from .signals import PostgresSignalStore
    PostgresJobQueue(url)
    PostgresSignalStore(url)
    _write_conf(url, "Written by `python -m universal.localdb use`")
    host = urlparse(url).hostname
    print(f"ready: PostgreSQL {server} + pgvector {version} at {host}; engine tables created")
    if not ssl and host not in ("127.0.0.1", "localhost"):
        print("⚠️  the connection is NOT encrypted — add ?sslmode=require (or verify-full) to the URL "
              "for a server on another machine")
    print(f"the engine picks it up automatically from {conf_file()}")


def status() -> dict:
    url = managed_url()
    local = _is_local_instance(url) if url else True
    info = {"database": ("local instance" if local else "existing server"), "configured": bool(url)}
    if local:
        info.update(binaries=str(bin_dir() or "not installed"), data=str(pgdata()), port=PORT, running=is_running())
    else:
        from urllib.parse import urlparse
        info["host"] = urlparse(url).hostname
    if url and (not local or info["running"]):
        import psycopg
        with psycopg.connect(managed_url()) as conn:
            info["server"] = conn.execute("SHOW server_version").fetchone()[0]
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            info["pgvector"] = row[0] if row else "missing"
            for table in ("universal_jobs", "universal_signals"):
                exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]
                info[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] if exists else 0
            info["size"] = conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
    return info


def backup() -> Path:
    b = _require_bin()
    ensure_running()
    out_dir = _data_root() / "backups"
    out_dir.mkdir(exist_ok=True)
    os.chmod(out_dir, 0o700)
    out = out_dir / f"kyc-{time.strftime('%Y%m%d-%H%M%S')}.sql.gz"
    dump = subprocess.run([str(b / "pg_dump"), "--no-owner", managed_url()], capture_output=True)
    if dump.returncode != 0:
        raise RuntimeError(dump.stderr.decode()[:500])
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(gzip.compress(dump.stdout))
    print(f"backup written: {out} (contains personal data — store it encrypted)")
    return out


def autostart(enable: bool) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("autostart uses launchd (macOS); on Linux use a systemd unit running pg_ctl start")
    plist = Path.home() / "Library/LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if not enable:
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        plist.unlink(missing_ok=True)
        print("autostart off")
        return
    b = _require_bin()
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>{b / 'pg_ctl'}</string><string>-D</string><string>{pgdata()}</string>
    <string>-l</string><string>{_data_root() / 'postgres.log'}</string><string>start</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
""")
    subprocess.run(["launchctl", "load", str(plist)], capture_output=True)
    print(f"autostart on ({plist})")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m universal.localdb", description="Local PostgreSQL + pgvector")
    ap.add_argument("command", choices=["setup", "use", "start", "stop", "status", "url", "psql", "backup", "autostart"])
    ap.add_argument("value", nargs="?", help="URL for use, on|off for autostart")
    args = ap.parse_args()
    try:
        if args.command == "setup":
            setup()
        elif args.command == "use":
            if not args.value:
                ap.error("use takes a postgresql://user:password@host:port/database URL")
            use(args.value)
        elif args.command == "start":
            start()
        elif args.command == "stop":
            stop()
        elif args.command == "status":
            for k, v in status().items():
                print(f"{k:18} {v}")
        elif args.command == "url":
            print(managed_url() or "no managed database — run: python -m universal.localdb setup")
        elif args.command == "psql":
            ensure_running()
            os.execv(str(_require_bin() / "psql"), ["psql", managed_url()])
        elif args.command == "backup":
            backup()
        elif args.command == "autostart":
            if args.value not in ("on", "off"):
                ap.error("autostart takes on or off")
            autostart(args.value == "on")
    except RuntimeError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
