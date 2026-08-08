"""Platform commands: `wmo login`, `wmo logout`, `wmo status`, `wmo push`, `wmo pull`.

`login` connects this machine to a platform account (browser flow by default,
`--token` for headless); `push`/`pull` round-trip world models and endpoint artifacts
against the platform registry, auto-detecting the artifact kind from what
exists locally (or remotely, for pulls).
"""

from __future__ import annotations

import json
import socket
import tempfile
import webbrowser
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

import wmo.cli.platform_cmds as _self
from wmo.common.config.store import WorldModelStore
from wmo.runtime.platform.credentials import (
    DEFAULT_WEB_URL,
    PlatformCredentials,
    clear_credentials,
    credentials_path,
    load_credentials,
    save_credentials,
)

if TYPE_CHECKING:
    from wmo.runtime.platform.client import PlatformClient, PlatformError, WhoAmI

# Kept here rather than importing wmo.cli.runs_app: platform commands are registered by the
# root CLI, and loading their run-history implementation would defeat the light-command boundary.
MANIFEST_RELPATH = Path("optimize") / "optimize-run.json"

_console = Console()
_CHECK = "[green]✓[/green]"

# Module-level singletons: typer.Option calls can't be defaults inline (ruff B008).
# Annotated-style options carry no default here; the parameter's own `=` does.
_LOGIN_URL = typer.Option(
    "--url", help="Platform URL (defaults to the saved one, then the hosted platform)."
)
_LOGIN_API_URL = typer.Option(
    "--api-url",
    help="Platform API URL (skips web discovery; useful for protected previews).",
)
_LOGIN_TOKEN = typer.Option(
    "--token", help="Paste an existing API key instead of using the browser."
)
_LOGIN_NO_BROWSER = typer.Option(
    "--no-browser", help="Print the authorization URL instead of opening a browser."
)
_ORG = typer.Option("--org", help="Organization id (defaults to the login's default organization).")
_PUSH_AS = typer.Option(
    "--as", help="Remote name to publish under (local names may not be slug-safe)."
)
_PULL_FORCE = typer.Option("--force", help="Replace an existing local artifact.")
_PUSH_SERVE_MODEL = typer.Option(
    "--serve-model",
    help="Day-one serving model for a newly created endpoint (a platform default "
    "serves when omitted; deployments without that model's credentials refuse "
    "and name the serveable set).",
)
_ROOT = typer.Option("--root", help="Artifact root directory.")


def login(
    url: Annotated[str | None, _LOGIN_URL] = None,
    api_url: Annotated[str | None, _LOGIN_API_URL] = None,
    token: Annotated[str | None, _LOGIN_TOKEN] = None,
    no_browser: Annotated[bool, _LOGIN_NO_BROWSER] = False,
) -> None:
    """Connect this machine to a platform account."""
    from wmo.runtime.platform.client import (
        PlatformError,
        PlatformUnreachable,
    )

    credentials = load_credentials()
    web_url: str | None

    if api_url is None:
        web_url = (url or credentials.web_url or DEFAULT_WEB_URL).rstrip("/")
        try:
            fetch_cfg = cast(Any, _self.fetch_cli_config)
            api_url = fetch_cfg(web_url)
        except PlatformUnreachable as error:
            raise typer.BadParameter(str(error)) from error
        except PlatformError as error:
            raise typer.BadParameter(f"{web_url} does not look like a platform: {error}") from error
        if api_url is None:
            raise typer.BadParameter(f"{web_url} did not advertise a backend URL; is it deployed?")
    else:
        api_url = api_url.rstrip("/")
        # --api-url names a backend directly; only --url can say which web app
        # fronts it. Recording the hosted default here would misreport every
        # later "connected to ..." line, so leave it unset — the API URL is
        # what gets displayed then.
        web_url = url.rstrip("/") if url else None

    if token is None:
        token = _browser_login(web_url or DEFAULT_WEB_URL, open_browser=not no_browser)
    if token is None or not token.strip():
        _console.print("[red]No key received; nothing saved.[/red]")
        raise typer.Exit(code=1)
    token = token.strip()

    with PlatformClient(api_url, token) as client:
        try:
            identity = client.whoami()
        except PlatformUnreachable as error:
            raise _platform_failure(error, "Connection failed") from error
        except PlatformError as error:
            # Only an auth status means the key itself is bad; a login wall or a
            # deploy that is not a platform must not be blamed on the key.
            rejected = error.status_code in (401, 403)
            headline = "The key was rejected" if rejected else "Login failed"
            raise _platform_failure(error, headline) from error

    # A relogin may land on a different account: keep the saved default
    # organization only if the new identity can still see it.
    visible_orgs = {org.id for org in identity.orgs}
    default_org = credentials.default_org if credentials.default_org in visible_orgs else None
    if default_org is None and len(identity.orgs) == 1:
        default_org = identity.orgs[0].id
    updated = credentials.model_copy(
        update={
            "web_url": web_url,
            "api_url": api_url,
            "token": token,
            "default_org": default_org,
        }
    )
    path = save_credentials(updated)
    org_names = ", ".join(org.name for org in identity.orgs) or "no organizations"
    _console.print(f"{_CHECK} Connected to [bold]{org_names}[/bold] ({path})")
    _print_orgs(identity, updated.default_org)


def logout() -> None:
    """Disconnect: delete the saved credential."""
    credentials = load_credentials()
    removed = clear_credentials()
    if not removed:
        _console.print("Not logged in; nothing to remove.")
        return
    _console.print(f"{_CHECK} Logged out.")
    if credentials.token:
        _console.print("The key itself stays valid until revoked on the platform's API keys page.")


def status() -> None:
    """Show the platform connection: account and organizations."""
    from wmo.runtime.platform.client import PlatformError

    credentials = load_credentials()
    if not credentials.is_complete():
        _console.print(
            f"Not connected (no credential at {credentials_path()}). Run [bold]wmo login[/bold]."
        )
        raise typer.Exit(code=1)
    with _client(credentials) as client:
        try:
            identity = client.whoami()
        except PlatformError as error:
            raise _platform_failure(error, "Connection check failed") from error
    # Env-var credentials (the headless path) carry no web_url; show the host
    # requests actually went to rather than "None".
    home = credentials.web_url or credentials.api_url
    _console.print(f"{_CHECK} Connected to [bold]{home}[/bold]")
    _console.print(f"  acting as: {identity.actor.kind} {identity.actor.id}")
    _print_orgs(identity, credentials.default_org)


def push(
    name: Annotated[str, typer.Argument(help="Local world model name.")],
    org: Annotated[str | None, _ORG] = None,
    push_as: Annotated[str | None, _PUSH_AS] = None,
    serve_model: Annotated[str | None, _PUSH_SERVE_MODEL] = None,
    root: Annotated[str, _ROOT] = ".wmo",
) -> None:
    """Publish a local world model to the platform registry.

    A model push carries everything the model directory holds: the simulation
    (the built bundle), the model (a measured policy.json + report.json become
    the endpoint's installed artifacts), and the pipeline (the optimize run's
    manifest replays into the platform's run history). Legs whose artifacts are
    absent are skipped with a note, so a bundle-only directory pushes exactly
    as before.
    """
    model_dir = WorldModelStore(root).dir_for(name)
    if model_dir is None:
        raise typer.BadParameter(_nothing_local(name, root))
    remote_name = push_as or name

    credentials, org_id = _require_connection(org)
    with _connected(credentials, "Push failed") as client:
        pushed_id = _push_model(client, org_id, remote_name, model_dir)
        _push_endpoint_artifacts(
            client,
            org_id,
            remote_name,
            model_dir,
            world_model_id=pushed_id,
            serve_model=serve_model,
        )
        _push_pipeline(client, org_id, remote_name, model_dir)


def pull(
    name: Annotated[str, typer.Argument(help="Remote world model or endpoint name.")],
    org: Annotated[str | None, _ORG] = None,
    force: Annotated[bool, _PULL_FORCE] = False,
    root: Annotated[str, _ROOT] = ".wmo",
) -> None:
    """Fetch the CURRENT platform state of a model or endpoint.

    A model pull restores the bundle as pushed, then overwrites policy.json,
    report.json, and the knn bank with what the same-named endpoint serves NOW,
    so a hosted optimizer's fit comes home with the pull. A name that is only
    an endpoint (platform-created, never pushed) pulls those artifacts into the
    model directory by themselves. The artifact overwrite is the point of the
    command and does not need --force; --force still governs replacing an
    existing local bundle.
    """
    credentials, org_id = _require_connection(org)
    with _connected(credentials, "Pull failed") as client:
        resolved_kind = _detect_pullable_kind(client, org_id, name)
        if resolved_kind == "model":
            _pull_model(client, org_id, name, root, force=force)
            _pull_endpoint_artifacts(client, org_id, name, WorldModelStore(root).model_dir(name))
        elif resolved_kind == "endpoint":
            dest_dir = WorldModelStore(root).model_dir(name)
            if not _pull_endpoint_artifacts(client, org_id, name, dest_dir):
                raise typer.BadParameter(
                    f"the organization has no world model or endpoint named {name!r}"
                )
            _console.print(
                "no simulation to pull: the endpoint's evidence came from a real benchmark "
                "or a hosted fit, so only its measured artifacts live locally"
            )


# -- helpers -------------------------------------------------------------------------------------


def _browser_login(web_url: str, *, open_browser: bool) -> str | None:
    """Run the loopback browser flow; fall back to a hidden paste prompt."""
    from wmo.runtime.platform.auth import BrowserLogin

    login_attempt = BrowserLogin(web_url)
    try:
        login_attempt.start()
        key_name = f"wmo on {socket.gethostname()}"
        authorize_url = login_attempt.authorize_url(key_name=key_name)
        _console.print(f"Approve the request in your browser:\n  [bold]{authorize_url}[/bold]")
        if open_browser:
            webbrowser.open(authorize_url)
        token = login_attempt.wait()
    finally:
        login_attempt.close()
    if token is None:
        _console.print("Timed out waiting for the browser.")
        return typer.prompt("Paste an API key instead", hide_input=True, default="") or None
    return token


def _client(credentials: PlatformCredentials) -> PlatformClient:
    if credentials.api_url is None or credentials.token is None:
        raise typer.BadParameter("not connected to a platform; run `wmo login` first")
    client_cls = cast(Any, _self.PlatformClient)
    return client_cls(credentials.api_url, credentials.token)


@contextmanager
def _connected(credentials: PlatformCredentials, headline: str) -> Iterator[PlatformClient]:
    """Open a client whose request failures end the command without a traceback.

    `PlatformClient` reports every failure as a `PlatformError` (unreachable
    hosts included), so this one handler covers every request the body makes.
    """
    from wmo.runtime.platform.client import PlatformError

    with _client(credentials) as client:
        try:
            yield client
        except PlatformError as error:
            raise _platform_failure(error, headline) from error


def _platform_failure(error: PlatformError, headline: str) -> typer.Exit:
    """Render a failed platform request as a clean error; the message carries the next step."""
    _console.print(f"[red]{headline}:[/red] {error}")
    return typer.Exit(code=1)


def _require_connection(org: str | None) -> tuple[PlatformCredentials, str]:
    credentials = load_credentials()
    if not credentials.is_complete():
        raise typer.BadParameter("not connected to a platform; run `wmo login` first")
    org_id = org or credentials.default_org
    if not org_id:
        raise typer.BadParameter(
            "no organization selected; pass --org <id> (see `wmo status` for your organizations)"
        )
    return credentials, org_id


def _nothing_local(name: str, root: str) -> str:
    """Say what was looked for, where, what is actually there, and what to run next."""
    have = WorldModelStore(root).list_names()
    found = f"have: {', '.join(have)}" if have else "nothing is built there"
    return (
        f"no local world model named {name!r} under {root} ({found}); "
        f"`wmo build --name {name}` builds one, or pass --root <dir>"
    )


def _detect_pullable_kind(client: PlatformClient, org_id: str, name: str) -> str:
    """What a pull by this name reaches: a model, or an endpoint with no simulation.

    The endpoint probe runs only after the model read comes up empty. An endpoint
    with no same-named simulation (platform-created, or its evidence is a real
    benchmark) is still pullable: its measured artifacts are the whole point of
    fetching current state.
    """
    model_names = {model.name for model in client.list_world_models(org_id)}
    if name in model_names:
        return "model"
    if client.get_endpoint(org_id, name) is not None:
        return "endpoint"
    raise typer.BadParameter(f"the organization has no world model or endpoint named {name!r}")


def _push_model(client: PlatformClient, org_id: str, remote_name: str, model_dir: Path) -> str:
    from wmo.runtime.platform.client import PlatformError
    from wmo.runtime.platform.transfer import extract_push_meta, pack_model_dir

    meta = extract_push_meta(model_dir)
    with tempfile.TemporaryDirectory(prefix="wmo-push-") as staging:
        bundle = pack_model_dir(model_dir, Path(staging) / f"{remote_name}.tar.gz")
        try:
            pushed = client.push_model_bundle(
                org_id,
                remote_name,
                bundle.path,
                bundle.sha256,
                bundle.byte_size,
                meta,
            )
        except PlatformError as error:
            if error.status_code == 422 and "name" in str(error):
                raise typer.BadParameter(
                    f"{error} — publish under a slug-safe name with --as"
                ) from error
            raise
    _console.print(
        f"{_CHECK} Pushed world model [bold]{pushed.name}[/bold] "
        f"({bundle.byte_size:,} bytes, sha256 {bundle.sha256[:12]}…)"
    )
    return pushed.id


def _push_endpoint_artifacts(
    client: PlatformClient,
    org_id: str,
    remote_name: str,
    model_dir: Path,
    *,
    world_model_id: str,
    serve_model: str | None,
) -> None:
    """Publish the model directory's measured policy and report as the endpoint.

    Creates the endpoint when the org has none by this name (linked to the
    just-pushed simulation), then installs policy.json + report.json on it. A
    knn policy carries an evidence bank and goes through the multipart
    installer; static and rank ride the JSON artifacts route. Skipped with a
    note when either artifact is absent: a bundle-only push is still a push.
    """
    policy_path = model_dir / "policy.json"
    report_path = model_dir / "report.json"
    if not policy_path.is_file():
        _console.print("no policy.json in the model directory; skipping the endpoint leg")
        return
    if not report_path.is_file():
        _console.print(
            "policy.json has no report.json beside it; skipping the endpoint leg "
            "(the report is the endpoint's customer-facing evidence)"
        )
        return
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise typer.BadParameter(
            f"the model directory's policy.json/report.json is not valid JSON ({error}); "
            "refit or regenerate the artifact before pushing"
        ) from error
    existing = client.get_endpoint(org_id, remote_name)
    if existing is None:
        client.create_endpoint(
            org_id, remote_name, world_model_id=world_model_id, model=serve_model
        )
        _console.print(f"{_CHECK} Created endpoint [bold]{remote_name}[/bold]")
    elif existing.get("world_model_id") != world_model_id:
        # The platform sets the simulation link at create time and exposes no
        # update surface for it; installing artifacts below is still correct
        # (the policy is self-contained), but the link must not silently lie.
        _console.print(
            f"[yellow]endpoint {remote_name} is linked to a different simulation "
            f"(or none); the link is set when an endpoint is created and was left "
            "untouched[/yellow]"
        )
    if isinstance(policy, dict) and policy.get("kind") == "knn":
        bank_path = Path(f"{policy_path}.bank.npz")
        client.install_endpoint_policy(
            org_id,
            remote_name,
            policy_path,
            bank_path if bank_path.is_file() else None,
            report_path,
        )
    else:
        client.install_endpoint_artifacts(org_id, remote_name, policy=policy, report=report)
    _console.print(
        f"{_CHECK} Installed measured policy + report on endpoint [bold]{remote_name}[/bold]"
    )


def _push_pipeline(client: PlatformClient, org_id: str, remote_name: str, model_dir: Path) -> None:
    """Replay the model's optimize pipeline into the platform's run history.

    The same derivation `wmo runs backfill` performs, riding the push's own
    client: events come from the manifest's recorded clocks, seqs are
    deterministic, and the platform discards replays, so re-pushing a model is
    free. A run that already reported itself live is left alone rather than
    double-counted.
    """
    from wmo.optimize.telemetry.backfill import (
        BackfillRefused,
        ensure_backfillable,
    )
    from wmo.runtime.runs.client import PushRejected, PushUnavailable, default_emitter_id
    from wmo.runtime.runs.schema import pipeline_external_id

    manifest = model_dir / MANIFEST_RELPATH
    if not manifest.is_file():
        _console.print("no optimize manifest in the model directory; skipping the pipeline leg")
        return
    external_id = pipeline_external_id(remote_name)
    opt_events = cast(Any, _self.optimize_events)
    reader_cls = cast(Any, _self.RunsReader)
    sink_cls = cast(Any, _self.RunsSink)
    events = opt_events(manifest, model=remote_name, external_id=external_id)
    recorded = reader_cls(client, org_id).event_count(external_id)
    try:
        ensure_backfillable(recorded)
    except BackfillRefused:
        # `recorded` cannot distinguish a completed earlier push from one that
        # died between batches, so the skip names the recovery command instead
        # of calling the run complete.
        _console.print(
            f"pipeline run [bold]{external_id}[/bold] already has {recorded} recorded "
            f"event(s); skipping the replay. If an earlier push was interrupted, "
            f"`wmo runs backfill {model_dir} --name {external_id} --force` completes it."
        )
        return
    sink = sink_cls(client, org_id=org_id, emitter_id=default_emitter_id())
    try:
        ack = sink.push(external_id, events)
    except (PushRejected, PushUnavailable) as error:
        raise _platform_failure(PlatformError(str(error)), "Pipeline push failed") from error
    _console.print(
        f"{_CHECK} Pushed pipeline run [bold]{external_id}[/bold] "
        f"({ack.accepted} of {len(events)} events newly accepted)"
    )


def _pull_model(client: PlatformClient, org_id: str, name: str, root: str, *, force: bool) -> None:
    from wmo.runtime.platform.transfer import unpack_model_bundle

    dest_dir = WorldModelStore(root).model_dir(name)
    with tempfile.TemporaryDirectory(prefix="wmo-pull-") as staging:
        bundle_path = Path(staging) / f"{name}.tar.gz"
        client.download_model_bundle(org_id, name, bundle_path)
        try:
            unpack_model_bundle(bundle_path, dest_dir, force=force)
        except FileExistsError as error:
            raise typer.BadParameter(str(error)) from error
    _console.print(f"{_CHECK} Pulled world model [bold]{name}[/bold] into {dest_dir}")


def _pull_endpoint_artifacts(
    client: PlatformClient, org_id: str, name: str, dest_dir: Path
) -> bool:
    """Overwrite the local dir's measured artifacts with the endpoint's CURRENT state.

    The pull half of D-LOCAL-PUSH: policy.json, report.json, and the knn bank
    reflect what the endpoint serves NOW (a hosted optimizer's fit included),
    not what was last pushed. Overwriting is the point, so no --force gate.

    Returns:
        Whether the org had an endpoint by this name at all.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = client.download_endpoint_policy(
        org_id, name, bank_dest=dest_dir / "policy.json.bank.npz"
    )
    if payload is None:
        _console.print(f"no endpoint named {name!r}; the bundle is all there is to pull")
        return False
    (dest_dir / "policy.json").write_text(
        json.dumps(payload["policy"], indent=2) + "\n", encoding="utf-8"
    )
    wrote = ["policy.json"]
    report = payload.get("report")
    report_path = dest_dir / "report.json"
    if report is not None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        wrote.append("report.json")
    elif report_path.is_file():
        # Same rule as the bank below: a reportless current endpoint must not
        # keep an as-pushed or earlier-pull report reading as current evidence.
        report_path.unlink()
        _console.print("removed a stale report.json from an earlier state")
    bank_path = dest_dir / "policy.json.bank.npz"
    if isinstance(payload.get("bank"), dict):
        wrote.append("policy.json.bank.npz")
    elif bank_path.is_file():
        # A bankless current policy must not leave an older fit's sidecar
        # beside it; the pair would disagree the next time anything reads them.
        bank_path.unlink()
        _console.print("removed a stale policy.json.bank.npz from an earlier fit")
    _console.print(
        f"{_CHECK} Pulled endpoint [bold]{name}[/bold]'s current {' + '.join(wrote)} "
        f"into {dest_dir}"
    )
    return True


def _print_orgs(identity: WhoAmI, default_org: str | None) -> None:
    if not identity.orgs:
        _console.print("  no organizations visible to this key")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("organization")
    table.add_column("id")
    table.add_column("")
    for org in identity.orgs:
        marker = "default" if org.id == default_org else ""
        table.add_row(org.name, org.id, marker)
    _console.print(table)


def register(app: typer.Typer) -> None:
    """Attach the platform commands to the root CLI."""
    app.command("login")(login)
    app.command("logout")(logout)
    app.command("status")(status)
    app.command("push")(push)
    app.command("pull")(pull)


def __getattr__(name: str) -> object:
    """Lazy module attribute resolution for deferred CLI imports."""
    if name == "PlatformClient":
        from wmo.runtime.platform.client import PlatformClient

        return PlatformClient
    if name == "fetch_cli_config":
        from wmo.runtime.platform.client import fetch_cli_config

        return fetch_cli_config
    if name == "RunsReader":
        from wmo.runtime.runs.reader import RunsReader

        return RunsReader
    if name == "RunsSink":
        from wmo.runtime.runs.client import RunsSink

        return RunsSink
    if name == "optimize_events":
        from wmo.optimize.telemetry.backfill import optimize_events

        return optimize_events
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



