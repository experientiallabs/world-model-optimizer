"""Platform commands: `wmh login`, `wmh logout`, `wmh status`, `wmh push`, `wmh pull`.

`login` connects this machine to a platform account (browser flow by default,
`--token` for headless); `push`/`pull` round-trip world models and harnesses
against the platform registry, auto-detecting the artifact kind from what
exists locally (or remotely, for pulls).
"""

from __future__ import annotations

import socket
import tempfile
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wmh.config.store import WorldModelStore
from wmh.harness.doc import HarnessDoc
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.platform.auth import BrowserLogin
from wmh.platform.client import PlatformClient, PlatformError, WhoAmI, fetch_cli_config
from wmh.platform.credentials import (
    DEFAULT_WEB_URL,
    PlatformCredentials,
    clear_credentials,
    credentials_path,
    load_credentials,
    save_credentials,
)
from wmh.platform.transfer import extract_push_meta, pack_model_dir, unpack_model_bundle

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
_KIND = typer.Option(
    "--kind", help="Artifact kind: model or harness (auto-detected when unambiguous)."
)
_PUSH_AS = typer.Option(
    "--as", help="Remote name to publish under (local names may not be slug-safe)."
)
_PUSH_REF = typer.Option(
    "--ref", help="Harness version or alias to push (default: champion, else latest)."
)
_PULL_VERSION = typer.Option("--version", help="Harness version to pull (default: latest).")
_PULL_FORCE = typer.Option("--force", help="Replace an existing local artifact.")
_ROOT = typer.Option("--root", help="Artifact root directory.")


def login(
    url: Annotated[str | None, _LOGIN_URL] = None,
    api_url: Annotated[str | None, _LOGIN_API_URL] = None,
    token: Annotated[str | None, _LOGIN_TOKEN] = None,
    no_browser: Annotated[bool, _LOGIN_NO_BROWSER] = False,
) -> None:
    """Connect this machine to a platform account."""
    credentials = load_credentials()
    web_url = (url or credentials.web_url or DEFAULT_WEB_URL).rstrip("/")

    if api_url is None:
        try:
            api_url = fetch_cli_config(web_url)
        except PlatformError as error:
            raise typer.BadParameter(f"{web_url} does not look like a platform: {error}") from error
        if api_url is None:
            raise typer.BadParameter(f"{web_url} did not advertise a backend URL; is it deployed?")
    else:
        api_url = api_url.rstrip("/")

    if token is None:
        token = _browser_login(web_url, open_browser=not no_browser)
    if token is None or not token.strip():
        _console.print("[red]No key received; nothing saved.[/red]")
        raise typer.Exit(code=1)
    token = token.strip()

    with PlatformClient(api_url, token) as client:
        try:
            identity = client.whoami()
        except PlatformError as error:
            _console.print(f"[red]The key was rejected:[/red] {error}")
            raise typer.Exit(code=1) from error

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
    credentials = load_credentials()
    if not credentials.is_complete():
        _console.print(
            f"Not connected (no credential at {credentials_path()}). Run [bold]wmh login[/bold]."
        )
        raise typer.Exit(code=1)
    with _client(credentials) as client:
        try:
            identity = client.whoami()
        except PlatformError as error:
            _console.print(f"[red]Connection check failed:[/red] {error}")
            raise typer.Exit(code=1) from error
    _console.print(f"{_CHECK} Connected to [bold]{credentials.web_url}[/bold]")
    _console.print(f"  acting as: {identity.actor.kind} {identity.actor.id}")
    _print_orgs(identity, credentials.default_org)


def push(
    name: Annotated[str, typer.Argument(help="Local world model or harness name.")],
    org: Annotated[str | None, _ORG] = None,
    kind: Annotated[str | None, _KIND] = None,
    push_as: Annotated[str | None, _PUSH_AS] = None,
    ref: Annotated[str | None, _PUSH_REF] = None,
    root: Annotated[str, _ROOT] = ".wmh",
) -> None:
    """Publish a local world model or harness to the platform registry."""
    model_dir = WorldModelStore(root).dir_for(name)
    harness_exists = HarnessStore(root).exists(name)
    resolved_kind = _resolve_kind(kind, model=model_dir is not None, harness=harness_exists)
    remote_name = push_as or name

    credentials, org_id = _require_connection(org)
    with _client(credentials) as client:
        if resolved_kind == "model" and model_dir is not None:
            _push_model(client, org_id, remote_name, model_dir)
        else:
            _push_harness(client, org_id, remote_name, name, ref, root)


def pull(
    name: Annotated[str, typer.Argument(help="Remote world model or harness name.")],
    org: Annotated[str | None, _ORG] = None,
    kind: Annotated[str | None, _KIND] = None,
    version: Annotated[int | None, _PULL_VERSION] = None,
    force: Annotated[bool, _PULL_FORCE] = False,
    root: Annotated[str, _ROOT] = ".wmh",
) -> None:
    """Fetch a world model or harness from the platform registry."""
    if kind is not None and kind not in ("model", "harness"):
        raise typer.BadParameter("--kind must be 'model' or 'harness'")
    credentials, org_id = _require_connection(org)
    with _client(credentials) as client:
        resolved_kind = kind or _detect_remote_kind(client, org_id, name)
        if resolved_kind == "model":
            _pull_model(client, org_id, name, root, force=force)
        else:
            _pull_harness(client, org_id, name, root, version=version)


# -- helpers -------------------------------------------------------------------------------------


def _browser_login(web_url: str, *, open_browser: bool) -> str | None:
    """Run the loopback browser flow; fall back to a hidden paste prompt."""
    login_attempt = BrowserLogin(web_url)
    try:
        login_attempt.start()
        key_name = f"wmh on {socket.gethostname()}"
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


def default_org(client: PlatformClient, configured: str | None) -> str:
    """Resolve the login's organization, auto-picking only an unambiguous sole org."""
    if configured is not None:
        return configured
    identity = client.whoami()
    if len(identity.orgs) == 1:
        return identity.orgs[0].id
    raise typer.BadParameter(
        "no default organization selected; run `wmh login` again and choose an organization"
    )


def _client(credentials: PlatformCredentials) -> PlatformClient:
    if credentials.api_url is None or credentials.token is None:
        raise typer.BadParameter("not connected to a platform; run `wmh login` first")
    return PlatformClient(credentials.api_url, credentials.token)


def _require_connection(org: str | None) -> tuple[PlatformCredentials, str]:
    credentials = load_credentials()
    if not credentials.is_complete():
        raise typer.BadParameter("not connected to a platform; run `wmh login` first")
    org_id = org or credentials.default_org
    if not org_id:
        raise typer.BadParameter(
            "no organization selected; pass --org <id> (see `wmh status` for your organizations)"
        )
    return credentials, org_id


def _resolve_kind(kind: str | None, *, model: bool, harness: bool) -> str:
    if kind is not None:
        if kind not in ("model", "harness"):
            raise typer.BadParameter("--kind must be 'model' or 'harness'")
        if kind == "model" and not model:
            raise typer.BadParameter("no local world model has this name")
        if kind == "harness" and not harness:
            raise typer.BadParameter("no local harness has this name")
        return kind
    if model and harness:
        raise typer.BadParameter("both a model and a harness have this name locally; pass --kind")
    if model:
        return "model"
    if harness:
        return "harness"
    raise typer.BadParameter("no local world model or harness has this name")


def _detect_remote_kind(client: PlatformClient, org_id: str, name: str) -> str:
    model_names = {model.name for model in client.list_world_models(org_id)}
    harness_names = {harness.name for harness in client.list_harnesses(org_id)}
    if name in model_names and name in harness_names:
        raise typer.BadParameter("both a model and a harness have this name remotely; pass --kind")
    if name in model_names:
        return "model"
    if name in harness_names:
        return "harness"
    raise typer.BadParameter(f"the organization has no world model or harness named {name!r}")


def _push_model(client: PlatformClient, org_id: str, remote_name: str, model_dir: Path) -> None:
    meta = extract_push_meta(model_dir)
    with tempfile.TemporaryDirectory(prefix="wmh-push-") as staging:
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


def _push_harness(
    client: PlatformClient,
    org_id: str,
    remote_name: str,
    local_name: str,
    ref: str | None,
    root: str,
) -> None:
    doc = HarnessStore(root).load(local_name, ref)
    if remote_name != local_name:
        doc = doc.model_copy(update={"name": remote_name})
    pushed = client.push_harness_version(
        org_id, remote_name, doc.model_dump(mode="json"), doc.doc_hash
    )
    if pushed.created:
        _console.print(
            f"{_CHECK} Pushed harness [bold]{pushed.name}[/bold] as remote v{pushed.version}"
        )
    else:
        _console.print(
            f"Remote [bold]{pushed.name}[/bold] v{pushed.version} already has this exact doc; "
            "nothing to push."
        )


def _pull_model(client: PlatformClient, org_id: str, name: str, root: str, *, force: bool) -> None:
    dest_dir = WorldModelStore(root).model_dir(name)
    with tempfile.TemporaryDirectory(prefix="wmh-pull-") as staging:
        bundle_path = Path(staging) / f"{name}.tar.gz"
        client.download_model_bundle(org_id, name, bundle_path)
        try:
            unpack_model_bundle(bundle_path, dest_dir, force=force)
        except FileExistsError as error:
            raise typer.BadParameter(str(error)) from error
    _console.print(f"{_CHECK} Pulled world model [bold]{name}[/bold] into {dest_dir}")


def _pull_harness(
    client: PlatformClient, org_id: str, name: str, root: str, *, version: int | None
) -> None:
    if version is None:
        harness, _versions = client.get_harness(org_id, name)
        if harness.latest_version < 1:
            raise typer.BadParameter(f"remote harness {name!r} has no versions yet")
        version = harness.latest_version
    payload = client.get_harness_version(org_id, name, version)
    doc = HarnessDoc.model_validate(payload.doc)
    # Versions the platform recorded before doc_hash covered materialized paths carry the legacy
    # hash; accept either so pre-change pathful (pi-node) records stay pullable.
    if payload.doc_hash not in (doc.doc_hash, doc.legacy_doc_hash):
        _console.print("[red]Pulled doc failed its integrity check; not saving.[/red]")
        raise typer.Exit(code=1)
    saved = HarnessStore(root).save_version(
        doc.model_copy(update={"name": name}), alias=CHAMPION_ALIAS
    )
    _console.print(
        f"{_CHECK} Pulled harness [bold]{name}[/bold] remote v{version} → local v{saved.version} "
        f"(champion)"
    )


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
