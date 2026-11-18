#!/usr/bin/env bash
#
# make-claude-profile-macosx.sh — create a macOS launcher app that starts Claude Desktop
# (or Claude Science) with its own, separate user-data directory (e.g. a second account).
#
# Usage:
#   make-claude-profile-macosx.sh -n NAME [options]
#   make-claude-profile-macosx.sh -n NAME --remove [-f]
#
#   NAME            Profile name (required; via -n/--name or as the last argument). Creates:
#                     app:     ~/Applications/<APP> <NAME>.app
#                     profile: ~/Library/Application Support/<APP>-<NAME>
#                   where <APP> is the target app's name ("Claude" or "Claude Science").
#                   Running without NAME does nothing but print this help.
#
# Options:
#   -n, --name NAME       Profile name (required), e.g. Work
#   -a, --app PATH        Path to the app to launch (default: /Applications/Claude.app)
#   -s, --science         Shortcut for: -a "/Applications/Claude Science.app"
#   -d, --dest DIR        Where to put the launcher (default: ~/Applications)
#   -p, --profile DIR     Explicit profile dir      (default: ~/Library/Application Support/<APP>-NAME)
#   -c, --copy-config     Copy claude_desktop_config.json (MCP servers) from the default profile
#   -f, --force           Don't ask: overwrite existing launcher / confirm --remove
#       --no-icon         Don't set the app's icon on the launcher
#   -o, --open            Launch the new app when done
#   -r, --remove          Revert: delete the launcher app AND profile dir for NAME
#                         (asks for confirmation unless -f)
#   -h, --help            Show this help
#
# Examples:
#   ./make-claude-profile-macosx.sh -n Work -c -o
#   ./make-claude-profile-macosx.sh -s -n Work
#   ./make-claude-profile-macosx.sh -n Work --remove

set -euo pipefail

# ---------- defaults ----------
NAME=""
REMOVE=0
CLAUDE_APP="/Applications/Claude.app"
DEST_DIR="$HOME/Applications"
PROFILE_DIR=""
COPY_CONFIG=0
FORCE=0
SET_ICON=1
OPEN_AFTER=0

# ---------- helpers ----------
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }
warn() { printf 'Warning: %s\n' "$*" >&2; }
usage() {
  # Works both from a file and when piped via `curl ... | bash -s --`
  if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  else
    printf 'Usage: make-claude-profile-macosx.sh -n NAME [options]\n'
    printf '       make-claude-profile-macosx.sh -n NAME --remove [-f]\n'
    printf 'Options: -n NAME -a APP -s -d DEST -p PROFILE -c -f -o -r --no-icon -h\n'
  fi
  exit "${1:-0}"
}

# Escape a string for use inside an AppleScript "..." literal
as_escape() { local s=${1//\\/\\\\}; printf '%s' "${s//\"/\\\"}"; }

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--app)         CLAUDE_APP="${2:?missing value for $1}"; shift 2 ;;
    -s|--science)     CLAUDE_APP="/Applications/Claude Science.app"; shift ;;
    -d|--dest)        DEST_DIR="${2:?missing value for $1}"; shift 2 ;;
    -p|--profile)     PROFILE_DIR="${2:?missing value for $1}"; shift 2 ;;
    -c|--copy-config) COPY_CONFIG=1; shift ;;
    -f|--force)       FORCE=1; shift ;;
    -n|--name)        [[ -z "$NAME" || "$NAME" == "${2:-}" ]] || die "only one NAME allowed."
                      NAME="${2:?missing value for $1}"; shift 2 ;;
    --no-icon)        SET_ICON=0; shift ;;
    -o|--open)        OPEN_AFTER=1; shift ;;
    -r|--remove)      REMOVE=1; shift ;;
    -h|--help)        usage 0 ;;
    --)               shift; break ;;
    -*)               die "unknown option: $1 (see --help)" ;;
    *)                [[ -z "$NAME" ]] || die "only one NAME allowed (got '$NAME' and '$1')."
                      NAME="$1"; shift ;;
  esac
done
[[ $# -gt 0 ]] && NAME="$1"

# No name given -> do nothing, just show help
if [[ -z "$NAME" ]]; then
  printf 'No profile NAME given — nothing done.\n\n' >&2
  usage 1
fi

# ---------- validation ----------
# ---------- platform check (macOS only) ----------
if [[ "$(uname -s)" != "Darwin" ]] || ! command -v sw_vers >/dev/null 2>&1; then
  die "this script is for macOS only (detected: $(uname -s))."
fi
info "macOS $(sw_vers -productVersion) detected."
command -v osacompile >/dev/null || die "osacompile not found."
[[ "$NAME" != */* ]]    || die "profile name must not contain '/'."
CLAUDE_APP="${CLAUDE_APP%/}"
[[ -d "$CLAUDE_APP" ]]  || die "app not found at: $CLAUDE_APP (use --app)."

# Base name of the target app, e.g. "Claude" or "Claude Science"
BASE_NAME="$(basename "$CLAUDE_APP" .app)"
# Its default (Electron) profile dir lives under its bundle name
BUNDLE_NAME="$(defaults read "$CLAUDE_APP/Contents/Info" CFBundleName 2>/dev/null || echo "$BASE_NAME")"
DEFAULT_PROFILE="$HOME/Library/Application Support/$BUNDLE_NAME"

APP_NAME="$BASE_NAME $NAME"
APP_PATH="$DEST_DIR/$APP_NAME.app"
PROFILE_DIR="${PROFILE_DIR:-$HOME/Library/Application Support/$BASE_NAME-$NAME}"

[[ "$PROFILE_DIR" != "$DEFAULT_PROFILE" ]] \
  || die "profile dir is the default $BASE_NAME profile; pick another name."

info "Launcher : $APP_PATH"
info "Profile  : $PROFILE_DIR"

# ---------- remove / revert ----------
if [[ $REMOVE -eq 1 ]]; then
  targets=()
  [[ -e "$APP_PATH" ]]    && targets+=("$APP_PATH")
  [[ -e "$PROFILE_DIR" ]] && targets+=("$PROFILE_DIR")
  if [[ ${#targets[@]} -eq 0 ]]; then
    info "Nothing to remove for profile '$NAME'."
    exit 0
  fi

  # Refuse while an instance using this profile is running
  if pgrep -f -- "--user-data-dir=$PROFILE_DIR" >/dev/null 2>&1; then
    die "a Claude instance using this profile is running — quit it first."
  fi

  printf 'The following will be PERMANENTLY deleted:\n'
  printf '  %s\n' "${targets[@]}"
  [[ -d "$PROFILE_DIR" ]] && printf '  (profile contains the sign-in, settings and cache of "%s")\n' "$APP_NAME"
  if [[ $FORCE -eq 0 ]]; then
    [[ -t 0 ]] || die "not interactive; re-run with -f to confirm removal."
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || die "aborted."
  fi
  rm -rf -- "${targets[@]}"
  info "Removed profile '$NAME'."
  exit 0
fi

# ---------- directories ----------
if [[ -d "$PROFILE_DIR" ]]; then
  info "Profile dir exists — reusing it (existing sign-in/settings kept)."
else
  mkdir -p "$PROFILE_DIR" || die "cannot create $PROFILE_DIR"
  info "Created profile dir."
fi

if [[ ! -d "$DEST_DIR" ]]; then
  mkdir -p "$DEST_DIR" || die "cannot create $DEST_DIR"
  info "Created $DEST_DIR."
fi

# ---------- existing launcher ----------
if [[ -e "$APP_PATH" ]]; then
  if [[ $FORCE -eq 0 ]]; then
    [[ -t 0 ]] || die "$APP_PATH exists; re-run with -f to overwrite."
    read -r -p "$APP_PATH exists. Overwrite? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || die "aborted."
  fi
  rm -rf "$APP_PATH"
fi

# ---------- build launcher ----------
osacompile -o "$APP_PATH" \
  -e "set appPath to \"$(as_escape "$CLAUDE_APP")\"" \
  -e "set profileDir to \"$(as_escape "$PROFILE_DIR")\"" \
  -e 'do shell script "open -n -a " & quoted form of appPath & " --args --user-data-dir=" & quoted form of profileDir' \
  || die "osacompile failed."
info "Built launcher app."

# ---------- icon ----------
# Recent osacompile applets carry their icon in Assets.car, which overrides
# applet.icns — so set a Finder "custom icon" taken directly from Claude.app.
if [[ $SET_ICON -eq 1 ]]; then
  ok=$(osascript -l JavaScript -e '
ObjC.import("AppKit");
function run(a){ const ws=$.NSWorkspace.sharedWorkspace;
  return ws.setIconForFileOptions(ws.iconForFile(a[0]), a[1], 0); }' \
    "$CLAUDE_APP" "$APP_PATH" 2>/dev/null || echo false)
  if [[ "$ok" == "true" ]]; then
    info "Set Claude icon on launcher."
  else
    warn "could not set icon; keeping the default script icon."
  fi
fi

# ---------- MCP config ----------
if [[ $COPY_CONFIG -eq 1 ]]; then
  src="$DEFAULT_PROFILE/claude_desktop_config.json"
  dst="$PROFILE_DIR/claude_desktop_config.json"
  if [[ ! -f "$src" ]]; then
    warn "no config at $src; nothing copied."
  elif [[ -f "$dst" && $FORCE -eq 0 ]]; then
    warn "$dst already exists; not overwritten (use --force)."
  else
    cp "$src" "$dst"
    info "Copied claude_desktop_config.json."
  fi
fi

# ---------- register & finish ----------
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[[ -x "$LSREG" ]] && "$LSREG" -f "$APP_PATH" >/dev/null 2>&1 || true
# Nudge Finder/Spotlight to pick up the new icon
touch "$APP_PATH"
command -v mdimport >/dev/null && mdimport "$APP_PATH" >/dev/null 2>&1 || true

info "Done: \"$APP_NAME\" is available in Spotlight/Launchpad."
[[ $OPEN_AFTER -eq 1 ]] && open "$APP_PATH"
exit 0