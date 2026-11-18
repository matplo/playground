#!/usr/bin/env bash
#
# make-claude-profile-macosx.sh — create a macOS launcher app that starts Claude Desktop
# with its own, separate user-data directory (e.g. a second "work" account).
#
# Usage:
#   make-claude-profile-macosx.sh [options] NAME
#   make-claude-profile-macosx.sh --remove [options] NAME
#
#   NAME            Profile name (required). Creates:
#                     app:     ~/Applications/Claude <NAME>.app
#                     profile: ~/Library/Application Support/Claude-<NAME>
#                   Running without NAME does nothing but print this help.
#
# Options:
#   -a, --app PATH        Path to Claude.app        (default: /Applications/Claude.app)
#   -d, --dest DIR        Where to put the launcher (default: ~/Applications)
#   -p, --profile DIR     Explicit profile dir      (default: derived from NAME)
#   -c, --copy-config     Copy claude_desktop_config.json (MCP servers) from the default profile
#   -f, --force           Overwrite an existing launcher without asking
#   -n, --no-icon         Don't copy Claude's icon
#   -o, --open            Launch the new app when done
#   -r, --remove          Revert: delete the launcher app AND profile dir for NAME
#                         (asks for confirmation unless -f)
#   -h, --help            Show this help
#
# Examples:
#   ./make-claude-profile-macosx.sh -c -o Work
#   ./make-claude-profile-macosx.sh --remove Work

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
DEFAULT_PROFILE="$HOME/Library/Application Support/Claude"

# ---------- helpers ----------
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }
warn() { printf 'Warning: %s\n' "$*" >&2; }
usage() {
  # Works both from a file and when piped via `curl ... | bash -s --`
  if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  else
    printf 'Usage: make-claude-profile-macosx.sh [options] NAME\n'
    printf '       make-claude-profile-macosx.sh --remove [-f] NAME\n'
    printf 'Options: -a APP -d DEST -p PROFILE -c -f -n -o -r -h\n'
  fi
  exit "${1:-0}"
}

# Escape a string for use inside an AppleScript "..." literal
as_escape() { local s=${1//\\/\\\\}; printf '%s' "${s//\"/\\\"}"; }

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--app)         CLAUDE_APP="${2:?missing value for $1}"; shift 2 ;;
    -d|--dest)        DEST_DIR="${2:?missing value for $1}"; shift 2 ;;
    -p|--profile)     PROFILE_DIR="${2:?missing value for $1}"; shift 2 ;;
    -c|--copy-config) COPY_CONFIG=1; shift ;;
    -f|--force)       FORCE=1; shift ;;
    -n|--no-icon)     SET_ICON=0; shift ;;
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
[[ -d "$CLAUDE_APP" ]]  || die "Claude.app not found at: $CLAUDE_APP (use --app)."

APP_NAME="Claude $NAME"
APP_PATH="$DEST_DIR/$APP_NAME.app"
PROFILE_DIR="${PROFILE_DIR:-$HOME/Library/Application Support/Claude-$NAME}"

[[ "$PROFILE_DIR" != "$DEFAULT_PROFILE" ]] \
  || die "profile dir is the default Claude profile; pick another name."

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
if [[ $SET_ICON -eq 1 ]]; then
  icon=$(defaults read "$CLAUDE_APP/Contents/Info" CFBundleIconFile 2>/dev/null || true)
  icon_src="$CLAUDE_APP/Contents/Resources/${icon%.icns}.icns"
  if [[ -n "$icon" && -f "$icon_src" ]]; then
    cp "$icon_src" "$APP_PATH/Contents/Resources/applet.icns"
    touch "$APP_PATH"
    info "Copied Claude icon."
  else
    warn "could not find Claude's icon; keeping the default script icon."
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

info "Done: \"$APP_NAME\" is available in Spotlight/Launchpad."
[[ $OPEN_AFTER -eq 1 ]] && open "$APP_PATH"
exit 0