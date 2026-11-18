# make-claude-profile-macosx.sh

> **macOS only.** This script relies on `osacompile`, `open`, `defaults` and
> LaunchServices. It does not work on Linux or Windows.

This script runs **a second Claude Desktop instance** with its own separate
profile, for example to keep a work subscription next to a personal one. It
makes a small launcher app that starts Claude Desktop with its own
`--user-data-dir`, so each instance keeps its own sign-in, settings and cache.

| | Default (existing) | Created by this script (`NAME=Work`) |
|---|---|---|
| App | `/Applications/Claude.app` | `~/Applications/Claude Work.app` |
| Profile | `~/Library/Application Support/Claude` | `~/Library/Application Support/Claude-Work` |

## Requirements

- macOS with Claude Desktop installed (default location `/Applications/Claude.app`)
- `bash`, which comes with macOS

## Quick start

Download the script, make it executable, and run it:

```bash
curl -fsSL -o make-claude-profile-macosx.sh \
  https://raw.githubusercontent.com/matplo/playground/main/tools/claude/make-claude-profile-macosx.sh
chmod +x make-claude-profile-macosx.sh
./make-claude-profile-macosx.sh -o Work
```

Or run it in one line without saving it. Arguments go after `-s --`:

```bash
curl -fsSL https://raw.githubusercontent.com/matplo/playground/main/tools/claude/make-claude-profile-macosx.sh \
  | bash -s -- -f -o Work
```

> When you pipe the script, it can't ask whether to overwrite an existing
> launcher, so pass `-f` if you are re-creating one. As with any script piped
> from the internet, read it before running it.

When it finishes, **Claude Work** shows up in Spotlight and Launchpad and can be
dragged to the Dock.

## Usage

```
make-claude-profile-macosx.sh [options] [NAME]

  NAME                  Profile name (default: "Work")

  -a, --app PATH        Path to Claude.app        (default: /Applications/Claude.app)
  -d, --dest DIR        Where to put the launcher (default: ~/Applications)
  -p, --profile DIR     Explicit profile dir      (default: ~/Library/Application Support/Claude-NAME)
  -c, --copy-config     Copy claude_desktop_config.json (local MCP servers) from the default profile
  -f, --force           Overwrite an existing launcher/config without asking
  -n, --no-icon         Don't copy Claude's icon
  -o, --open            Launch the new app when done
  -h, --help            Show help
```

Examples:

```bash
./make-claude-profile-macosx.sh                 # "Claude Work"
./make-claude-profile-macosx.sh Lab             # "Claude Lab", profile Claude-Lab
./make-claude-profile-macosx.sh -c -o Work      # copy local MCP config, then launch
```

## What the script does

1. Checks that it is running on macOS and that `osacompile` and Claude.app exist.
2. Creates the profile folder, or reuses it if it already exists. An existing
   sign-in there is kept.
3. Builds an AppleScript launcher app that runs
   `open -n -a Claude.app --args --user-data-dir=<profile>`.
4. Copies Claude's icon into the launcher (unless you pass `-n`).
5. Optionally copies `claude_desktop_config.json` (`-c`).
6. Registers the launcher with LaunchServices so Spotlight finds it.

The script never touches the default profile, and it refuses to use that
profile as the target.

## Should I copy the MCP config (`-c`)?

Usually not. By default the new profile starts empty.

- `claude_desktop_config.json` lists only **local** MCP servers (the ones that
  run on your Mac). Cloud connectors such as Gmail and Drive belong to the
  account, so they don't depend on this file.
- The config can hold API keys or tokens in its `env` blocks, and starting fresh
  keeps personal and work credentials apart.
- Use `-c` only if you want the same local tools in both instances. Even then,
  copying just the one server entry you need is often cleaner.

## Caveats

- **Dock icon:** the launcher starts Claude and then quits, so the running window
  still shows up in the Dock as "Claude". Giving one instance a different theme
  helps tell them apart.
- **Signing in:** the sign-in step sends you back to the app through a
  `claude://` link, and macOS may hand that link to the instance that is already
  open. If that happens, quit the other instance, sign in with the new launcher,
  then reopen the other one. You only need to do this once, because the session
  is saved in the profile.
- **Updates:** both instances run the same copy of Claude.app, so they update
  together.

## Uninstall

```bash
rm -rf "$HOME/Applications/Claude Work.app"
rm -rf "$HOME/Library/Application Support/Claude-Work"   # deletes that profile's sign-in & data
```
