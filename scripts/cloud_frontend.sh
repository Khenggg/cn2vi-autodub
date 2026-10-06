#!/usr/bin/env bash
set -Eeuo pipefail

NODE_VERSION="24.14.1"
NODE_ARCHIVE="node-v${NODE_VERSION}-linux-x64.tar.xz"
NODE_SHA256="84d38715d449447117d05c3e71acd78daa49d5b1bfa8aacf610303920c3322be"
NODE_NPM_VERSION="11.11.0"
NODE_URL="https://nodejs.org/download/release/v${NODE_VERSION}/${NODE_ARCHIVE}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
NODE_PARENT="$DATA_ROOT/cache/autodub-node"
NODE_DIR="$NODE_PARENT/v${NODE_VERSION}-linux-x64"

if [[ ! -f "$FRONTEND_DIR/package.json" || ! -f "$FRONTEND_DIR/package-lock.json" ]]; then
  printf 'cloud_frontend: frontend package.json and package-lock.json are required\n' >&2
  exit 66
fi

mkdir -p -- "$NODE_PARENT" "$DATA_ROOT/cache/npm"
archive_tmp=""
extract_tmp=""
cleanup() {
  if [[ -n "$archive_tmp" && -f "$archive_tmp" ]]; then
    rm -f -- "$archive_tmp"
  fi
  if [[ -n "$extract_tmp" && -d "$extract_tmp" ]]; then
    rm -rf -- "$extract_tmp"
  fi
}
trap cleanup EXIT

verify_node_install() {
  local node_version npm_version marker
  marker="$NODE_DIR/.autodub-node-sha256"
  [[ -f "$marker" ]] || return 1
  [[ "$(cat -- "$marker")" == "$NODE_SHA256" ]] || return 1
  [[ -x "$NODE_DIR/bin/node" && -x "$NODE_DIR/bin/npm" ]] || return 1
  node_version="$("$NODE_DIR/bin/node" --version 2>/dev/null)" || return 1
  npm_version="$(PATH="$NODE_DIR/bin:$PATH" "$NODE_DIR/bin/npm" --version 2>/dev/null)" || return 1
  [[ "$node_version" == "v$NODE_VERSION" && "$npm_version" == "$NODE_NPM_VERSION" ]]
}

if [[ -e "$NODE_DIR" || -L "$NODE_DIR" ]]; then
  if ! verify_node_install; then
    printf 'cloud_frontend: existing pinned Node directory failed verification; preserving it: %s\n' \
      "$NODE_DIR" >&2
    exit 1
  fi
  printf 'cloud_frontend: reusing verified Node v%s from %s\n' "$NODE_VERSION" "$NODE_DIR"
else
  if ! command -v curl >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1 || \
    ! command -v tar >/dev/null 2>&1 || ! command -v xz >/dev/null 2>&1; then
    printf 'cloud_frontend: curl, sha256sum, tar, and xz are required; run cloud_bootstrap.sh first\n' >&2
    exit 69
  fi
  archive_tmp="$(mktemp "$DATA_ROOT/cache/.${NODE_ARCHIVE}.XXXXXX")"
  printf 'cloud_frontend: downloading pinned Node v%s\n' "$NODE_VERSION"
  curl --fail --location --retry 3 --output "$archive_tmp" "$NODE_URL"
  printf '%s  %s\n' "$NODE_SHA256" "$archive_tmp" | sha256sum --check --status || {
    printf 'cloud_frontend: official Node archive checksum mismatch\n' >&2
    exit 1
  }
  tar --list --xz --file "$archive_tmp" | awk -v root="node-v${NODE_VERSION}-linux-x64/" '
    BEGIN { bad=0; count=0 }
    {
      count++
      if (substr($0, 1, length(root)) != root || $0 ~ /(^|\/)\.\.(\/|$)/ || $0 ~ /^\//) bad=1
    }
    END { if (count == 0 || bad) exit 1 }
  ' || {
    printf 'cloud_frontend: Node archive contains an unexpected path; refusing extraction\n' >&2
    exit 1
  }
  if [[ -e "$NODE_DIR" || -L "$NODE_DIR" ]]; then
    printf 'cloud_frontend: Node install target appeared during setup; preserving it: %s\n' "$NODE_DIR" >&2
    exit 1
  fi
  extract_tmp="$(mktemp -d "$NODE_PARENT/.node-extract.XXXXXX")"
  tar --extract --xz --file "$archive_tmp" --directory "$extract_tmp" \
    --strip-components=1 --no-same-owner --no-same-permissions
  [[ -x "$extract_tmp/bin/node" && -x "$extract_tmp/bin/npm" ]] || {
    printf 'cloud_frontend: verified archive did not contain Node and npm executables\n' >&2
    exit 1
  }
  printf '%s\n' "$NODE_SHA256" > "$extract_tmp/.autodub-node-sha256"
  if [[ -e "$NODE_DIR" || -L "$NODE_DIR" ]]; then
    printf 'cloud_frontend: Node install target appeared during setup; preserving it: %s\n' "$NODE_DIR" >&2
    exit 1
  fi
  mv -T -- "$extract_tmp" "$NODE_DIR"
  extract_tmp=""
  rm -f -- "$archive_tmp"
  archive_tmp=""
fi

if ! verify_node_install; then
  printf 'cloud_frontend: installed Node/npm versions or verification marker do not match pins\n' >&2
  exit 1
fi

export PATH="$NODE_DIR/bin:$PATH"
export npm_config_cache="$DATA_ROOT/cache/npm"
export npm_config_audit=false
export npm_config_fund=false
export npm_config_update_notifier=false
printf 'cloud_frontend: npm ci and production build in %s\n' "$FRONTEND_DIR"
cd -- "$FRONTEND_DIR"
npm ci --no-audit --no-fund
npm run build
[[ -s "$FRONTEND_DIR/dist/index.html" ]] || {
  printf 'cloud_frontend: build did not produce dist/index.html\n' >&2
  exit 1
}
printf 'cloud_frontend: READY node=%s npm=%s dist=%s\n' \
  "$NODE_VERSION" "$NODE_NPM_VERSION" "$FRONTEND_DIR/dist/index.html"
