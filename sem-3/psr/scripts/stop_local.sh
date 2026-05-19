#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .local/pids ]; then
  echo "No local app PID file found."
  exit 0
fi

while read -r pid; do
  if [ -n "$pid" ]; then
    kill "$pid" >/dev/null 2>&1 || true
  fi
done < .local/pids

rm -f .local/pids
echo "Stopped Book AI Library local app."
