#!/usr/bin/env bash
# Set up every Table-1 baseline (clone official repos at pinned commits, create envs, fetch weights).
set -uo pipefail
cd "$(dirname "$0")"
status=0
for s in alphaflow confdiff bioemu str2str mdgen eba biomd_biokinema; do
  echo "==================== $s"
  bash "$s.sh" || { echo "!! $s setup failed"; status=1; }
done
exit $status
