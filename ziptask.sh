#!/bin/bash -e

# Beware of a possible shell injection via an unescaped $@ variable.
if [[ $# -eq 0 ]] ; then
  echo "Usage $0 task-1.txt task-2.1.txt task-2.2.txt"
  exit
fi

TIMESTAMP=$(TZ=UTC date '+%Y%m%d-%H%M%S')
SUBMISSION=$(mktemp -d) || exit 1
trap 'rm -rf "$SUBMISSION"' EXIT

zip "$SUBMISSION/system.zip" -j $@
zip "codalab-$TIMESTAMP.zip" -j "$SUBMISSION/system.zip"

