#!/usr/bin/env bash
set -e
: "${LA_TEST_GGUF:?}"; : "${LA_CLI_BIN:?}"
OUT=$(mktemp -d)
"$LA_CLI_BIN" detect --model "$LA_TEST_GGUF" --input tests/fixtures/parity_image.png \
  --prompt "Locate all the instances that matches the following description: cat</c>remote." \
  --mode slow --output "$OUT/out.json" --annotated "$OUT/ann.png"
grep -q '"label":"cat"' "$OUT/out.json"
grep -q '"label":"remote"' "$OUT/out.json"
test -s "$OUT/ann.png"
echo "cli smoke OK"
