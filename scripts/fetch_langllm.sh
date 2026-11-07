#!/bin/sh
# Download the LangLLM files this project builds on, at the pinned commit. They are not redistributed here.
# Source: https://github.com/adrian-erlikhman/LangLLM (Erlikhman, Tarekegn and Juang, IEEE BigData 2026 HSS)
set -e
C=06109dc206a7f34a57be751e66fb5c73e4993bfc
RAW="https://raw.githubusercontent.com/adrian-erlikhman/LangLLM/$C"
cd "$(dirname "$0")/.."

mkdir -p data/original/raw vendor/langllm/langllm
curl -sf "$RAW/data/features/features.csv" -o data/original/features.csv
curl -sf "$RAW/results/rq3_transfer_summary.json" -o data/original/rq3_transfer_summary.json
curl -sf "$RAW/results/rq3_transfer_matrix.csv" -o data/original/rq3_transfer_matrix.csv
for m in gpt gemini claude grok deepseek; do
  curl -sf "$RAW/data/raw/$m.jsonl" -o "data/original/raw/$m.jsonl"
done
for f in langllm/__init__.py langllm/config.py langllm/features.py config.yaml; do
  curl -sf "$RAW/$f" -o "vendor/langllm/$f"
done
# Point the extractor at Stanza 1.14.0 models (the version that reproduces features.csv exactly).
sed -i.bak 's|stanza_dir: stanza_resources.*|stanza_dir: stanza_resources_1.14|' vendor/langllm/config.yaml && rm vendor/langllm/config.yaml.bak
echo "Fetched LangLLM data and extractor at $C."
