#!/bin/bash

# Script to convert final_project_report.md to PDF with rendered Mermaid diagrams using Pandoc and Lua filters

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
INPUT_MD="final_project_report.md"
OUTPUT_PDF="final_project_report.pdf"
LUA_FILTER="mermaid.lua"

# Check for required tools
command -v pandoc >/dev/null 2>&1 || {
  echo >&2 "Pandoc is required but it's not installed. Aborting."
  exit 1
}
command -v mmdc >/dev/null 2>&1 || {
  echo >&2 "Mermaid CLI (mmdc) is required but it's not installed. Aborting."
  exit 1
}

# Check if Lua filter exists
if [ ! -f "$LUA_FILTER" ]; then
  echo "Lua filter '$LUA_FILTER' not found. Please ensure it exists in the current directory."
  exit 1
fi

# Convert the Markdown to PDF using Pandoc with the Lua filter
pandoc "$INPUT_MD" \
  --from markdown \
  --to pdf \
  --output "$OUTPUT_PDF" \
  --pdf-engine=xelatex \
  --lua-filter="$LUA_FILTER" \
  --variable geometry:margin=1in \
  --highlight-style=tango \
  --toc

echo "Conversion complete! Output PDF: $OUTPUT_PDF"
