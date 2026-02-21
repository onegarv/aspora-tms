#!/bin/bash
# Discovers all Prometheus metrics across services in ~/code/aspora/prom/

PROM_DIR=~/code/aspora/prom
OUTPUT_FILE="metrics-discovery-report.md"

echo "# Metrics Discovery Report" > "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for service_dir in "$PROM_DIR"/*/ ; do
  service=$(basename "$service_dir")

  # Skip non-service directories
  if [[ "$service" == "java-commons" ]]; then
    continue
  fi

  echo "## Service: $service" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"

  # Find MetricsUtil.java files
  metrics_files=$(find "$service_dir" -name "MetricsUtil.java" -o -name "MetricsConstants.java" 2>/dev/null)

  if [[ -z "$metrics_files" ]]; then
    echo "❌ No MetricsUtil found" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    continue
  fi

  echo "### Discovered Metrics" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"

  # Extract metric names from the files
  for file in $metrics_files; do
    echo "**Source:** \`$(basename "$file")\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Extract Counter.builder and Timer.builder calls
    grep -E "Counter\.builder|Timer\.builder" "$file" | \
      sed -E 's/.*"([^"]+)".*/- \1/' | \
      sort -u >> "$OUTPUT_FILE"

    echo "" >> "$OUTPUT_FILE"
  done

  echo "---" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
done

echo "✅ Discovery complete. Report saved to $OUTPUT_FILE"
