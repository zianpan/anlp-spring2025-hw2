#!/bin/bash


# mkdir -p test
echo "Running evaluation..."

python eval/evaluation_metrics.py \
  --references_dir test/references \
  --predictions_dir test/outputs \
  --output_dir test/results \
  --semantic \
  --semantic_threshold 0.7

echo "Multi-comparison complete! Results saved to test/results/"
echo "Here's the summary:"
cat test/results/summary.txt