#!/bin/bash


# mkdir -p test
echo "Running evaluation..."

python scripts/eval/evaluation_metrics.py \
  --references_dir test/references \
  --predictions_dir test/outputs \
  --output_dir test/eval_results \
  --semantic \
  --semantic_threshold 0.7

echo "Multi-comparison complete! Results saved to test/eval_results/"
echo "Here's the summary:"
cat test/eval_results/summary.txt