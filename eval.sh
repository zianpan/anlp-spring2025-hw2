#!/bin/bash


# mkdir -p test
echo "Running evaluation..."

python scripts/eval/evaluation_metrics.py \
  --references_dir test/references \
  --predictions_dir test/outputs \
  --output_dir test/eval_results \

echo "Multi-comparison complete! Results saved to test/eval_results/"
