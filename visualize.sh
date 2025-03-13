#! /bin/bash

for file in test/eval_results/*categories.json; do
    python scripts/eval/visualize.py $file
done
