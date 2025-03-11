#!/bin/bash

for file in results/labeled/*.csv; do
    python scripts/utils/csv_to_json.py \
        --csv_file $file \
        --output_dir test/outputs/ \
        --reference_dir test/references/
done
