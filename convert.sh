#!/bin/bash

python scripts/utils/csv_to_json.py \
    --csv_file results/test_30.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/

python scripts/utils/csv_to_json.py \
    --csv_file results/test_qa_pairs_new_100_2.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/

python scripts/utils/csv_to_json.py \
    --csv_file results/test_qa_pairs_new_600_38.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/