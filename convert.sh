#!/bin/bash

python scripts/utils/csv_to_json.py \
    --csv_file results/test_30.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/

python scripts/utils/csv_to_json.py \
    --csv_file results/test_qa_pairs_new_400.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/

python scripts/utils/csv_to_json.py \
    --csv_file results/test_qa_pairs_400_no_rag.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/

python scripts/utils/csv_to_json.py \
    --csv_file results/test_qa_pairs_new_2500_38.csv \
    --output_dir test/outputs/ \
    --reference_dir test/references/