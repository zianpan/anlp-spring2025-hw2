#!/bin/bash

python scripts/utils/csv_to_json.py \
    --csv_file annotations/qa_pairs.csv \
    --output_file annotations/qa_pairs.json

echo "Conversion complete! Results saved to annotations/qa_pairs.json"