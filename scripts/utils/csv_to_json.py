import csv
import json
import os
import argparse

def convert_reference_answers_to_json(csv_string):
    csv_reader = csv.reader(csv_string.strip().split('\n'))
    headers = next(csv_reader)  # Skip the header row
    reference_answer_index = headers.index('reference_answer')
    reference_answers = {}
    
    for i, row in enumerate(csv_reader, 1):
        if len(row) > reference_answer_index:
            reference_answers[str(i)] = row[reference_answer_index]
    
    return json.dumps(reference_answers, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--csv_file', type=str, required=True)
    args.add_argument('--output_file', type=str, required=True)
    args = args.parse_args()
        
    csv_data = open(args.csv_file, 'r').read()

    json_output = convert_reference_answers_to_json(csv_data)

    with open(args.output_file, 'w') as f:
        f.write(json_output)