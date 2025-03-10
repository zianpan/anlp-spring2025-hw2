import csv
import json
import os
import argparse

def convert_csv_to_json(csv_string, column_name):
    csv_reader = csv.reader(csv_string.strip().split('\n'))
    headers = next(csv_reader)
    
    target_column_index = headers.index(column_name)
    
    result_dict = {}
    
    for i, row in enumerate(csv_reader, 1):
        if len(row) > target_column_index:
            result_dict[str(i)] = row[target_column_index]
    
    return json.dumps(result_dict, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV columns to separate JSON files.')
    parser.add_argument('--csv_file', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON files')
    parser.add_argument('--reference_dir', type=str, required=True, help='Column name to convert to JSON')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(args.csv_file, 'r') as f:
        csv_data = f.read()
    
    base_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    
    answer_json = convert_csv_to_json(csv_data, 'answer')
    answer_output_path = os.path.join(args.output_dir, f"{base_filename}_answers.json")
    with open(answer_output_path, 'w') as f:
        f.write(answer_json)
    
    reference_json = convert_csv_to_json(csv_data, 'reference_answer')
    reference_output_path = os.path.join(args.reference_dir, f"{base_filename}_answers.json")
    with open(reference_output_path, 'w') as f:
        f.write(reference_json)
    
    print(f"Created {answer_output_path}")
    print(f"Created {reference_output_path}")