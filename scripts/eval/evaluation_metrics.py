#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, util
import string
import csv

# =======================================load and process answers======================================
def process_answer(answer):
    """
    Process an answer that might be a string or a dictionary with 'answer' field.
    
    Args:
        answer: String or dictionary containing answer
        
    Returns:
        List containing the processed answer
    """
    # If answer is a dictionary with 'answer' key, extract the actual answer text
    if isinstance(answer, dict) and 'answer' in answer:
        answer_text = answer['answer']
    else:
        answer_text = answer
    
    # Process the extracted answer text
    if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
        return []
    
    return [answer_text.strip()]

def get_tokens(answer):
    answer = str(answer).lower()
    answer = answer.replace(u'\u00a0', ' ')
    while len(answer) > 1 and answer[0] in set(string.whitespace + string.punctuation):
        answer = answer[1:]
    while len(answer) > 1 and answer[-1] in set(string.whitespace + string.punctuation):
        answer = answer[:-1]
    answer = answer.split()
    if len(answer) > 1 and answer[0] in {"a", "an", "the"}:
        answer = answer[1:]
    processed_answer = ' '.join(answer)
    for delimiter in set(string.whitespace + string.punctuation):
        processed_answer = processed_answer.replace(delimiter, ' ')
    return processed_answer.split()

def load_reference_answers(reference_file: str) -> dict:
    """
    Load a JSON file mapping question IDs to answers.
    Handles both simple answer strings and dictionaries with 'answer' field.
    """
    with open(reference_file, 'r', encoding='utf-8') as f:
        raw_answers = json.load(f)
    return {qid: process_answer(ans) for qid, ans in raw_answers.items()}

def load_system_predictions(prediction_file: str) -> dict:
    """
    Load system predictions from a JSON file.
    Handles both simple answer strings and dictionaries with 'answer' field.
    """
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    return {qid: process_answer(pred) for qid, pred in predictions.items()}

def extract_categories_from_json(json_file: str) -> dict:
    """
    Extract categories from a JSON file where each entry might have a 'category' field.
    
    Args:
        json_file: Path to JSON file with potential category information
        
    Returns:
        Dictionary mapping question IDs to categories
    """
    categories = {}
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for qid, entry in data.items():
            if isinstance(entry, dict) and 'category' in entry:
                categories[qid] = entry['category']
    except Exception as e:
        print(f"Error extracting categories from JSON file: {e}")
    
    return categories

def load_categories(csv_file: str = None, json_file: str = None) -> dict:
    """
    Load categories from a CSV or JSON file.
    
    Args:
        csv_file: Path to CSV file containing questions with categories
        json_file: Path to JSON file containing entries with categories
        
    Returns:
        Dictionary mapping question IDs to categories
    """
    categories = {}
    
    # Try loading from CSV if provided
    if csv_file and os.path.exists(csv_file):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    if 'category' in row:
                        categories[str(i)] = row['category']
        except Exception as e:
            print(f"Error loading categories from CSV: {e}")
    
    # If no categories loaded from CSV or CSV not provided, try JSON
    if not categories and json_file and os.path.exists(json_file):
        categories = extract_categories_from_json(json_file)
    
    return categories

# =======================================evaluation metrics======================================
def exact_match(predictions_dict, references_dict):
    def exact_match_single(reference_list, prediction):
        """Check if normalized prediction matches any normalized reference answer."""
        if not prediction or not reference_list:
            return 0
        for reference in reference_list:
            if prediction.lower() == reference.lower():
                return 1
        return 0

    total_questions = 0
    exact_match_count = 0
    for qid in references_dict:
        reference_list = references_dict[qid]
        prediction_list = predictions_dict.get(qid, [])
        
        if reference_list and prediction_list:
            prediction = prediction_list[0]  # Take first prediction
            exact_match_count += exact_match_single(reference_list, prediction)
        
        total_questions += 1

    if total_questions == 0:
        return 0.0
    return 100.0 * (exact_match_count) / total_questions

def answer_recall(prediction: str, ground_truths: list) -> float:
    """Compute recall as the maximum token overlap between prediction and any ground truth answer."""
    if not prediction or not ground_truths:
        return 0.0

    pred_tokens = Counter(get_tokens(prediction))  # Tokenize prediction
    num_pred_tokens = sum(pred_tokens.values())
    if num_pred_tokens == 0:
        return 0.0  # Avoid division by zero
    max_recall = 0.0
    for ground_truth in ground_truths:
        gt_tokens = Counter(get_tokens(ground_truth))
        num_gt_tokens = sum(gt_tokens.values())

        if num_gt_tokens == 0:
            continue  # Skip empty ground truths

        num_same = sum((pred_tokens & gt_tokens).values())

        if num_same == 0:
            continue  # No common words, recall remains 0

        recall = num_same / num_gt_tokens
        max_recall = max(max_recall, recall)

    return max_recall

def f1_score(prediction: str, ground_truths: list) -> float:
    """Compute max F1 score between a prediction and a set of ground truth answers."""
    if not prediction or not ground_truths:
        return 0.0

    pred_tokens = Counter(get_tokens(prediction))
    num_pred_tokens = sum(pred_tokens.values())

    if num_pred_tokens == 0:
        return 0.0  # Avoid division by zero

    max_f1 = 0.0

    for ground_truth in ground_truths:
        gt_tokens = Counter(get_tokens(ground_truth))
        num_gt_tokens = sum(gt_tokens.values())

        if num_gt_tokens == 0:
            continue  # Skip empty ground truths

        num_same = sum((pred_tokens & gt_tokens).values())  # Compute word overlap

        if num_same == 0:
            continue  # No common words, F1 remains 0

        precision = num_same / num_pred_tokens
        recall = num_same / num_gt_tokens

        current_f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        max_f1 = max(max_f1, current_f1)  # Take the best match
    return max_f1

def compute_recall_f1(gold_answers, generated_answers):
    total_f1 = 0
    total_recall = 0
    count = 0
    
    for qid in gold_answers:
        if qid not in generated_answers:
            continue
        gold_answer = gold_answers[qid]
        pred = generated_answers[qid][0] if generated_answers[qid] else ""
        
        recall, f1 = answer_recall(pred, gold_answer), f1_score(pred, gold_answer)
        total_f1 += f1
        total_recall += recall
        count += 1
    
    avg_f1 = 100 * total_f1 / count if count else 0.0
    avg_recall = 100 * total_recall / count if count else 0.0
    
    return avg_recall, avg_f1

def evaluate_system(references: dict, predictions: dict) -> dict:
    """Evaluate system performance using multiple metrics."""
    # Calculate exact match
    em_scores = exact_match(predictions, references)
    avg_recall, avg_f1 = compute_recall_f1(references, predictions)
    
    results = {
        "Exact Match": em_scores,
        "F1 Score": avg_f1,
        "Recall": avg_recall,
        "Count": len(references),
    }
    return results

def evaluate_by_category(references: dict, predictions: dict, categories: dict) -> dict:
    """
    Evaluate system performance grouped by categories.
    
    Args:
        references: Dictionary mapping question IDs to reference answers
        predictions: Dictionary mapping question IDs to system predictions
        categories: Dictionary mapping question IDs to categories
        
    Returns:
        Dictionary with evaluation results for each category and overall
    """
    # Group questions by category
    category_questions = defaultdict(dict)
    for qid in references:
        if qid in categories:
            category = categories[qid]
            category_questions[category][qid] = references[qid]
        else:
            # For questions without a category, put in "unknown" category
            category_questions["unknown"][qid] = references[qid]
    
    # Evaluate overall performance
    overall_results = evaluate_system(references, predictions)
    
    # Evaluate performance for each category
    category_results = {}
    for category, category_refs in category_questions.items():
        # Filter predictions for this category
        category_preds = {qid: predictions.get(qid, []) for qid in category_refs}
        category_results[category] = evaluate_system(category_refs, category_preds)
    
    return {
        "overall": overall_results,
        "by_category": category_results
    }

def main():
    parser = argparse.ArgumentParser(description="Compare RAG system outputs against reference answers with category breakdown")
    parser.add_argument("--references_dir", type=str, required=True, help="Directory containing reference answer JSON files")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory containing system output JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output comparison files")
    parser.add_argument("--csv_file", type=str, help="CSV file containing question categories (optional)")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate input directories
    if not os.path.exists(args.references_dir):
        print(f"References directory not found: {args.references_dir}")
        exit(1)
    if not os.path.exists(args.predictions_dir):
        print(f"Predictions directory not found: {args.predictions_dir}")
        exit(1)
    
    # Find reference and prediction files
    reference_files = [f for f in os.listdir(args.references_dir) if f.endswith('.json')]
    prediction_files = [f for f in os.listdir(args.predictions_dir) if f.endswith('.json')]
    
    if not reference_files:
        print(f"No reference JSON files found in {args.references_dir}")
        exit(1)
    if not prediction_files:
        print(f"No prediction JSON files found in {args.predictions_dir}")
        exit(1)
    
    print(f"Found {len(reference_files)} reference answer files")
    print(f"Found {len(prediction_files)} system output files")
    
    all_results = {}
    
    for ref_file in reference_files:
        ref_path = os.path.join(args.references_dir, ref_file)
        ref_name = os.path.splitext(ref_file)[0]
        
        print(f"\nEvaluating against reference: {ref_file}")
        
        try:
            # Load reference answers
            references = load_reference_answers(ref_path)
            if not references:
                print(f"  Skipping empty reference file: {ref_file}")
                continue
            
            # Try to extract categories from reference file first
            categories = extract_categories_from_json(ref_path)
            
            # If no categories found in reference file, try CSV if provided
            if not categories and args.csv_file and os.path.exists(args.csv_file):
                print(f"  Loading categories from {args.csv_file}")
                categories = load_categories(csv_file=args.csv_file)
            
            if categories:
                print(f"  Loaded {len(categories)} question categories")
                unique_categories = sorted(set(categories.values()))
                print(f"  Found {len(unique_categories)} unique categories: {', '.join(unique_categories)}")
            else:
                print("  No category information found. Will only compute overall metrics.")
            
            results = {}
            
            for pred_file in prediction_files:
                # Only compare files with matching names
                if pred_file != ref_file:
                    continue
                    
                pred_path = os.path.join(args.predictions_dir, pred_file)
                try:
                    predictions = load_system_predictions(pred_path)
                    
                    if not predictions:
                        print(f"  Skipping empty prediction file: {pred_file}")
                        continue
                    
                    # Evaluate with category breakdown if categories are available
                    if categories:
                        evaluation_results = evaluate_by_category(references, predictions, categories)
                        results[pred_file] = evaluation_results
                        
                        # Print overall results
                        overall = evaluation_results["overall"]
                        print(f"  {pred_file} (Overall - {overall['Count']} questions):")
                        print(f"    Exact Match: {overall['Exact Match']:.2f}%")
                        print(f"    F1 Score: {overall['F1 Score']:.2f}%")
                        print(f"    Recall: {overall['Recall']:.2f}%")
                        
                        # Print category results
                        print("  Results by category:")
                        for category, metrics in evaluation_results["by_category"].items():
                            print(f"    {category} ({metrics['Count']} questions):")
                            print(f"      Exact Match: {metrics['Exact Match']:.2f}%")
                            print(f"      F1 Score: {metrics['F1 Score']:.2f}%")
                            print(f"      Recall: {metrics['Recall']:.2f}%")
                    else:
                        # Basic evaluation without categories
                        system_results = evaluate_system(references, predictions)
                        results[pred_file] = {"overall": system_results}
                        
                        print(f"  {pred_file} ({system_results['Count']} questions):")
                        print(f"    Exact Match: {system_results['Exact Match']:.2f}%")
                        print(f"    F1 Score: {system_results['F1 Score']:.2f}%")
                        print(f"    Recall: {system_results['Recall']:.2f}%")
                        
                except Exception as e:
                    print(f"  Error evaluating {pred_file} against {ref_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            if not results:
                print(f"  No valid results for reference: {ref_file}")
                continue
            
            all_results[ref_name] = results
            
            # For each prediction file, generate only two files:
            # 1. One for overall results
            # 2. One for all category results
            for pred_file, pred_results in results.items():
                base_name = os.path.splitext(pred_file)[0]
                
                # Save overall results
                overall_file = os.path.join(args.output_dir, f"{base_name}_overall.json")
                with open(overall_file, 'w', encoding='utf-8') as f:
                    json.dump(pred_results["overall"], f, indent=2)
                print(f"  Overall results saved to {overall_file}")
                
                # Save all category results in a single file if available
                if "by_category" in pred_results:
                    categories_file = os.path.join(args.output_dir, f"{base_name}_categories.json")
                    with open(categories_file, 'w', encoding='utf-8') as f:
                        json.dump(pred_results["by_category"], f, indent=2)
                    print(f"  All category results saved to {categories_file}")
                        
        except Exception as e:
            print(f"Error processing reference file {ref_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\nNo valid evaluations were performed.")

if __name__ == "__main__":
    main()