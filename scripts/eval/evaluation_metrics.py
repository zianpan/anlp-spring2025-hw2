#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import string

# =======================================load and process answers======================================
def process_answer(answer: str) -> list:
    """
    Given an answer string, split it on semicolons (if present),
    strip whitespace, and return a list of clean answers.
    """
    if not answer or not answer.strip():
        return []
    # if ";" in answer:
    #     return [a.strip() for a in answer.split(";") if a.strip()]
    return [answer.strip()]

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
    Load a JSON file mapping question IDs to answers. The answers are
    processed using process_answer so that multiple correct answers (if
    separated by semicolons) are normalized into a list.
    """
    with open(reference_file, 'r', encoding='utf-8') as f:
        raw_answers = json.load(f)
    return {qid: process_answer(ans) for qid, ans in raw_answers.items()}

def load_system_predictions(prediction_file: str) -> dict:
    """
    Load system predictions from a JSON file. Predictions are trimmed so
    that extra whitespace is removed.
    """
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    return {qid: process_answer(pred) for qid, pred in predictions.items()}

# =======================================evaluation metrics======================================
def exact_match(predictions_dict, references_dict):
    def exact_match_single(reference_list, prediction):
        """Check if normalized prediction matches any normalized reference answer."""
        if not prediction or not reference_list:
            return 0
        for reference in reference_list:
            if prediction == reference:
                return 1
        return 0

    total_questions = 0
    exact_match_count = 0
    for qid in references_dict:
        reference_list = references_dict[qid]
        prediction_list = predictions_dict[qid]
        
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
    em_scores = 0
    empty_reference_count = 0
    empty_prediction_count = 0
    for qid in sorted(references.keys()):
        reference = references.get(qid, "")
        if not reference:
            empty_reference_count += 1
            continue
        prediction = predictions[qid]
        if not prediction:
            empty_prediction_count += 1
            continue
            
        # Use first prediction if we have a list
        if isinstance(prediction, list) and prediction:
            prediction = prediction[0]
        
        if not isinstance(prediction, str) or not prediction.strip():
            empty_prediction_count += 1
            continue
            
    
    # Calculate exact match separately    
    em_scores = exact_match(predictions, references)
    avg_recall, avg_f1 = compute_recall_f1(predictions, references)
    results = {
        "Exact Match": em_scores,
        "F1 Score": avg_f1,
        "Recall": avg_recall,
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare multiple RAG system outputs against multiple reference answers")
    parser.add_argument("--references_dir", type=str, required=True, help="Directory containing reference answer JSON files")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory containing system output JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output comparison files")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.references_dir, exist_ok=True)
        
    reference_files = [f for f in os.listdir(args.references_dir) if f.endswith('.json')]
    if not reference_files:
        print(f"No reference JSON files found in {args.references_dir}")
        exit(1)
    if not os.path.exists(args.predictions_dir):
        print(f"No predictions directory found: {args.predictions_dir}")
        exit(1)
    prediction_files = [f for f in os.listdir(args.predictions_dir) if f.endswith('.json')]
    if not prediction_files:
        print(f"No prediction JSON files found in {args.predictions_dir}")
        exit(1)
    
    reference_files = [f for f in os.listdir(args.references_dir) if f.endswith('.json')]
    if not reference_files:
        print(f"No reference JSON files found in {args.references_dir}")
        exit(1)
    
    print(f"Found {len(reference_files)} reference answer files")
    print(f"Found {len(prediction_files)} system output files")
    
    all_results = {}
    
    for ref_file in reference_files:
        ref_path = os.path.join(args.references_dir, ref_file)
        ref_name = os.path.splitext(ref_file)[0]
        
        print(f"\nEvaluating against reference: {ref_file}")
        
        try:
            references = load_reference_answers(ref_path)
            if not references:
                print(f"  Skipping empty reference file: {ref_file}")
                continue
            
            results = {}
            
            for pred_file in prediction_files:
                pred_path = os.path.join(args.predictions_dir, pred_file)
                try:
                    predictions = load_system_predictions(pred_path)
                    
                    # if len(predictions) != len(references):
                    #     print(f"  Skipping {pred_file} - different lengths: {len(predictions)} vs {len(references)}")
                    #     continue
                    if pred_file != ref_file:
                        print(f"  Skipping {pred_file} - different files: {pred_file} vs {ref_file}")
                        continue
                    if not predictions:
                        print(f"  Skipping empty prediction file: {pred_file}")
                        continue
                    
                    def evaluate_with_params(refs, preds):
                        result = evaluate_system(refs, preds)
                        return result
                    
                    system_results = evaluate_with_params(references, predictions)
                    results[pred_file] = system_results
                    
                    print(f"  {pred_file}:")
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
            
            # Only try to sort if we have valid results
            if any(r['F1 Score'] > 0 for _, r in results.items()):
                sorted_results = sorted(results.items(), key=lambda x: x[1]['F1 Score'], reverse=True)
            else:
                sorted_results = list(results.items())
            
            output_file = os.path.join(args.output_dir, f"comparison_{ref_name}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to {output_file}")
            
            print(f"\n  Systems ranked by F1 Score against {ref_file}:")
            for i, (pred_file, metrics) in enumerate(sorted_results, 1):
                print(f"  {i}. {pred_file}: {metrics['F1 Score']:.2f}%")
                
        except Exception as e:
            print(f"Error processing reference file {ref_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        summary_file = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# RAG System Evaluation Summary\n\n")
            for ref_name, results in all_results.items():
                f.write(f"## Reference: {ref_name}\n\n")
                # Only try to sort if we have valid results
                if any(r['F1 Score'] > 0 for _, r in results.items()):
                    sorted_results = sorted(results.items(), key=lambda x: x[1]['F1 Score'], reverse=True)
                else:
                    sorted_results = list(results.items())
                for pred_file, metrics in sorted_results:
                    main_metrics = ["Exact Match", "F1 Score", "Recall"]
                    metrics_str = ", ".join([f"'{k}': {metrics[k]:.2f}" for k in main_metrics])
                    f.write(f"{{{metrics_str}}} {pred_file}\n")
                f.write("\n")
        
        print(f"\nSummary of all comparisons saved to {summary_file}")
    else:
        summary_file = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# RAG System Evaluation Summary\n\n")
            f.write("No valid evaluations were performed. Please check your input files.\n")
        
        print(f"\nNo valid evaluations were performed. Empty summary saved to {summary_file}")

if __name__ == "__main__":
    main()