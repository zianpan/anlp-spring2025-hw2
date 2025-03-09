#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

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

def is_meaningful_answer(prediction: str, ground_truths: list, use_semantic: bool = False, semantic_threshold: float = 0.7) -> bool:
    def semantic_similarity(prediction: str, ground_truths: list, threshold: float = 0.7) -> bool:
        """
        Use a language model to compute semantic similarity between prediction and reference answers.
        Returns True if the prediction is semantically similar to any of the ground truths.
        """
        pred_embedding = MODEL.encode(prediction, convert_to_tensor=True)
        
        for ground_truth in ground_truths:
            gt_embedding = MODEL.encode(ground_truth, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_embedding, gt_embedding).item()
            if similarity >= threshold:
                return True
        
        return False
    
    if use_semantic:
        if semantic_similarity(prediction, ground_truths, semantic_threshold):
            return True
        else:
            return False
    print("semantic not used")
    return True

def exact_match(predictions_dict, references_dict):
    def exact_match_single(reference_list, prediction):
        """Check if prediction matches any reference answer."""
        if not prediction or not reference_list:
            return 0
        for reference in reference_list:
            if prediction == reference:
                return 1
        return 0
    
    total_questions = 0
    exact_match_count = 0
    for qid in references_dict:
        if qid not in predictions_dict:
            continue
        reference_list = references_dict[qid]
        prediction_list = predictions_dict[qid]
        
        if not reference_list or not prediction_list:
            continue
        prediction = prediction_list[0]
        total_questions += 1
        exact_match_count += exact_match_single(reference_list, prediction)
    if total_questions == 0:
        return 0.0
    return 100.0 * exact_match_count / total_questions

def answer_recall(prediction: str, ground_truths: list) -> float:
    if not prediction or not ground_truths:
        return 0.0
    
    def get_tokens(text: str) -> list:
        """Tokenize text by splitting on whitespace."""
        if not text or not isinstance(text, str):
            return []
        return text.lower().split()
    
    max_recall = 0.0
    pred_tokens = Counter(get_tokens(prediction))
    
    for ground_truth in ground_truths:
        gt_tokens = Counter(get_tokens(ground_truth))
        num_gt_tokens = sum(gt_tokens.values())
        
        if num_gt_tokens == 0:
            continue    
        
        num_same = sum((pred_tokens & gt_tokens).values())
        recall = num_same / num_gt_tokens
        max_recall = max(max_recall, recall)
    
    return max_recall

def f1_score(prediction: str, ground_truths: list) -> float:
    if not prediction or not ground_truths:
        return 0.0
    
    def get_tokens(text: str) -> list:
        """Tokenize text by splitting on whitespace."""
        if not text or not isinstance(text, str):
            return []
        return text.lower().split()
    
    max_f1 = 0.0
    pred_tokens = Counter(get_tokens(prediction))
    num_pred_tokens = sum(pred_tokens.values())
    
    if num_pred_tokens == 0:
        return 0.0
    
    for ground_truth in ground_truths:
        gt_tokens = Counter(get_tokens(ground_truth))
        num_gt_tokens = sum(gt_tokens.values())
        
        if num_gt_tokens == 0:
            continue
        num_same = sum((pred_tokens & gt_tokens).values())
        
        if num_same == 0:
            continue
            
        precision = num_same / num_pred_tokens
        recall = num_same / num_gt_tokens
        
        if precision + recall == 0:
            current_f1 = 0
        else:
            current_f1 = 2 * precision * recall / (precision + recall)
            
        max_f1 = max(max_f1, current_f1)
    
    return max_f1

def compute_recall_f1_single(gold_answer_list, generated_answer):
    return answer_recall(generated_answer, gold_answer_list), f1_score(generated_answer, gold_answer_list)

def compute_recall_f1(gold_answers, generated_answers):
    total_f1 = 0
    total_recall = 0
    
    for gold_answer_list, generated_answer in zip(gold_answers, generated_answers):
        recall, f1 = compute_recall_f1_single(gold_answer_list, generated_answer)
        total_f1 += f1
        total_recall += recall
    
    avg_f1 = 100 * total_f1 / len(gold_answers) if gold_answers else 0.0
    avg_recall = 100 * total_recall / len(gold_answers) if gold_answers else 0.0
    
    return avg_recall, avg_f1

def load_reference_answers(reference_file: str) -> dict:
    """
    Load a JSON file mapping question IDs to answers. The answers are
    processed using process_answer so that multiple correct answers (if
    separated by semicolons) are normalized into a list.
    """
    with open(reference_file, 'r', encoding='utf-8') as f:
        raw_answers = json.load(f)
    return {qid: process_answer(ans) for qid, ans in raw_answers.items() if ans and str(ans).strip()}

def load_system_predictions(prediction_file: str) -> dict:
    """
    Load system predictions from a JSON file. Predictions are trimmed so
    that extra whitespace is removed.
    """
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    return {qid: process_answer(pred) for qid, pred in predictions.items() if pred and str(pred).strip()}

def evaluate_system(references: dict, predictions: dict, use_semantic: bool = False, semantic_threshold: float = 0.7) -> dict:
    em_scores = 0
    f1_scores = []
    recall_scores = []
    missing_count = 0
    empty_reference_count = 0
    empty_prediction_count = 0
    not_meaningful_count = 0
    for qid in sorted(references.keys()):
        reference = references.get(qid, "")
        if not reference:
            empty_reference_count += 1
            continue
        if qid not in predictions:
            missing_count += 1
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
            
        if not is_meaningful_answer(prediction, reference, use_semantic=use_semantic, semantic_threshold=semantic_threshold):
            not_meaningful_count += 1
            continue
    
    # Calculate exact match separately    
    em_scores = exact_match(predictions, references)
    avg_recall, avg_f1 = compute_recall_f1(predictions, references)
    results = {
        "Exact Match": em_scores,
        "F1 Score": avg_f1,
        "Recall": avg_recall,
        "Missing Answers": missing_count,
        "Empty References": empty_reference_count,
        "Not Meaningful": not_meaningful_count,
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare multiple RAG system outputs against multiple reference answers")
    parser.add_argument("--references_dir", type=str, required=True, help="Directory containing reference answer JSON files")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory containing system output JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output comparison files")
    parser.add_argument("--semantic", action="store_true", help="Use semantic similarity to determine if an answer is meaningful")
    parser.add_argument("--semantic_threshold", type=float, default=0.7, help="Threshold for semantic similarity")
    
    args = parser.parse_args()
    use_semantic = args.semantic
    semantic_threshold = args.semantic_threshold
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
                    if not predictions:
                        print(f"  Skipping empty prediction file: {pred_file}")
                        continue
                    
                    # Patch the is_meaningful_answer to use current parameters
                    def evaluate_with_params(refs, preds):
                        global is_meaningful_answer
                        original_func = is_meaningful_answer
                        def patched_is_meaningful(pred, refs, 
                                                use_semantic=use_semantic, 
                                                semantic_threshold=semantic_threshold):
                            return original_func(pred, refs, use_semantic, semantic_threshold)
                        is_meaningful_answer = patched_is_meaningful
                        result = evaluate_system(refs, preds, use_semantic, semantic_threshold)
                        is_meaningful_answer = original_func
                        return result
                    
                    system_results = evaluate_with_params(references, predictions)
                    results[pred_file] = system_results
                    
                    print(f"  {pred_file}:")
                    print(f"    Exact Match: {system_results['Exact Match']:.2f}%")
                    print(f"    F1 Score: {system_results['F1 Score']:.2f}%")
                    print(f"    Recall: {system_results['Recall']:.2f}%")
                    print(f"    Missing Answers: {system_results['Missing Answers']}")
                    
                    if system_results['Empty References'] > 0:
                        print(f"    Empty References: {system_results['Empty References']}")
                    if system_results['Not Meaningful'] > 0:
                        print(f"    Not Meaningful Answers: {system_results['Not Meaningful']}")
                        
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
                    metrics_str += f", 'Missing': {metrics['Missing Answers']}"
                    metrics_str += f", 'Empty Refs': {metrics['Empty References']}"
                    metrics_str += f", 'Not Meaningful': {metrics['Not Meaningful']}"
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