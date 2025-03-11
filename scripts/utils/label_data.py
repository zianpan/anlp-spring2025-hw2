import pandas as pd
from transformers import pipeline
import torch
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if device == "cuda" else -1
)

categories = ["General Info", "Events in Pittsburgh and CMU", "Music and Culture", "Sports"]

def classify_question(question):
    """Classifies a question into predefined categories."""
    if pd.isna(question) or not isinstance(question, str) or question.strip() == "":
        return "Others"

    try:
        result = classifier(question, categories, multi_label=False)
        return result["labels"][0]
    except Exception as e:
        print(f"Error processing question: {question}\nException: {e}")
        return "Others"

results_directory = "results/"
output_directory = "results/labeled/"
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(results_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_directory, filename)
        
        df = pd.read_csv(file_path)

        if "question" not in df.columns:
            print(f"Skipping {filename}: No 'question' column found.")
            continue

        df["category"] = df["question"].apply(classify_question)

        # Save labeled data
        labeled_file_path = os.path.join(output_directory, filename)
        df.to_csv(labeled_file_path, index=False)
        print(f"Labeled questions saved to {labeled_file_path}")
