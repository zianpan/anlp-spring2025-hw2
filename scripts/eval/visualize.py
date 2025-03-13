import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def get_model_name(filename):
    if 'Qwen' in filename:
        return 'Qwen2.5'
    elif 'Llama' in filename:
        return 'Llama3.1'
    elif 'no_rag' in filename:
        return 'Baseline'
    else:
        return 'Other'

def create_accuracy_graph(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' contains invalid JSON.")
        return

    categories = list(data.keys())
    metrics = ['Exact Match', 'F1 Score', 'Recall']     

    metric_values = {metric: [] for metric in metrics}

    for category in categories:
        for metric in metrics:
            metric_values[metric].append(data[category][metric])

    plt.style.use('grayscale')
    fig, ax = plt.subplots(figsize=(12, 8))
    xticks = ['General', 'Events', 'Music/Culture', 'Sports']
    x = np.arange(len(categories))
    width = 0.25 

    patterns = ['', '', '']
    grays = [0.3, 0.5, 0.7]

    for i, metric in enumerate(metrics):
        position = x + (i - 1) * width
        bars = ax.bar(position, metric_values[metric], width, label=metric, 
                     color=f'{grays[i]}', hatch=patterns[i], edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)

    ax.set_ylabel('Score (%)', fontsize=16, fontweight='bold')
    name = get_model_name(filename)
    ax.set_title(f'Accuracy Metrics by Category ({name})', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    fig.tight_layout()
    
    os.makedirs('visualize', exist_ok=True)
    
    base_filename = os.path.basename(filename)
    output_path = f'visualize/{os.path.splitext(base_filename)[0]}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    
    print(f"Successfully generated bar graph from '{filename}'")
    print(f"Image saved to '{output_path}'")

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("Enter the JSON filename: ")
    
    create_accuracy_graph(filename)

if __name__ == "__main__":
    main()