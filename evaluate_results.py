'''
Wyatt McCurdy
2024-10-4
Information Retrieval Project Part 1
evaluate_results.py

This script evaluates results given by the ResultRetriever.

Sample Usage:

python evaluate_results.py --results data/out/results/results_bim_1.tsv --qrels data/in/qrel_1.tsv --outmetricsdir metrics --outchartdir figs

'''

import argparse
import pytrec_eval
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_results(results_path):
    """
    Load TREC-formatted results from a file.

    Args:
        results_path (str): Path to the TREC-formatted results file.

    Returns:
        dict: A dictionary where keys are query IDs and values are dictionaries
              mapping document IDs to their respective scores.
    """
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = float(score)
    return results

def load_qrels(qrels_path):
    """
    Load qrels from a file.

    Args:
        qrels_path (str): Path to the qrel file.

    Returns:
        dict: A dictionary where keys are query IDs and values are dictionaries
              mapping document IDs to their respective relevance scores.
    """
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)
    return qrels

def evaluate_results(results, qrels):
    """
    Evaluate the results using pytrec_eval.

    Args:
        results (dict): A dictionary of results where keys are query IDs and values
                        are dictionaries mapping document IDs to their respective scores.
        qrels (dict): A dictionary of qrels where keys are query IDs and values
                      are dictionaries mapping document IDs to their respective relevance scores.

    Returns:
        dict: A dictionary of mean evaluation metrics.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'P', 'map', 'bpref'})
    metrics = evaluator.evaluate(results)

    # Calculate mean metrics
    mean_metrics = {}
    for query_id, metric in metrics.items():
        for metric_name, value in metric.items():
            if metric_name not in mean_metrics:
                mean_metrics[metric_name] = []
            mean_metrics[metric_name].append(value)

    for metric_name in mean_metrics:
        mean_metrics[metric_name] = sum(mean_metrics[metric_name]) / len(mean_metrics[metric_name])

    return mean_metrics

def generate_ski_jump_plot(metrics, outchartdir, model, version):
    """
    Generate a ski-jump bar plot of P@5 for the top 100 results.

    Args:
        metrics (dict): A dictionary of evaluation metrics for each query.
        outchartdir (str): Directory to save the ski-jump plot.
        model (str): The model name.
        version (str): The version of the model.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(outchartdir, exist_ok=True)
    p_at_5 = {query_id: metric['P_5'] for query_id, metric in metrics.items()}
    sorted_p_at_5 = sorted(p_at_5.items(), key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_p_at_5)), [x[1] for x in sorted_p_at_5])
    plt.title('Ski-Jump Bar Plot of P@5')
    plt.xlabel('Query')
    plt.ylabel('P@5')
    plt.grid(True)
    plt.xticks([])  # Remove x-axis labels
    plt.savefig(os.path.join(outchartdir, f'ski_jump_plot_{model}_{version}.png'))
    plt.close()

def main():
    """
    Main function to evaluate TREC-formatted results using PyTrecEval.

    This function parses command-line arguments, loads the results and qrels,
    evaluates the results, saves the evaluation metrics, and generates a ski-jump plot.
    """
    parser = argparse.ArgumentParser(description="Evaluate TREC-formatted results using PyTrecEval.")
    parser.add_argument('--results', default="data/out/results/results_bim_1.tsv", help="Path to the TREC-formatted results file.")
    parser.add_argument('--qrels', default="data/in/qrel_1.tsv", help="Path to the qrel file.")
    parser.add_argument('--outmetricsdir', default="metrics", help="Directory to save the evaluation metrics.")
    parser.add_argument('--outchartdir', default="figs", help="Directory to save the ski-jump plot.")

    args = parser.parse_args()

    results = load_results(args.results)
    qrels = load_qrels(args.qrels)
    metrics = evaluate_results(results, qrels)

    # Get the name of the model and version
    results_name = os.path.basename(args.results).split('.')[0]
    model_name = results_name.split('_')[1]
    version = results_name.split('_')[2]

    os.makedirs(args.outmetricsdir, exist_ok=True)
    metrics_file = os.path.join(args.outmetricsdir, f'evaluation_metrics_{model_name}_{version}.txt')
    with open(metrics_file, 'w') as f:
        for metric_name, mean_value in metrics.items():
            f.write(f"{metric_name}: {mean_value}\n")

    generate_ski_jump_plot(metrics, args.outchartdir, model_name, version)

if __name__ == "__main__":
    main()