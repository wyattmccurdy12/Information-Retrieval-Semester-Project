'''
Wyatt McCurdy
2024-10-4
Information Retrieval Project Part 1
evaluate_results.py

This script evaluates results given by the ResultRetriever.
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
        dict: A dictionary of evaluation metrics for each query.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'P', 'map', 'bpref'})
    metrics = evaluator.evaluate(results)
    return metrics

def generate_ski_jump_plot(metrics, outchartdir):
    """
    Generate a ski-jump plot of P@5 for the top 100 results.

    Args:
        metrics (dict): A dictionary of evaluation metrics for each query.
        outchartdir (str): Directory to save the ski-jump plot.
    """
    os.makedirs(outchartdir, exist_ok=True)
    p_at_5 = {query_id: metric['P_5'] for query_id, metric in metrics.items()}
    sorted_p_at_5 = sorted(p_at_5.items(), key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot([x[1] for x in sorted_p_at_5], marker='o')
    plt.title('Ski-Jump Plot of P@5')
    plt.xlabel('Query')
    plt.ylabel('P@5')
    plt.grid(True)
    plt.savefig(os.path.join(outchartdir, 'ski_jump_plot.png'))
    plt.close()

def main():
    """
    Main function to evaluate TREC-formatted results using PyTrecEval.

    This function parses command-line arguments, loads the results and qrels,
    evaluates the results, saves the evaluation metrics, and generates a ski-jump plot.
    """
    parser = argparse.ArgumentParser(description="Evaluate TREC-formatted results using PyTrecEval.")
    parser.add_argument('--results', required=True, help="Path to the TREC-formatted results file.")
    parser.add_argument('--qrels', required=True, help="Path to the qrel file.")
    parser.add_argument('--outmetricsdir', required=True, help="Directory to save the evaluation metrics.")
    parser.add_argument('--outchartdir', required=True, help="Directory to save the ski-jump plot.")

    args = parser.parse_args()

    results = load_results(args.results)
    qrels = load_qrels(args.qrels)
    metrics = evaluate_results(results, qrels)

    os.makedirs(args.outmetricsdir, exist_ok=True)
    metrics_file = os.path.join(args.outmetricsdir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        for query_id, metric in metrics.items():
            f.write(f"{query_id}: {metric}\n")

    generate_ski_jump_plot(metrics, args.outchartdir)

if __name__ == "__main__":
    main()