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
from pyterrier import Experiment

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

    mean_metrics_df = pd.DataFrame(metrics).T.mean()
    return mean_metrics_df

def generate_ski_jump_plot(metrics, outchartdir, model, version):
    """
    Generate a ski-jump bar plot of P@5 for the top 100 results.

    Args:
        metrics (dict): A dictionary of evaluation metrics for each query.
        outchartdir (str): Directory to save the ski-jump plot.
        model (str): The model name.
        version (str): The version of the model.
    """
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
    parser.add_argument('--resultsdir', default="data/out/results", help="Path to the TREC-formatted results file.")
    parser.add_argument('--qrels', default="data/in/qrel_1.tsv", help="Path to the qrel file.")
    parser.add_argument('--outmetricsdir', default="metrics", help="Directory to save the evaluation metrics.")
    parser.add_argument('--outchartdir', default="figs", help="Directory to save the ski-jump plot.")

    args = parser.parse_args()

    qrel_df = pd.read_csv(args.qrels, sep="\t", header=None)
    qrel_df.columns = ["qid", "Q0", "docno", "label"]
    topics_df = pd.read_json("data/in/topics_1.json")
    topics_df.columns = ["qid", "query"]

    # Get the name and version of the qrels file, so that we know which results we are comparing
    qrel_name = os.path.basename(args.qrels)
    qrel_version = qrel_name.split("_")[1].split(".")[0]
    
    results_dfs = []
    # Load dataframes for each results file
    for r in os.listdir(args.resultsdir):
        if r.endswith(".tsv"): 
            short_resultsname = r.split(".")[0]
            if short_resultsname.endswith(str(qrel_version)):
                results_dfs.append(pd.read_csv(os.path.join(args.resultsdir, r), sep="\t", header=None))

    # Now that we have dataframes, let's run pyterrier experiment
    exp = Experiment(retr_systems=results_dfs, topics=topics_df,
                     qrels=qrel_df, eval_metrics=["map", "P.1", "P.5", "P.10", "ndcg"])

    print()



if __name__ == "__main__":
    main()