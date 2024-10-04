import argparse
import pytrec_eval
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_results(results_path):
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = float(score)
    return results

def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)
    return qrels

def evaluate_results(results, qrels):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'P', 'map', 'bpref'})
    metrics = evaluator.evaluate(results)
    return metrics

def generate_ski_jump_plot(metrics, outchartdir):
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