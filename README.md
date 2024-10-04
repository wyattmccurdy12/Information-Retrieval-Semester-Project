# Information Retrieval Project Part 1

## Goal
This project is meant to explore traditional information retrieval models with PyTerrier. In the first section of the project, PyTerrier will be used to implement three classic information retrieval models - TF-IDF, BIM, and BM25. The data will be from the classic AskUbuntu site, with topics json files (queries) and an answers json file (documents). A qrel file will be given for the first topics file, and the second qrel will be reserved for instructor testing. In order to evaluate effectiveness, nDCG@k, P@k (k={5, 10, all}), mAP and bpref will be used. Significance testing will then be applied between models in order to determine which models performed best and worst on the AskUbuntu task. 

## Files
### Code
Two python files will be used: 
1. `retrieve_results.py`: A program which will retrieve the results from the AskUbuntu answers files based on input from topics files. The results will be in the form of a trec-formatted tsv file with the top 100 ranked documents per query. 

2. `evaluate_results.py`: A program which will evaluate the trec-formatted results using PyTrecEval. This script will return the metrics listed in the above section, as well as generate ski-jump plots using P@5 for the top 100 results. 

### Data
1. `topics_1.json`
2. `topics_2.json`
The above two files are json file containing queries with keys "Id": str (representing int), "Title": str, "Body": str (html formatted), "Tags": str (representing list).
3. `Answers.json`: A json file containing keys "Id": str (representing int), "Text": str (html formatted), "Score": str (representing int - number of AskUbuntu community votes)
4. `qrel_1.tsv`: A trec-formatted qrel file with query ids from the topics files and document ids from the answers file. 


## Sample command line command
### Retrieve results
```bash 
python retrieve_results.py --model tf-idf --queries data/in/topics_1.json --documents data/in/Answers.json --outdir data/out/results/
```

### Evaluate results
```bash 
python evaluate_results.py --results data/out/results/results_tfidf_1.tsv --qrels qrel_1.tsv --outmetricsdir data/out/evaluation/ --outchartdir figs/
```

