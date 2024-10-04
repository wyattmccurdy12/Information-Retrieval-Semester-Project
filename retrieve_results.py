'''
Wyatt McCurdy
2024-10-4
Information Retrieval Project Part 1
retrieve_results.py

This script implements the ResultRetriever class, which allows a user to select a model from three options: TF-IDF, BIM, or BM25. 
These are three traditional statistical models for information retrieval. 
'''

import json
import os
import pyterrier as pt

class ResultRetriever:
    def __init__(self, model, queries_path, documents_path, outdir):
        self.model = model
        self.queries_path = queries_path
        self.documents_path = documents_path
        self.outdir = outdir
        self.queries = None
        self.documents = None
        self.index = None

    def load_data(self):
        with open(self.queries_path, 'r') as f:
            self.queries = json.load(f)
        with open(self.documents_path, 'r') as f:
            self.documents = json.load(f)

    def build_index(self):
        if not pt.started():
            pt.init()
        indexer = pt.IterDictIndexer(self.outdir + "/index")
        self.index = indexer.index(self.documents, fields=['Text'])

    def retrieve(self):
        '''
        This is the retrieval function. Based on the user's choice, it calls the corresponding pyterrier model.

        Returns: 
            List of tuples each containing trec compliant results.
        '''
        if self.model == 'tf-idf':
            retriever = pt.BatchRetrieve(self.index, wmodel="TF_IDF")
        elif self.model == 'bim':
            retriever = pt.BatchRetrieve(self.index, wmodel="BM25")
        elif self.model == 'bm25':
            retriever = pt.BatchRetrieve(self.index, wmodel="BM25")
        else:
            raise ValueError("Unsupported model: " + self.model)

        results = []
        for query in self.queries:

            # Get and clean id, title, and body.
            query_id = query['Id']
            query_title = query['Title']
            query_body = query['Body']
            query_body = clean_string_html(query_body)

            query_text = query['Title'] + " " + query['Body']
            res = retriever.search(query_text)
            for rank, row in enumerate(res.itertuples()):
                results.append((query_id, "Q0", row.docno, rank + 1, row.score, self.model))
        return results

    def save_results(self, results):
        '''
        Save results from selected model. 

        Args: 
            results: a list of strings, one for each line of a trec-formatted tsv
        '''
        os.makedirs(self.outdir, exist_ok=True)

        # Get topic number 
        topic_num_str = os.path.basename(self.queries_path)
        topic_num_str = topic_num_str.split("_")[1]
        topic_num_str = topic_num_str.split(".")[0]

        output_file = os.path.join(self.outdir, f"results_{self.model}_{topic_num_str}.tsv")

        # Open output file and write results
        with open(output_file, 'w') as f:
            for result in results:
                f.write("\t".join(map(str, result)) + "\n")

if __name__ == "__main__":
    import argparse

    # Arguments for the user
    parser = argparse.ArgumentParser(description="Retrieve results from documents based on queries.")
    parser.add_argument('--model', required=True, help="The retrieval model to use (tf-idf, bim, bm25).")
    parser.add_argument('--queries', required=True, help="Path to the queries JSON file.")
    parser.add_argument('--documents', required=True, help="Path to the documents JSON file.")
    parser.add_argument('--outdir', required=True, help="Directory to save the results.")

    args = parser.parse_args()

    # Define the result retriever, load data, and write results
    retriever = ResultRetriever(args.model, args.queries, args.documents, args.outdir)
    retriever.load_data()
    retriever.build_index()
    results = retriever.retrieve()
    retriever.save_results(results)