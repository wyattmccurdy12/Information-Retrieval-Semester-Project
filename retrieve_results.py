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
from bs4 import BeautifulSoup
import ast
import string

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
        """
        Load queries and documents from JSON files.
        """
        with open(self.queries_path, 'r') as f:
            self.queries = json.load(f)
        

        with open(self.documents_path, 'r') as f:
            self.documents = json.load(f)
        
        # This is to make sure that PyTerrier recognizes the documents for indexing
        for d in self.documents:
            d['docno'] = d.pop('Id')
            d['body'] = d.pop('Text')

    def build_index(self):
        """
        Initialize PyTerrier and build an index from the documents.
    
        This method checks if PyTerrier is started, initializes it if not,
        creates an indexer, and indexes the documents using the 'Text' field.
        """
        if not pt.started():
            pt.init()
        indexer = pt.IterDictIndexer(self.outdir + "/index")
        self.index = indexer.index(self.documents, fields=['Text'])

    def clean_string_html(self, html_str):
        '''
        Given a string with html tags, return a string without the html tags.
        '''
        soup = BeautifulSoup(html_str, 'html.parser')
        return soup.get_text()

    def get_tags_str(self, tags_str):
        '''
        Given an input string that represents a python list, 
        return a string with just the tags separated by spaces.
        '''
        tags_list = ast.literal_eval(tags_str)

        return ' '.join(tags_list)

    def remove_punctuation(self, input_string):
        '''
        Remove punctuation from the input string and return an output string that is just words, spaces, and digits.
        '''
        return ''.join(char for char in input_string if char not in string.punctuation)


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
            query_body = self.clean_string_html(query_body)
            query_tags = query['Tags']
            query_tags = self.get_tags_str(query_tags)

            # The query text will just be a concatenation of title, body and tags as an unbroken string.
            query_text = f"{query_title} {query_body} {query_tags}"
            query_text = self.remove_punctuation(query_text)

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
    parser.add_argument('--model', default='tf-idf', help="The retrieval model to use (tf-idf, bim, bm25).")
    parser.add_argument('--queries', default='data/in/topics_1.json', help="Path to the queries JSON file.")
    parser.add_argument('--documents', default='data/in/Answers.json', help="Path to the documents JSON file.")
    parser.add_argument('--outdir', default='data/out/results', help="Directory to save the results.")

    args = parser.parse_args()

    # Define the result retriever, load data, and write results
    retriever = ResultRetriever(args.model, args.queries, args.documents, args.outdir)
    retriever.load_data()
    retriever.build_index()
    results = retriever.retrieve()
    retriever.save_results(results)