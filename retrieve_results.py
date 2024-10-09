'''
Wyatt McCurdy
2024-10-4
Information Retrieval Project Part 1
retrieve_results.py

This script implements the ResultRetriever class, which allows a user to select a model from three options: TF-IDF, BIM, or BM25. 
These are three traditional statistical models for information retrieval. 
'''

import os
import pyterrier as pt
import ast
import argparse
import re
import pandas as pd
import string


class ResultRetriever:
    def __init__(self, queries_path, documents_path, qrels_path, outdir):
        self.queries_path = queries_path
        self.documents_path = documents_path
        self.qrels_path = qrels_path
        self.outdir = outdir
        self.queries = None
        self.documents = None
        self.qrels = None
        self.index = None

    def load_data(self):
        """
        Load queries, documents, and qrels from JSON files into pandas DataFrames.
        """
        self.queries = pd.read_json(self.queries_path)
        self.documents = pd.read_json(self.documents_path)
        self.qrels = pd.read_csv(self.qrels_path, sep='\t', header=None)
        self.qrels.columns = ['qid', 'q0', 'docno', 'relevance']

    def preprocess_documents(self):
        """
        Preprocess documents to ensure they have the required fields for indexing.

        Clean the body field by removing html tags.
        """
        # Rename 'Id' column to 'docno' and stringify the 'docno' value
        self.documents.rename(columns={'Id': 'docno'}, inplace=True)
        self.documents['docno'] = self.documents['docno'].astype(str)

        # Apply clean_string_html
        self.documents['text'] = self.documents['Text'].apply(self.clean_string_html)

        # Remove punctuation
        self.documents['text'] = self.documents['text'].apply(self.remove_punctuation)

        # Drop 'Text' and 'Score' columns
        self.documents.drop(columns=['Text', 'Score'], inplace=True)

        # Save 100 rows of the dataframe to a tsv file
        self.documents.head(100).to_csv('sample_docs.tsv', sep='\t', index=False)

    def preprocess_queries(self):
        """
        Preprocess queries to ensure they have the required fields for retrieval.
    
        Clean the body field by removing html tags and punctuation.
        Clean the title field by removing punctuation.
        Clean the tags field by literally evaluating the stringified list and joining the tags.
        """
        # Preprocess title
        self.queries['Title'] = self.queries['Title'].apply(self.remove_punctuation)

        # Preprocess tags
        self.queries['Tags'] = self.queries['Tags'].apply(self.get_tags_str)

        # Rename 'Id' column to 'qid' and stringify the 'qid' value
        self.queries.rename(columns={'Id': 'qid'}, inplace=True)
        self.queries['qid'] = self.queries['qid'].astype(str)
    
        # Apply clean_string_html
        self.queries['query'] = self.queries['Body'].apply(self.clean_string_html)
    
        # Remove punctuation
        self.queries['query'] = self.queries['query'].apply(self.remove_punctuation)

        # Add title and tags to the query text
        self.queries['query'] = self.queries['Title'] + ' '  + self.queries['query'] + ' '  + self.queries['Tags']
    
        # Drop 'Body' column
        self.queries.drop(columns=['Body'], inplace=True)
    
        # Save 100 rows of the dataframe to a tsv file
        self.queries.head(100).to_csv('sample_queries.tsv', sep='\t', index=False)
    
    def preprocess_qrels(self):
        """
        Preprocess qrels to ensure they have the required fields for evaluation.
        """

        self.qrels['qid'] = self.qrels['qid'].astype(str)
        self.qrels['docno'] = self.qrels['docno'].astype(str)

    def remove_punctuation(self, text):
        """
        Replace punctuation in the given text with spaces.
        """
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return text.translate(translation_table)
    
    def build_index(self):
        """
        Initialize PyTerrier and build an index from the documents if it doesn't exist.
        Otherwise, load the existing index.

        This method checks if PyTerrier is started, initializes it if not,
        creates an indexer, and indexes the documents using the 'text' field.
        If the index directory already exists, it is loaded instead of being recreated.
        """
        if not pt.java.started():
            pt.java.init()
        
        index_dir = "./index"
        
        if os.path.exists(index_dir):
            print(f"Loading existing index from: {index_dir}")
            self.index = pt.IndexFactory.of(index_dir)
        else:
            print(f"Creating new index at: {index_dir}")
            os.makedirs(index_dir, exist_ok=True)
            print(f"Created index directory: {index_dir}")

            # Ensure the directory has read and write permissions
            os.chmod(index_dir, 0o755)
            print(f"Set read and write permissions for index directory: {index_dir}")
            
            indexer = pt.IterDictIndexer(index_dir, meta={'docno': 20, 'text': 4096}, stemmer='porter', stopwords='terrier')

            # Index the documents using the DataFrame directly
            self.index = indexer.index(self.documents.to_dict(orient='records'))

    def clean_string_html(self, html_str):
        '''
        Given a string with html tags, return a string without the html tags.
        '''
        clean_text = re.sub(r'<[^>]+>', '', html_str)
        return clean_text

    def get_tags_str(self, tags_str):
        '''
        Given an input string that represents a python list, 
        return a string with just the tags separated by spaces.
        '''
        tags_list = ast.literal_eval(tags_str)
        return ' '.join(tags_list)

    def process_query(self, query):
        query_id = query['Id']
        query_title = query['Title']
        query_body = self.clean_string_html(query['Body'])
        query_tags = self.get_tags_str(query['Tags'])
        
        # The query text will just be a concatenation of title, body and tags as an unbroken string.
        query_text = f"{query_title} {query_body} {query_tags}"
        query_text = self.remove_stopwords(query_text)
        query_text = self.remove_punctuation(query_text)
        query_text = self.add_spaces_before_capitals(query_text)
        
        return query_id, query_text


if __name__ == "__main__":

    # Arguments for the user
    parser = argparse.ArgumentParser(description="Retrieve results from documents based on queries.")

    parser.add_argument('--queries', default='data/in/topics_1.json', help="Path to the queries JSON file.")
    parser.add_argument('--documents', default='data/in/Answers.json', help="Path to the documents JSON file.")
    parser.add_argument('--qrels', default='data/in/qrel_1.tsv')
    parser.add_argument('--outdir', default='data/out/results', help="Directory to save the results.")

    args = parser.parse_args()

    # Define the result retriever, load data, preprocess documents, build index, and write results 
    rr = ResultRetriever(args.queries, args.documents, args.qrels, args.outdir)
    rr.load_data()
    print("Loaded data")
    rr.preprocess_documents()
    print()
    print("Preprocessed documents")
    print()
    rr.preprocess_queries()
    print()
    print("Preprocessed queries")
    print()
    rr.preprocess_qrels()
    print()
    print("Preprocessed qrels")
    print()
    rr.build_index()
    print()
    print("Built or loaded index")
    print()

    # After building the index, we enter a new phase - retrieval.
    tf_idf = pt.terrier.Retriever(rr.index, wmodel='TF_IDF')
    bm25 = pt.terrier.Retriever(rr.index, wmodel='BM25')

    print("Running experiment...")
    exp_sig = pt.Experiment(
        [tf_idf, bm25], 
        rr.queries, 
        rr.qrels, 
        eval_metrics=["map", "ndcg", "recip_rank", "ndcg_cut_5", "ndcg_cut_10", "P.5", "P.10", "P.1000", "bpref"],
        save_dir=args.outdir, 
        verbose=True, 
        baseline=0
    )

    exp_full = pt.Experiment(
        [tf_idf, bm25],
        rr.queries,
        rr.qrels,
        eval_metrics=["map", "ndcg", "recip_rank", "ndcg_cut_5", "ndcg_cut_10", "P.5", "P.10", "P.1000", "bpref"],
        save_dir='data/out/results/full', 
        verbose=True, 
        perquery=True
    )

    # Save the results dataframe to a CSV file
    exp_sig.to_csv('sig_results.csv')
    exp_full.to_csv('full_results.csv')