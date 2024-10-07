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
import string
import shutil
from tqdm import tqdm
import argparse
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

class ResultRetriever:
    def __init__(self, model, queries_path, documents_path, outdir):
        self.model = model
        self.queries_path = queries_path
        self.documents_path = documents_path
        self.outdir = outdir
        self.queries = None
        self.documents = None
        self.index = None
        self.stop_words = set(stopwords.words('english'))

    def load_data(self):
        """
        Load queries and documents from JSON files into pandas DataFrames.
        """
        self.queries = pd.read_json(self.queries_path)
        self.documents = pd.read_json(self.documents_path)

    def preprocess_documents(self):
        """
        Preprocess documents to ensure they have the required fields for indexing.
        """
        # Rename 'Id' column to 'docno' and stringify the 'docno' value
        self.documents.rename(columns={'Id': 'docno'}, inplace=True)
        self.documents['docno'] = self.documents['docno'].astype(str)

        # Apply clean_string_html, remove_stopwords, remove_punctuation, and add_spaces_before_capitals to 'Text' column
        self.documents['body'] = self.documents['Text'].apply(self.clean_string_html).apply(self.remove_stopwords).apply(self.remove_punctuation).apply(self.add_spaces_before_capitals)

        # Drop 'Text' and 'Score' columns
        self.documents.drop(columns=['Text', 'Score'], inplace=True)

        # Remove rows where 'body' is empty
        self.documents = self.documents[self.documents['body'].str.strip().astype(bool)]

    def build_index(self):
        """
        Initialize PyTerrier and build an index from the documents.

        This method checks if PyTerrier is started, initializes it if not,
        creates an indexer, and indexes the documents using the 'body' field.
        If the index directory already exists, it is deleted and recreated.
        """
        if not pt.java.started():
            pt.java.init()
        
        index_dir = "./index"
        
        # Delete the index directory if it already exists
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            print(f"Deleted existing index directory: {index_dir}")

        # Create the index directory
        os.makedirs(index_dir, exist_ok=True)
        print(f"Created index directory: {index_dir}")

        # Ensure the directory has read and write permissions
        os.chmod(index_dir, 0o755)
        print(f"Set read and write permissions for index directory: {index_dir}")
        
        indexer = pt.DFIndexer(index_dir)
        print("Indexer created...")

        # Index the documents using the DataFrame directly
        self.index = indexer.index(self.documents["body"], self.documents["docno"])

    def clean_string_html(self, html_str):
        '''
        Given a string with html tags, return a string without the html tags.
        '''
        clean_text = re.sub(r'<[^>]+>', '', html_str)
        return clean_text

    def remove_stopwords(self, text):
        '''
        Remove stopwords from the input text.
        '''
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

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
        translator = str.maketrans('', '', string.punctuation)
        return input_string.translate(translator)

    def add_spaces_before_capitals(self, text):
        '''
        Add spaces before capital letters in the input text.
        '''
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

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

    def retrieve(self):
        '''
        This is the retrieval function. Based on the user's choice, it calls the corresponding pyterrier model.

        Returns: 
            List of tuples each containing trec compliant results.
        '''
        if self.model == 'tf-idf':
            retriever = pt.terrier.Retriever(self.index, wmodel="TF_IDF")
        elif self.model == 'bim':
            retriever = pt.terrier.Retriever(self.index, wmodel="BM25")
        elif self.model == 'bm25':
            retriever = pt.terrier.Retriever(self.index, wmodel="BM25")
        else:
            raise ValueError("Unsupported model: " + self.model)

        # Apply the process_query function to each row in the queries DataFrame
        processed_queries = self.queries.apply(self.process_query, axis=1)

        # Extract query IDs and texts
        query_ids = processed_queries.apply(lambda x: x[0])
        query_texts = processed_queries.apply(lambda x: x[1])

        results = []
        for query_id, query_text in zip(query_ids, query_texts):
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

    # Arguments for the user
    parser = argparse.ArgumentParser(description="Retrieve results from documents based on queries.")
    parser.add_argument('--model', default='tf-idf', help="The retrieval model to use (tf-idf, bim, bm25).")
    parser.add_argument('--queries', default='data/in/topics_1.json', help="Path to the queries JSON file.")
    parser.add_argument('--documents', default='data/in/Answers.json', help="Path to the documents JSON file.")
    parser.add_argument('--outdir', default='data/out/results', help="Directory to save the results.")

    args = parser.parse_args()

    # Define the result retriever, load data, preprocess documents, build index, and write results
    retriever = ResultRetriever(args.model, args.queries, args.documents, args.outdir)
    retriever.load_data()
    retriever.preprocess_documents()
    retriever.build_index()
    results = retriever.retrieve()
    retriever.save_results(results)