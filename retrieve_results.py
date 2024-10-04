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
            query_id = query['Id']
            query_text = query['Title'] + " " + query['Body']
            res = retriever.search(query_text)
            for rank, row in enumerate(res.itertuples()):
                results.append((query_id, "Q0", row.docno, rank + 1, row.score, self.model))
        return results

    def save_results(self, results):
        os.makedirs(self.outdir, exist_ok=True)
        output_file = os.path.join(self.outdir, f"results_{self.model}.tsv")
        with open(output_file, 'w') as f:
            for result in results:
                f.write("\t".join(map(str, result)) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve results from documents based on queries.")
    parser.add_argument('--model', required=True, help="The retrieval model to use (tf-idf, bim, bm25).")
    parser.add_argument('--queries', required=True, help="Path to the queries JSON file.")
    parser.add_argument('--documents', required=True, help="Path to the documents JSON file.")
    parser.add_argument('--outdir', required=True, help="Directory to save the results.")

    args = parser.parse_args()

    retriever = ResultRetriever(args.model, args.queries, args.documents, args.outdir)
    retriever.load_data()
    retriever.build_index()
    results = retriever.retrieve()
    retriever.save_results(results)