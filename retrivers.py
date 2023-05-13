import numpy as np
import jsonlines
import torch
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
from rank_bm25 import BM25Okapi


class RetrieverBM25:

    def __init__(self, qa_path="TouhouQANet.jsonl"):

        edges = []

        for item in jsonlines.open(qa_path):
            for edge in item['output']:
                edges.append(edge)

        questions = [edge["question"] for edge in edges]

        tokenized_corpus = [list(question) for question in questions]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, topk=5):

        tokenized_query = list(query)
        doc_scores = bm25.get_scores(tokenized_query)
        indices = np.argsort(-doc_scores)[:topk]

        return [edges[index] for index in indices]


class RetrieverSBERT:

    def __init__(self, qa_path="TouhouQANet.jsonl", model_path="uer/sbert-base-chinese-nli"):

        edges = []

        for item in jsonlines.open(qa_path):
            for edge in item['output']:
                edges.append(edge)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = BertModel.from_pretrained(model_path).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        self.edges = edges

        self.questions = [edge["question"] for edge in edges]

        vecs = []

        bs = 64

        with torch.no_grad():
            for idx in range(0, len(questions), bs):
                batch = self.questions[idx:idx + bs]
                inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
                vecs.append(model(**inputs)['pooler_output'])

        self.dense = torch.cat(vecs, 0)

    def retrieve(self, query, topk=5):

        inputs = tokenizer([query], return_tensors="pt", padding=True).to(device)
        vec = model(**inputs)['pooler_output']
        scores = cosine_similarity(vec, self.dense)
        indices = scores.topk(5).indices
        return [edges[index] for index in indices]

if __name__ == "__main__":
    query = "古明地恋为什么闭上了第三只眼？"
    retriever = RetrieverBM25(qa_path="TouhouQANet.jsonl")
    retrieved = retriever.retrieve(query)
    print("****BM25****")
    print(retrieved)
    
    retriever = RetrieverSBERT(qa_path="TouhouQANet.jsonl")
    retrieved = retriever.retrieve(query)
    print("****SBERT****")
    print(retrieved)
