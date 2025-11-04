import json
import numpy as np
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, mode="instruction"):
        assert mode in ("instruction", "code")
        self.bm25: BM25Okapi = None
        self.content_input_path: str = ""
        self.mode = mode
    
    def process(self, content_input_path: str):
        self.content_input_path = content_input_path
        with open(content_input_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        # to ensure the order
        self.chunks = []
        self.corpus = []
        for c in content:
            self.chunks.append(c["code"])
            self.corpus.append(c["description_1"])

        if self.mode == "instruction" and self.corpus:
            tokenized_corpus = [co.split(" ") for co in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        elif self.mode == "code" and self.chunks:
            tokenized_corpus = [co.split(" ") for co in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def query(
            self,
            query: str,
            top_k: int = 1
    ) -> List[Dict[str, Any]]:
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self.bm25 is None or not self.chunks:
            raise ValueError(
                "BM25 model is not initialized. Call `process` first."
            )
        
        # Preprocess query similarly to how documents were processed
        processed_query = query.split(" ")
        # Retrieve documents based on BM25 scores
        scores = self.bm25.get_scores(processed_query)

        top_k_indices = np.argpartition(scores, -top_k)[-top_k:]

        formatted_results = []
        for i in top_k_indices:
            result_dict = {
                    "similarity score": scores[i],
                    "original instruction": self.corpus[i],
                    "code": self.chunks[i]
            }
            formatted_results.append(result_dict)
        
        # Sort the list of dictionaries by 'similarity score' from high to low
        formatted_results.sort(
            key=lambda x: x['similarity score'], reverse=True
        )

        return formatted_results