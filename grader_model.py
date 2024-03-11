import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class MyGrader:

    def __init__(self,url,from_ = 'sb'):
        self.url = url
        self.from_ = from_
        if self.from_ == 'tf':
            self.model = hub.load(url)
        elif self.from_ == 'sb':
            self.model = SentenceTransformer(self.url)
            self.model.max_seq_length = 500
        elif self.from_ == 'hf':
            self.model = AutoModel.from_pretrained(url)
            self.tokenizer = AutoTokenizer.from_pretrained(url)
            pass

    def get_embeddings(self,text):
        if self.from_ == 'tf':
            return self.model(text)
        elif self.from_ == 'sb':
            return self.model.encode(text)
        elif self.from_ == 'hf':
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            return output
    def get_cosines(self,student_answer,key_answer):
        answer_embeddings = self.get_embeddings(student_answer)
        key_embeddings = self.get_embeddings(key_answer)

        cosine_scores = cosine_similarity(answer_embeddings, key_embeddings)
        return cosine_scores

