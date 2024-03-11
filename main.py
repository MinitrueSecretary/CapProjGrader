import numpy as np
import pandas as pd

from transformers import AutoModel, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer,util

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from grader_model import MyGrader

from time import time

import sys

def choose_position(embeddings,indices=None,k=20):
    if indices is None:
        indices = np.arange(len(embeddings))
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(embeddings)
    labels = kmeans.labels_
    centres = kmeans.cluster_centers_

    cos_sim = cosine_similarity(embeddings,centres)
    max_cs = np.argmax(cos_sim,axis=0)
    return indices[max_cs]

def main():
    # csv file 
    filename = sys.argv[1]
    question = sys.argv[2]

    # Load the data
    data = pd.read_csv(filename)
    student_ids = data['Student ID']
    student_answers = data[f'Qstn # {question}']
    num_students = len(student_ids)
    # Create a grader object
    print("Loading the grader model...")
    grader = MyGrader('paraphrase-multilingual-mpnet-base-v2',from_='sb')
    
    print("Calculating the embeddings...")
    embeddings = grader.get_embeddings(student_answers)
    emb_to_check = embeddings.copy()
    thresh = float(input("Enter the threshold (should be around 0.5-0.9): "))
    indices = np.arange(num_students)

    checked = np.zeros((num_students,),bool)
    pred_scores = np.zeros((num_students,),float)

    n_checked = 0
    is_first_time = True

    while (~checked).sum() > 0:
        k = 5 if is_first_time else 2
        is_first_time = False
        if (~checked).sum() <= 5:
            # check all remaining answers by hand
            for i in range(num_students):
                if not checked[i]:
                    print(f"Answer {i}: {student_answers[i]}")
                    pred_scores[i] = float(input("Enter the score: "))
                    checked[i] = True
            break
        # select the k most representative answers
        sel_indices = choose_position(emb_to_check,indices,k)
        for i in sel_indices:
            print(f"Answer {i}: {student_answers[i]}")
            pred_scores[i] = float(input("Enter the score: "))
            checked[i] = True
        # cand_score = df['total_score'].to_numpy()[sel_indices]
        cand_score = pred_scores[sel_indices]

        # get the embeddings of the selected answers
        sel_emb = embeddings[sel_indices]

        # get the cosine similarity between the selected answers and the remaining answers
        cos_sim = cosine_similarity(embeddings,sel_emb)

        closest = cos_sim.argmax(axis = 1)
        max_cos_sim = cos_sim.max(axis = 1)
        closest_score = cand_score[closest]
        pass_thresh = max_cos_sim > thresh

        to_change =  (~checked) & pass_thresh
        pred_scores[to_change] = closest_score[to_change]
        checked[pass_thresh] = True
        indices = np.arange(num_students)[~checked]
        emb_to_check = embeddings[~checked]

        print(len(emb_to_check),end =' ')

    # Save the scores
    to_save = pd.DataFrame({'Student ID':student_ids,f'Qstn # {question}':question,f'Score # {question}':pred_scores})
    out_name = input("Enter the name of the output file: ")
    
    #if out name does not end with .csv, add it
    if out_name[-4:] != '.csv':
        out_name += '.csv'
    data.to_csv(out_name,index=False)


if __name__ == "__main__":
    main()