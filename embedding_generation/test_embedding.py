#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import os
from gensim.test.utils import datapath
import argparse
import gensim.downloader as api



def get_related_terms(model, token, topn=10):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """
    try:
        for word, similarity in model.most_similar(positive=[token], topn=topn):
            print (word, round(similarity, 3))
    except:
        print("Error!")

def retrieve_model(model_path):
    fname = os.path.join(model_path)
    model = gensim.models.Word2Vec.load(fname)
    return model



def most_similar(model):
    while True:
        word = input("Enter a word (QUIT to exit):")
        if word == "QUIT":
            break
        print("====================")
        print(get_related_terms(model, word))
        print("====================")

def embedding_evaluation(model):  
    similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))  #triple (pearson, spearman, ratio of pairs with unknown words).
    analogy = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
    return analogy[0], similarities[0][0]



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-mp", "--model_path", default="/models/independent/1/2012-01_2012-01/vec_200_w10_mc500_iter15_sg0/saved_model.gensim", help = "path to saved word embedding model")  
    ap.add_argument("-bs", "--base_line", required=False, default="0", help="0 for no baseline")
    ap.add_argument("-ms", "--most_similar", required=False, default="0", help="0 for no most similar list")
    
    args = vars(ap.parse_args())
    args_known, leftovers = ap.parse_known_args()
   
    if int(args_known.base_line) == 1:
        model = api.load("glove-twitter-200")
        analogy_score, similarity_score = embedding_evaluation(model, vs,ws,mc,nit,sg)
        print("With baseline -- Analogy: ", analogy_score, "Similarity: ", similarity_score, "\n")
        
    mp = args["model_path"]
    model = retrieve_model(mp) 
    analogy_score, similarity_score = embedding_evaluation(model)
    print("Analogy: ", analogy_score, "Similarity: ", similarity_score)
    
    if int(args_known.most_similar) == 1:
        most_similar(model)
