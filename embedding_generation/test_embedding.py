#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import os
from gensim.test.utils import datapath
import argparse
import gensim.downloader as api


#corpus_location = "/data/twitter_spritzer/corpus_by_month/"
#model_save_locaton = "/data/twitter_spritzer/"

#corpus_location = "/Users/fliza/Documents/clean_corpus/"
#model_save_locaton = "/Users/fliza/Documents/clean_corpus/"


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

def retrieve_model(model_save_locaton, vector_size,window_size,min_count,no_of_iter,skipgram, year="2011", month="09", whole=False, oy=False):
    if whole == True:
        fname = os.path.join(model_save_locaton, "models", "vec_"+str(vector_size)+"_w"+str(window_size)+"_mc"+str(min_count)+"_iter"+str(no_of_iter)+"_sg"+str(skipgram), "saved_model.gensim")
    elif oy == True :
        fname = os.path.join(model_save_locaton, year,"vec_"+str(vector_size)+"_w"+str(window_size)+"_mc"+str(min_count)+"_iter"+str(no_of_iter)+"_sg"+str(skipgram), "saved_model.gensim")
    else:
        fname = os.path.join(model_save_locaton, "models", year, month, "vec_"+str(vector_size)+"_w"+str(window_size)+"_mc"+str(min_count)+"_iter"+str(no_of_iter)+"_sg"+str(skipgram), "saved_model.gensim")
    model = gensim.models.Word2Vec.load(fname)
    return model



def most_similar(model):
    while True:
        word = input("Enter a word (QUIT to exit):")
        if word == "QUIT":
            break
        #print(model.wv.most_similar(word))
        print("====================")
        print(get_related_terms(model, word))
        print("====================")

def embedding_evaluation(model, vector_size,window_size,min_count,no_of_iter, skipgram, year="2011", month="09", start=2011, end=2018, whole=False, One_month=False):  
    similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))  #triple (pearson, spearman, ratio of pairs with unknown words).
    analogy = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
    return analogy[0], similarities[0][0]



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-msl", "--model_save_location", default="/Users/fliza/Documents/clean_corpus/", help = "path to save the model")  
    ap.add_argument("-v", "--vector_size", required=True, help="vector size")
    ap.add_argument("-w", "--window_size", required=True, help="window size")
    ap.add_argument("-mc", "--min_count", required=True, help="min count")
    ap.add_argument("-i", "--no_of_iter", required=True, help="no of iteration")
    ap.add_argument("-sg", "--skipgram", required=False, type=int, default=0, help="1 = skipgram; 0 = cbow")
    ap.add_argument("-y", "--year", required=False, help="year")
    ap.add_argument("-m", "--month", required=False, help="month")
    ap.add_argument("-s", "--start_year", required=False, help="start year")
    ap.add_argument("-e", "--end_month", required=False, help="end year")
    ap.add_argument("-wl", "--whole", required=False,default="0", help="whole")
    ap.add_argument("-om", "--one_month", required=False, help="one month")
    ap.add_argument("-oy", "--one_year", required=False, default="0", help="one year")
    ap.add_argument("-bs", "--base_line", required=False, default="0", help="0 for no baseline")
    ap.add_argument("-ms", "--most_similar", required=False, default="0", help="0 for no most similar list")
    
    
    args = vars(ap.parse_args())
    
    args_known, leftovers = ap.parse_known_args()
    
    
    vs = int(args["vector_size"])
    ws = int(args["window_size"])
    mc = int(args["min_count"])
    nit = int(args["no_of_iter"])
    sg = int(args["skipgram"])
    
    
    
    if int(args_known.base_line) == 1:
        model = api.load("glove-twitter-200")
        analogy_score, similarity_score = embedding_evaluation(model, vs,ws,mc,nit,sg)
        print("With baseline -- Analogy: ", analogy_score, "Similarity: ", similarity_score)

    if int(args_known.one_year) == 1:
        y = args["year"]
        msl = args["model_save_location"]
        model = retrieve_model(msl,vs,ws,mc,nit,sg,y, oy = True) 
        analogy_score, similarity_score = embedding_evaluation(model, vs,ws,mc,nit,sg,year=y, whole=False, One_month=False)
        print("Analogy: ", analogy_score, "Similarity: ", similarity_score)
        if int(args_known.most_similar) == 1:
            most_similar(model)
            
    elif args_known.year and args_known.month is not None:                     # for one month's embedding
        y = args["year"]
        m = args["month"]
        msl = args["model_save_location"]
        model = retrieve_model(msl,vs,ws,mc,nit,sg,y,m) 
        analogy_score, similarity_score = embedding_evaluation(model, vs,ws,mc,nit,sg,year=y,month=m, whole=False, One_month=True)
        print("Analogy: ", analogy_score, "Similarity: ", similarity_score)
        if int(args_known.most_similar) == 1:
            most_similar(model)
    elif args_known.start_year and args_known.end_month is not None:         # for each month in a range 
        s = int(args["start_year"])
        e  = int(args["end_month"])
        msl = args["model_save_location"]
        model = retrieve_model(msl,vs,ws,mc,nit,sg) 
        analogy_score, similarity_score  = embedding_evaluation(model, vs,ws,mc,nit,sg,start=s,end=e,whole=False, One_month=False)
        print("Analogy: ", analogy_score, "Similarity: ", similarity_score)
        if int(args_known.most_similar) == 1:
            most_similar(model)
    else:
        if int(args["whole"])==1:
            msl = args["model_save_location"]
            model = retrieve_model(msl,vs,ws,mc,nit,sg) 
            analogy_score, similarity_score  = embedding_evaluation(model, vs,ws,mc,nit,sg,whole=True, One_month=False)               # for the whole corpus
            print("Analogy: ", analogy_score, "Similarity: ", similarity_score)
            if int(args_known.most_similar) == 1:
                most_similar(model)
        else:
            print("Please respecify the input arguments.")



