#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import os
import gzip 
import multiprocessing
import string
import argparse
import glob
import itertools
import datetime
import logging
import sys
from collections import Counter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TweetYielder(object):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __iter__(self):
        for filepath in self.filepaths:                                       
            for line in gzip.open(filepath,"rt"):
                line = line.split('\t')
                tweet = line[-1].split()
                yield tweet
            print("Passed through {}".format(filepath))



def write_logfile(outfile_path, args, missing_files, month_range, start_time):
    logfile_path = outfile_path + '.log'
    with open(logfile_path, 'w') as logfile:
        logfile.write('Script started at: {}\n\n'.format(start_time))
        logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
        logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
        logfile.write('Options used:-\n')
        for (option, value) in args.items():
            logfile.write('{}\t{}\n'.format(option,value))
        logfile.write('\nMonth range:- {}\n\n'.format(month_range))
        if missing_files:
            logfile.write('Missing files:- {}'.format(missing_files))
        else:
            logfile.write('Missing files: None.')


def train_and_save_model_independent(tweets, params, model_save_location, training_mode, step_size, month_range, start_time, missing_files):

    # train the model
    model = gensim.models.Word2Vec(tweets, **params) 


    # save the model and write the logfile.
    output_dir = '/'.join([model_save_location, training_mode, str(step_size), "_".join(month_range), "vec_"+str(params['size'])+"_w"+str(params['window'])+"_mc"+str(params['min_count'])+"_iter"+str(params['iter'])])

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = '/'.join([output_dir, "saved_model.gensim"])

    model.save(output_filepath)
    
    write_logfile(output_filepath, args, missing_files, month_range, start_time)
    print("Output and log file for {} written to {}".format(month_range, output_dir))

    return month_range


def success(month_range): 
    sys.stdout.write("{} completed at {}\n".format(month_range, datetime.datetime.now()))
    sys.stdout.flush()


def error(a):
    sys.stderr.write("\nError: {}\n\n".format(a))
    sys.stderr.flush()



def generate(vector_size, window_size, min_count, no_of_iter, start_year, end_year, start_month, end_month, step_size, skipgram, training_mode, corpus_location, model_save_location, start_time):

    if training_mode == 'independent':
        pool = multiprocessing.Pool() 


    params = {'sg': skipgram, 'size': vector_size , 'window': window_size, 'min_count': min_count , 'iter' : no_of_iter,'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3}


    i = 1
    missing_files = []
    filepaths = []
    model = None
    for year in range(start_year,end_year+1):
        print("\n\non year {}".format(year))
        for month in range(1,13):

            print("\non month {}".format(month))
                
            if year == start_year and month < start_month:
                print("not got to start year_month yet, so continuing the loop")
                continue
            elif year == end_year and month > end_month:
                print("gone past end year_month, so breaking the loop")
                break


            # if we're at the beginning of a time-slice:
            elif i == 1:

                print("i = 1. so we're at the beginning of a time-slice.")

                dirPath = os.path.join(corpus_location,str(year))
                file_name = "{}-{:02}.csv.gz".format(year,month)
                month_range = [("{}-{:02}".format(year,month))]

                # append filepath for the current month to the list of filepaths for the current time-slice
                if file_name in os.listdir(dirPath):
                    print("We'll add the file for month {}-{:02} to our (empty) list of filepaths.".format(year,month))
                    filepaths.append(os.path.join(dirPath,file_name))

                else:
                    print("The data for {}-{:02} is not available.".format(year,month))
                    missing_files.append(file_name)


            # if we're already part way through a time-slice:
            else: 

                print("i = {}.".format(i))

                dirPath = os.path.join(corpus_location,str(year))
                file_name = "{}-{:02}.csv.gz".format(year,month)



                # append filepath for the current month to the list of filepaths for the current time-slice
                if file_name in os.listdir(dirPath):
                    print("We'll add the file for month {}-{:02} to our list of filepaths.".format(year,month))
                    filepaths.append(os.path.join(dirPath,file_name))

                else:
                    print("The data for {}-{:02} is not available.".format(year,month))
                    missing_files.append(file_name)


            # if we're at the end of a time-slice:
            if i == step_size:

                print("i = {}. We're at the end of the time-slice!".format(i))

                month_range.append("{}-{:02}".format(year,month))


                # if any of the files for this time-series actually exist (hence we have constructed an iterator for them):
                if filepaths:

                    # create an iterator over all of the tweets for the current time-slice
                    tweets = TweetYielder(filepaths)



                    if training_mode == 'independent':

                        # create a new word2vec model and train it on all the tweets for the current time-slice
                        print("we'll now train a new model on the data for this time-slice.")
                        
                        # model = gensim.models.Word2Vec(tweets, **params) 
                        pool.apply_async(train_and_save_model_independent, (tweets, params, model_save_location, training_mode, step_size, month_range, start_time, missing_files), callback=success, error_callback=error)




                    elif training_mode == 'continuous':

                        # if we have already initialised a model:
                        if model:

        
                            print("started building vocab at {}".format(datetime.datetime.now()))
                            # first we need to do the freq thesholding on the new vocab
                            corpus_count = 0

                            c = Counter()
                            for t in tweets:
                                corpus_count += 1
                                c.update(t)
                            for w in list(c.keys()):
                                if c[w] < min_count:
                                    del c[w]

                            print("done building vocab at {}".format(datetime.datetime.now()))

                            # train the existing model.
                            print("we'll now train our EXISTING model on the data for this time-slice.")

                            model.build_vocab_from_freq(c, update=True)
                            del c # to free up RAM
                            model.train(tweets, total_examples=corpus_count, epochs=model.iter)

                        # initialise and train model
                        else:
                            print("we'll now train a new model on the data for this time-slice.")
                            model = gensim.models.Word2Vec(tweets, **params)


                        # save the model and write the logfile.
                        output_dir = '/'.join([model_save_location, training_mode, str(step_size), "_".join(month_range), "vec_"+str(params['size'])+"_w"+str(params['window'])+"_mc"+str(params['min_count'])+"_iter"+str(params['iter'])+"_sg"+str(params['sg'])])

                        os.makedirs(output_dir, exist_ok=True)
                        output_filepath = '/'.join([output_dir, "saved_model.gensim"])

                        model.save(output_filepath)
                        
                        write_logfile(output_filepath, args, missing_files, month_range, start_time)
                        print("Output and log file for {} written to {}".format(month_range, output_dir))




                else:
                    pass
                    # all files in the month range were missing.


                i = 1
                missing_files = []
                filepaths = []

            else:
                i += 1

    if training_mode == 'independent':
        try:
            pool.close() 
            pool.join() 
        except:
            pass

        print("All threads finished.")

                

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vector_size", required=True, help="vector size")
    ap.add_argument("-w", "--window_size", required=True, help="window size")
    ap.add_argument("-mc", "--min_count", required=True, help="min count")
    ap.add_argument("-i", "--no_of_iter", required=True, help="no of iteration")
    ap.add_argument("-sy", "--start_year", required=True, help="start year: integer, e.g. 2014")
    ap.add_argument("-sm", "--start_month", required=True, help="start month: integer, e.g. 6")
    ap.add_argument("-ey", "--end_year", required=True, help="end year: integer, e.g. 2014")
    ap.add_argument("-em", "--end_month", required=True, help="end month: integer, e.g. 6")
    ap.add_argument("-s", "--step_size", required=True, help="number of months per time-step: integer between 1 and 12")
    ap.add_argument("-t", "--training_mode", type=str, default="independent", help="'continuous' = continue training each time-slice from where we left off; 'independent' = create separate, independent models for each time-slice.")
    ap.add_argument("-c", "--corpus_location", type=str, default="/data/twitter_spritzer/cleaner_001p_nodups/", help="directory where corpus is located")
    ap.add_argument("-m", "--model_save_location", type=str, default="/models/cleaner_001p_nodups_models/", help="top-level directory where models are to be saved")
    ap.add_argument("-sg", "--skipgram", type=int, default=0, help="1 = skipgram; 0 = cbow")


    start_time = datetime.datetime.now()

    
    args = vars(ap.parse_args())
    print(args)
    
    args_known, leftovers = ap.parse_known_args()
    
    
    vs = int(args["vector_size"])
    ws = int(args["window_size"])
    mc = int(args["min_count"])
    nit = int(args["no_of_iter"])

    start_year = int(args["start_year"])
    start_month = int(args["start_month"])
    end_year = int(args["end_year"])
    end_month = int(args["end_month"])
    step_size = int(args["step_size"])
    skipgram = int(args["skipgram"])

    training_mode = args["training_mode"]

    corpus_location = args["corpus_location"]
    model_save_location = args["model_save_location"]



    generate(vs,ws,mc,nit,start_year,end_year,start_month,end_month,step_size,skipgram,training_mode,corpus_location,model_save_location, start_time)

