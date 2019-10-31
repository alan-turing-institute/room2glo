#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import os
import gzip 
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


def write_logfile(outfile_path, args, start_time):
	logfile_path = outfile_path + '.log'
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Options used:-\n')
		for (option, value) in args.items():
			logfile.write('{}\t{}\n'.format(option,value))
		


def generate(corpus_location, year, month, model_load_location, model_save_location, start_time, min_count):

	# read the tweets from the last timestep
	dirPath = os.path.join(corpus_location,str(year))
	file_name = "{}-{:02}.csv.gz".format(year,month)
	filepaths = []
	if file_name in os.listdir(dirPath):
		print("We'll read the tweets for month {}-{:02} to our (empty) list of filepaths.".format(year,month))
		filepaths.append((os.path.join(dirPath,file_name)))

	tweets = TweetYielder(filepaths)



	# load saved model from first timestep
	model = gensim.models.Word2Vec.load(model_load_location)




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



	# save the new model and write the logfile.
	model.save(model_save_location)
	write_logfile(model_save_location, args, start_time)
	print("Output and log file written to {}".format(model_save_location))

				

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--month", type=int, default=6, help="month: integer, e.g. 6")
	ap.add_argument("-y", "--year", type=int, default=2017, help="year: integer, e.g. 2014")
	# ap.add_argument("-c", "--corpus_location", type=str, default="/data/twitter_spritzer/cleaner_001p_nodups/", help="directory where corpus is located")
	# ap.add_argument("-ms", "--model_save_location", type=str, default="/data/twitter_spritzer/models/cleaner_001p_nodups_models/", help="path to where model should be saved")
	# ap.add_argument("-ml", "--model_load_location", type=str, default="/data/twitter_spritzer/models/cleaner_001p_nodups_models/", help="path to model to load for pre-initialization")
	ap.add_argument("-c", "--corpus_location", type=str, default="/data2/synthetic_evaluation_dataset/subsampled_70/", help="directory where corpus is located")
	ap.add_argument("-ms", "--model_save_location", type=str, default="/data2/models/synthetic_evaluation_dataset_models/subsampled_70/nov_29/continuous2step/1/2017-06_2017-06/vec_200_w9_mc100_iter15_sg0/saved_model.gensim", help="path to where model should be saved")
	ap.add_argument("-ml", "--model_load_location", type=str, default="/data2/models/synthetic_evaluation_dataset_models/subsampled_70/nov_29/continuous/1/2012-01_2012-01/vec_200_w9_mc100_iter15_sg0/saved_model.gensim", help="path to model to load for pre-initialization")
	ap.add_argument("-mc", "--min_count", type=int, default=100, help="minimum number of times word must appear in data for time-slice in order to be included in training examples")


	start_time = datetime.datetime.now()

	
	args = vars(ap.parse_args())
	print(args)
	
	args_known, leftovers = ap.parse_known_args()
	

	month = args["month"]
	year = args["year"]

	corpus_location = args["corpus_location"]
	model_load_location = args["model_load_location"]
	model_save_location = args["model_save_location"]

	min_count = args["min_count"]


	generate(corpus_location, year, month, model_load_location, model_save_location, start_time, min_count)

