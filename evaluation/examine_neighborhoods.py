#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import argparse
import os

def get_time_slices(start_year,end_year,start_month,end_month,step_size):
	time_slices = []
	i = 1
	for year in range(start_year,end_year+1):
		for month in range(1,13):
				
			if year == start_year and month < start_month:
				continue
			elif year == end_year and month > end_month:
				break
			elif i == 1:
				month1 = "{}-{:02}".format(year,month)


			if i == step_size:
				month2 = "{}-{:02}".format(year,month)
				time_slices.append(month1+"_"+month2)

				i = 1
			else:
				i += 1
	return time_slices


def get_neighborhoods(word, model_path):

	try:
		model = gensim.models.Word2Vec.load(model_path)

	except FileNotFoundError:
		return '{} not found'.format(model_path)

	else:

		model = model.wv
		model.init_sims(replace=True)

		try:
			neighborhood = [i[0] for i in model.most_similar(word, topn=n)]
		except:
			return '{} not in vocab'.format(word)
		else:
			return neighborhood



if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("-w", "--words", type=str, default="snowflake stan snaps", help = "list of  words (separated by spaces) to examine")
	parser.add_argument("-n", "--n_neighbors", type=int, default=10, help="number of neighbor to display for each word")

	parser.add_argument("-v", "--vector_size", type=int, default=200, help="what vector size was used?")
	parser.add_argument("-ws", "--window_size", type=int, default=10, help="what window size was used?")
	parser.add_argument("-mc", "--min_count", type=int, default=600, help="what min count was used?")
	parser.add_argument("-i", "--no_of_iter", type=int, default=15, help="what no of iteration was used?")
	
	parser.add_argument("-sy", "--start_year", type=int, default=2012, help="start year: integer, e.g. 2014")
	parser.add_argument("-sm", "--start_month", type=int, default=1, help="start month: integer, e.g. 6")
	parser.add_argument("-ey", "--end_year", type=int, default=2017, help="end year: integer, e.g. 2014")
	parser.add_argument("-em", "--end_month", type=int, default=6, help="end month: integer, e.g. 6")
	parser.add_argument("-s", "--step_size", type=int, default=6, help="number of months per time-step: integer between 1 and 12")

	parser.add_argument("-t", "--training_mode", type=str, default="independent", help="What training mode was used? 'continuous' = continue training each time-slice from where we left off; 'independent' = create separate, independent models for each time-slice.")
	parser.add_argument("-m", "--model_save_location", type=str, default="/data/twitter_spritzer/models/cleaner_001p_nodups_models/", help="What was the top-level directory into which this group of models was saved?")

	options = parser.parse_args()


	words = options.words.split()
	n = options.n_neighbors

	time_slices = get_time_slices(options.start_year,options.end_year,options.start_month,options.end_month,options.step_size)
	

	for word in words:
		print(word)
		print('============================\n')
		for time_slice in time_slices:

			param_string = "vec_"+str(options.vector_size)+"_w"+str(options.window_size)+"_mc"+str(options.min_count)+"_iter"+str(options.no_of_iter)

			model_path = os.path.join(options.model_save_location, options.training_mode, str(options.step_size),time_slice, param_string, "saved_model.gensim")

			print(time_slice)
			print(get_neighborhoods(word, model_path))
			print('\n')

