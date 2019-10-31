#!/usr/bin/python3

import gensim
import argparse
import os
from hamilton_semantic_change_measures import *
import numpy as np
import datetime
from collections import Counter
import json

# to do: acual time series should be written to a file. Then they can be clustered, and also if we change the way we want to rank them we don't actually have to compute them all over again.

def load_model(model_path):
	"""
	Load the trained gensim word embedding model stored at model_path. 

	Since we don’t need the full model state any more (don’t need to continue training), the state can be discarded, and we just return the trained vectors (i.e. the KeyedVectors instance, model.wv).

	We call init_sims() to precompute L2-normalized vectors, using 'replace=True' to forget the original vectors and only keep the normalized ones (saves lots of memory).
	"""
	model = gensim.models.Word2Vec.load(model_path)
	model = model.wv
	model.init_sims(replace=True)
	return model


def get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, vocab, distance_measure, k, training_mode):
	"""
	Return a dictionary which contains, for each word which appears in the intersection of the vocabularies of the current timestep's model,the alignment reference model (if applicable), and the comparison reference model, the distance between that word's representation in the current timestep's model and its representation in the comparison reference model.

	"""

	# Load the model for the current timestep, the model we are aligning everything to, and the model we are comparing everything to.
	model = load_model(model_path)
	alignment_reference_model = load_model(alignment_reference_model_path)
	comparison_reference_model = load_model(comparison_reference_model_path)

	# Align both the current timestep's model and the comparison reference model to the alignment reference model.
	if distance_measure == 'cosine':
		model = smart_procrustes_align_gensim(alignment_reference_model, model)
		comparison_reference_model = smart_procrustes_align_gensim(alignment_reference_model, comparison_reference_model)


	# This will be a dictionary with keys = words, values = distance between the word's vector in the current timestep and its vector in the comparison reference model.
	dist_dict = {}


	for word in vocab:

		if word in comparison_reference_model and word in model:
			if distance_measure == 'cosine' and training_mode == 'independent':
				dist_dict[word] = cosine(comparison_reference_model[word], model[word])
			else: # distance_measure == 'neighborhood':
				dist_dict[word] = measure_semantic_shift_by_neighborhood(comparison_reference_model, model, word, k)
		
		# if the word does not occur in both the current timestep's model and the comparison reference model's vocab (and implicitly, in the alignment reference model, since we aligned them both to that), then we can't calculate a distance measure for this word and this timestep, so we just set its value to None. 
		else:
			dist_dict[word] = None
		# else:
		# 	raise RunTimeError("Invalid command line argument: Only possible values for option -m (--distance_measure) are 'cosine' or 'neighborhood'")

	return dist_dict


def get_z_score_dict(dist_dict):
	"""
	Convert the dictionary of distance scores for a given timestep into a dictionary of z-scores - i.e. how many standard deviations is a given word's distance score from the mean of all word's distance scores at this timestep?
	"""

	# calculate mean and variance of distance scores, ignoring any words for whom the value was None -- calculate the mean and variance of the distance scores for all words which are represented in both the current timestep's model and the comparison reference model.
	mean = np.mean(list([i for i in dist_dict.values() if i]))
	var = np.var(list([i for i in dist_dict.values() if i]))

	z_score_dict = {}
	for word in dist_dict:
		# if we actually have a distance score for the word, standardize it.
		if dist_dict[word]:
			z_score_dict[word] = (dist_dict[word] - mean) / np.sqrt(var)
		# otherwise, return None, to keep track of the fact that this word was not represented in both models.
		else:
			z_score_dict[word] = None

	return z_score_dict


def get_zero_crossings(x):
	return np.where(np.diff(np.sign(x)))[0]


def detect_change_point(word, time_slice_labels, z_score_series, compare_to):
	"""
	This function computes the mean-shift time-series from the given word's z-score series, then computes the p-value series, 
	"""

	notNone_time_slice_labels = [time_slice_labels[i] for i in range(len(time_slice_labels)) if z_score_series[i]]

	if notNone_time_slice_labels:

		z_score_series = [i for i in z_score_series if i]


		change_points = []
		zero_crossings = get_zero_crossings(z_score_series)
		if zero_crossings.size > 0:
			for i in zero_crossings:
				mean_before = np.mean(z_score_series[:i+1])
				mean_after = np.mean(z_score_series[i+1:])
				if compare_to == 'last' or compare_to == 'previous':
					score = mean_before-mean_after
				else:
					score = mean_after-mean_before
				change_points.append((score, time_slice_labels[i]))

			change_point = max(change_points)

			return (word, change_point[1], change_point[0])
		else:
			return None
	else: # word must not have occured in one or both of the reference models, hence there are no time slices in which we actually were able to compute a distance for it. 
		return None


def write_logfile(outfilepath, options, start_time):
	logfile_path = outfilepath + '.log'
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Options used:- {}\n')
		for (option, value) in vars(options).items():
			logfile.write('{}\t{}\n'.format(option,value))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--models_rootdir", type=str, default="/data/twitter_spritzer/models/synthetic_evaluation_dataset_models/independent/1/", help = "path to directory where models are stored")
	parser.add_argument("-f", "--first_timeslice", type=str, default='2012_01', help = "path to file where output should be written")
	parser.add_argument("-l", "--last_timeslice", type=str, default='2014_06', help = "path to file where output should be written")
	parser.add_argument("-a", "--align_to", type=str, default="last", help = "which model to align every other model to: 'first', 'last', or 'previous'")
	parser.add_argument("-c", "--compare_to", type=str, default="last", help = "which model's vector to compare every other model's vector to: 'first', 'last', or 'previous'")
	parser.add_argument("-m", "--distance_measure", type=str, default="cosine", help = "which distance measure to use 'cosine', or 'neighborhood'")
	parser.add_argument("-k", "--k_neighbors", type=int, default=25, help = "Number of neighbors to use for neighborhood shift distance measure")
	parser.add_argument("-n", "--n_best", type=int, default=1000, help = "Size of n-best list to print")
	parser.add_argument("-v", "--vocab_threshold", type=int, default=75, help = "percent of models which must contain word in order for it to be included")
	parser.add_argument("-o", "--outfiles_dir", type=str, default="/data/twitter_spritzer/analysis/synthetic_evaluation_dataset/z_score_analysis_candidates/monthly/independent/vec_200_w9_mc100_iter15/2012_01_to_2014_06/", help = "Path to file where results will be written")


	parser.add_argument("-vs", "--vector_size", type = int, default=200, help="vector size")
	parser.add_argument("-ws", "--window_size", type = int, default=9, help="window size")
	parser.add_argument("-mc", "--min_count", type = int, default=100, help="min count")
	parser.add_argument("-ni", "--no_of_iter", type = int, default=15, help="no of iteration")

	parser.add_argument("-t", "--training_mode", type = str, default='independent', help="training mode: was it independent or continuous?")

	options = parser.parse_args()



	start_time = datetime.datetime.now()

	print("Starting at {}".format(start_time))

	# First, we construct a list of the filepaths of all the models we have, and a list of the time-slices they correspond to.

	# We initalize the vocab to the set of words which occur in at least v% of the models.
	
	model_paths = []
	time_slice_labels = []
	vocab_filepath = "{}/time_series_vocab_{}pc_{}_to_{}.txt".format(options.models_rootdir, options.vocab_threshold, options.first_timeslice, options.last_timeslice)

	(first_year, first_month) = (int(i) for i in options.first_timeslice.split('_'))
	(last_year, last_month) = (int(i) for i in options.last_timeslice.split('_'))

	# if we've already stored the common vocab, can just read it, don't have to load all the models and check their vocab
	if os.path.isfile(vocab_filepath):

		print("vocab already stored.")

		vocab = set()
		with open(vocab_filepath, 'r') as infile:
			for line in infile:
				vocab.add(line.strip())

		for year in range(2011,2019):

			if year < first_year:
				continue
			elif year > last_year:
				break

			for month in range(1,13):
					
				if year == first_year and month < first_month:
					continue
				elif year == last_year and month > last_month:
					break

				time_slice = "{}_{:02}".format(year, month)
				model_path = "{}/{}-{:02}_{}-{:02}/vec_{}_w{}_mc{}_iter{}/saved_model.gensim".format(options.models_rootdir, year, month, year, month, options.vector_size, options.window_size, options.min_count, options.no_of_iter)
				if os.path.isfile(model_path):
					model_paths.append(model_path)
					time_slice_labels.append(time_slice)

	else:
		# if we HAVEN'T already stored the common vocab, we DO need to load all the models and check their vocab

		print("vocab NOT already stored.")

		vocab_counter = Counter()
		for year in range(2011,2019):

			if year < first_year:
				continue
			elif year > last_year:
				break


			for month in range(1,13):

				if year == first_year and month < first_month:
					continue
				elif year == last_year and month > last_month:
					break

				time_slice = "{}_{:02}".format(year, month)
				model_path = "{}/{}-{:02}_{}-{:02}/vec_{}_w{}_mc{}_iter{}/saved_model.gensim".format(options.models_rootdir, year, month, year, month, options.vector_size, options.window_size, options.min_count, options.no_of_iter)
				try:
					model = load_model(model_path)
				except FileNotFoundError:
					pass
				else:
					print("loaded {} at {}".format(time_slice, datetime.datetime.now()))
					model_paths.append(model_path)
					time_slice_labels.append(time_slice)
					vocab_counter.update(model.vocab.keys())
		
		n_models = len(model_paths)
		print(n_models)
		print(vocab_counter.most_common(10))
		vocab = set([w for w in vocab_counter if vocab_counter[w] >= options.vocab_threshold * 0.01 * n_models])
		del vocab_counter

		with open(vocab_filepath, 'w') as outfile:
			for word in vocab:
				outfile.write(word+'\n')

	print("\nGot vocab at {}".format(datetime.datetime.now()))
	print("size of vocab: {}\n".format(len(vocab)))



	zscores_filepath = options.outfiles_dir+'/time_series_analysis_z_scores_f{}_l{}_a{}_c{}_m{}_k{}_v{}.json'.format(options.first_timeslice, options.last_timeslice, options.align_to, options.compare_to, options.distance_measure, options.k_neighbors, options.vocab_threshold)
	# if os.path.isfile(zscores_filepath):

	# 	# read it
	# 	dict_of_z_score_dicts = json.load(zscores_filepath)

	# else:


	dict_of_z_score_dicts = {}
	time_slice_labels_used = []

	for (i, model_path) in enumerate(model_paths):

		if i == 0 and (options.compare_to == 'previous' or options.align_to =='previous' or options.compare_to == 'first'):
			continue

		elif i == len(model_paths) - 1 and options.compare_to == 'last': 
			continue

		else:

			if options.align_to == 'first':
				alignment_reference_model_path = model_paths[0]
			elif options.align_to == 'last':
				alignment_reference_model_path = model_paths[-1]
			else:
				alignment_reference_model_path = model_paths[i-1]


			if options.compare_to == 'first':
				comparison_reference_model_path = model_paths[0]
			elif options.compare_to =='last':
				comparison_reference_model_path = model_paths[-1]
			else:
				comparison_reference_model_path = model_paths[i-1]

			dict_of_z_score_dicts[time_slice_labels[i]] = get_z_score_dict(get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, vocab, options.distance_measure, options.k_neighbors, options.training_mode))

			time_slice_labels_used.append(time_slice_labels[i])

	print("GOT DICT OF Z-SCORE DICTS at {}\n".format(datetime.datetime.now()))

	os.makedirs(options.outfiles_dir,exist_ok=True)
	with open(zscores_filepath, 'w') as outfile:
		json.dump(dict_of_z_score_dicts, outfile)





	# Finally, we do the change-point analysis on each word's z-score time-series. We keep a ranked list of the n 'best' change-points detected, and print it when we're done.

	results = []
	for (i,word) in enumerate(vocab):
		#print('{}: {}\t starting at {}'.format(i, word, datetime.datetime.now()))
		z_score_series = [dict_of_z_score_dicts[time_slice][word] for time_slice in time_slice_labels_used]
		change_point = detect_change_point(word, time_slice_labels_used, z_score_series, options.compare_to)
		if change_point:
			#(word, time_slice, score) = change_point
			results.append(change_point)

	results = sorted(results, key=lambda x:-x[2])
	


	os.makedirs(options.outfiles_dir,exist_ok=True)
	outfile_path = options.outfiles_dir+'/time_series_analysis_z_scores_only_output_f{}_l{}_a{}_c{}_m{}_k{}_v{}.tsv'.format(options.first_timeslice, options.last_timeslice, options.align_to, options.compare_to, options.distance_measure, options.k_neighbors, options.vocab_threshold)
	#with open(options.outfile_path, 'w') as outfile:
	with open(outfile_path, 'w') as outfile:
		for (i, item) in enumerate(results[:options.n_best]):
			#print(i, ":", item)
			outfile.write('\t'.join([str(s) for s in item])+'\n')

	print("All done at {}. Writing log file...\n".format(datetime.datetime.now()))
	write_logfile(outfile_path, options, start_time)
	print("Written log file.")