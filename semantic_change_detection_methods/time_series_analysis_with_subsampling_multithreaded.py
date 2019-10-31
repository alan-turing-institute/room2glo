#!/usr/bin/python3

import argparse
import os
import numpy as np
import datetime
from collections import Counter, defaultdict
import json
import glob
import itertools
import multiprocessing

# to do: acual time series should be written to a file. Then they can be clustered, and also if we change the way we want to rank them we don't actually have to compute them all over again.




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
			#z_score_dict[word] = None
			pass

	return z_score_dict


def compute_mean_shift(time_series_dict, j, compare_to):
	"""
	Compute the mean_shift score at index j of the given time-series.
	"""

	timestep_to_index = {}
	for (i,timestep) in enumerate(time_series_dict.keys()):
		timestep_to_index[timestep] = i

	xs = list(itertools.chain.from_iterable([timestep_to_index[timestep]]*len(time_series_dict[timestep]) for timestep in time_series_dict))
	ys = list(itertools.chain.from_iterable(time_series_dict.values()))


	# Mean-shift score for timestep j = mean(scores after j) - mean(scores up to j). So if representations for a word after time j are on average much further from the representation in the comparison reference model than the representations before time j, the mean-shift score will be large.
	if compare_to == 'first':
		return np.mean([ys[i] for i in range(len(ys)) if xs[i] > j])  - np.mean([ys[i] for i in range(len(ys)) if xs[i] <= j])
	else: # compare_to == 'last' or compare_to == 'previous':


		# if we are comparing to the last time-slice, we are looking for a point j where before j, the vector was not very similar to the one in the last time-slice, but ater j, it became significantly *more* similar to the one in the last time-slice. So we want the z-scores BEFORE j to be bigger than the ones after. Which would make mean(up to j) - mean(after j) be large and positive.

		# if we are comparing to the previous time-slice, then doing it this way round would mean we detect words which were not very-self similar at first, and then started to become more self-similar at some point. i.e. unstable meaning replaced by a stable one?
		return np.mean([ys[i] for i in range(len(ys)) if xs[i] <= j])  - np.mean([ys[i] for i in range(len(ys)) if xs[i] > j])



def get_mean_shift_series(time_series_dict, compare_to):
	"""
	Compute a given word's mean_shift time-series from its time-series of z-scores. 
	"""
	return [compute_mean_shift(time_series_dict, j, compare_to) for j in range(len(time_series_dict.keys())-1)]



def get_p_value_series(word, mean_shift_series, n_samples, z_scores_dict, compare_to):
	"""
	Randomly permute the z-score time series n_samples times, and for each
	permutation, compute the mean-shift time-series of those permuted z-scores, and at each index, check if the mean-shift score from the permuted series is greater than the mean-shift score from the original series. The p-value is the proportion of randomly permuted series which yielded a mean-shift score greater than the original mean-shift score.
	"""
	p_value_series = np.zeros(len(mean_shift_series))
	for i in range(n_samples):
		permuted_z_score_series = np.random.permutation(list(z_scores_dict.values()))
		permuted_z_scores_dict = {}
		for (i,z_scores) in enumerate(permuted_z_score_series):
			permuted_z_scores_dict[i] = z_scores
		mean_shift_permuted_series = get_mean_shift_series(permuted_z_scores_dict, compare_to)

		for x in range(len(mean_shift_permuted_series)):
			# if the original mean_shift_series has a NaN value, then we just increment the counter, so that we'll end up with a p-value of 1 for this index. We would get a NaN if we tried to take the mean of an empty slice in the mean-shift calculation. Which would happen if either before or after index j, there were no time-steps in which the word had actually occured in both models. 
			# if np.isnan(mean_shift_series[x]):
			# 	print("Mean_shift_series score is 'NaN'! Word: {}\nSeries:{}".format(word,mean_shift_series))
			# 	p_value_series[x] += 1
			# if we got a NaN in the permuted series, this will evaluate as False.
			#  * Are NaNs problematic for the statistical validity here? *
			# could I instead throw away all indices of the original z-score series which have None values (and discard the corresponding time-slices from the time-slice series, and just proceed using only the time-slices for which we have values that are not None? 
			# elif mean_shift_permuted_series[x] > mean_shift_series[x]:
			if mean_shift_permuted_series[x] > mean_shift_series[x]:
				p_value_series[x] += 1
	p_value_series /= n_samples
	return p_value_series



def detect_change_point(word, z_scores_dict, n_samples, p_value_threshold, gamma_threshold, compare_to):
	"""
	This function computes the mean-shift time-series from the given word's z-score series, then computes the p-value series, 
	"""


	index_to_timestep = {}
	for (i,timestep) in enumerate(z_scores_dict.keys()):
		index_to_timestep[i] = timestep


	mean_shift_series = get_mean_shift_series(z_scores_dict, compare_to)

	p_value_series = get_p_value_series(word, mean_shift_series, n_samples, z_scores_dict, compare_to)

	# set p-values for any time-slices with average z-scores below gamma threshold to 1, so that these time-slices won't get chosen. 
	for i in range(len(p_value_series)):
		if np.mean(z_scores_dict[index_to_timestep[i]]) < gamma_threshold:
			p_value_series[i] = 1

	# find minimum p_value:
	p_value_series = np.array(p_value_series)
	try:
		min_p_val = p_value_series.min()
	except ValueError:
		print(word)
		print(z_scores_dict)
		print(mean_shift_series)
		print(p_value_series)


	# if minimum p_value is below the threshold:
	if min_p_val < p_value_threshold:

		# get indices of time-slices with minimum p_value:
		indices = np.where(p_value_series == min_p_val)[0]

		# as a tie-breaker, return the one which corresponds to the biggest mean_shift
		(change_point, mean_shift) = max([(i, mean_shift_series[i]) for i in indices], key = lambda x:x[1])

		z_score = np.mean(z_scores_dict[index_to_timestep[change_point]])
		time_slice_label = index_to_timestep[change_point]


		return (word, time_slice_label, min_p_val, mean_shift, z_score)

	else: 
		return None



def get_word_change_point(word):
	z_scores_dict = dict_of_z_scores_by_word[word]

	change_point = detect_change_point(word, z_scores_dict, options.n_samples, options.p_value_threshold, options.gamma_threshold, options.compare_to)

	if change_point:
		return change_point



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
	parser.add_argument("-d", "--models_rootdir", type=str, default="/data/twitter_spritzer/models/synthetic_evaluation_dataset_models/subsampled_70/subsamples/10_70_percent_samples/", help = "path to directory where models are stored")
	parser.add_argument("-f", "--first_timeslice", type=str, default='2012_01', help = "path to file where output should be written")
	parser.add_argument("-l", "--last_timeslice", type=str, default='2017_06', help = "path to file where output should be written")
	parser.add_argument("-a", "--align_to", type=str, default="last", help = "which model to align every other model to: 'first', 'last', or 'previous'")
	parser.add_argument("-c", "--compare_to", type=str, default="last", help = "which model's vector to compare every other model's vector to: 'first', 'last', or 'previous'")
	parser.add_argument("-m", "--distance_measure", type=str, default="cosine", help = "which distance measure to use 'cosine', or 'neighborhood'")
	parser.add_argument("-k", "--k_neighbors", type=int, default=25, help = "Number of neighbors to use for neighborhood shift distance measure")
	parser.add_argument("-s", "--n_samples", type=int, default=1000, help = "Number of samples to draw for permutation test")
	parser.add_argument("-p", "--p_value_threshold", type=float, default=0.05, help = "P-value cut-off")
	parser.add_argument("-g", "--gamma_threshold", type=float, default=0, help = "Minimum z-score magnitude.")
	parser.add_argument("-r", "--rank_by", type=str, default='p_value', help = "What to rank words by: 'p_value', 'z_score', or 'mean_shift'")
	parser.add_argument("-n", "--n_best", type=int, default=1000, help = "Size of n-best list to print")
	parser.add_argument("-v", "--vocab_threshold", type=int, default=75, help = "percent of models which must contain word in order for it to be included")
	parser.add_argument("-o", "--outfiles_dir", type=str, default="/data/twitter_spritzer/analysis/synthetic_evaluation_dataset/kulkarni_candidates/monthly/independent/subsampled_70/subsamples/10_70_percent_samples/", help = "Path to file where results will be written")

	parser.add_argument("-vs", "--vector_size", type = int, default=200, help="vector size")
	parser.add_argument("-ws", "--window_size", type = int, default=9, help="window size")
	parser.add_argument("-mc", "--min_count", type = int, default=100, help="min count")
	parser.add_argument("-ni", "--no_of_iter", type = int, default=15, help="no of iteration")

	parser.add_argument("-t", "--training_mode", type = str, default='independent', help="training mode: was it independent or continuous?")

	parser.add_argument("-ss", "--step_size", type = int, default=1, help="step size: how many months per embedding model?")

	options = parser.parse_args()


	start_time = datetime.datetime.now()
	print("Starting at {}".format(start_time))



	sample_dirs = glob.glob("{}/*/".format(options.models_rootdir))



	dict_of_z_scores_by_word = defaultdict(lambda: defaultdict(list))

	for (sample_num, sample_dir) in enumerate(sorted(sample_dirs)):


		vocab_filepath = "{}/time_series_vocab_{}pc_{}_to_{}__{}_{}_vec_{}_w{}_mc{}_iter{}.txt".format(sample_dir, options.vocab_threshold, options.first_timeslice, options.last_timeslice, options.training_mode, options.step_size, options.vector_size, options.window_size, options.min_count, options.no_of_iter)

		vocab = set()
		with open(vocab_filepath, 'r') as infile:
			for line in infile:
				vocab.add(line.strip())

		
		distances_filepath = "{}/time_series_analysis_distances_{}pc_{}_to_{}__{}_{}_vec_{}_w{}_mc{}_iter{}__c{}_m{}_k{}.txt".format(sample_dir, options.vocab_threshold, options.first_timeslice, options.last_timeslice, options.training_mode, options.step_size, options.vector_size, options.window_size, options.min_count, options.no_of_iter, options.compare_to, options.distance_measure, options.k_neighbors)

		with open(distances_filepath, 'r') as infile:
			sample_dists = json.load(infile)



		zscores_filepath = "{}/time_series_analysis_z_scores_{}pc_{}_to_{}__{}_{}_vec_{}_w{}_mc{}_iter{}__c{}_m{}_k{}.txt".format(sample_dir, options.vocab_threshold, options.first_timeslice, options.last_timeslice, options.training_mode, options.step_size, options.vector_size, options.window_size, options.min_count, options.no_of_iter, options.compare_to, options.distance_measure, options.k_neighbors)


		dict_of_z_score_dicts = {}

		for timestep in sample_dists:
			dict_of_z_score_dicts[timestep] = get_z_score_dict(sample_dists[timestep])


		print("GOT DICT OF Z-SCORE DICTS for sample {} at {}\n".format(sample_num, datetime.datetime.now()))

		with open(zscores_filepath, 'w') as outfile:
			json.dump(dict_of_z_score_dicts, outfile)


		time_slices = sorted(dict_of_z_score_dicts.keys())
		for word in vocab:
			for time_slice in dict_of_z_score_dicts:
				if word in dict_of_z_score_dicts[time_slice]:
					dict_of_z_scores_by_word[word][time_slice].append(dict_of_z_score_dicts[time_slice][word])






	# Finally, we do the change-point analysis on each word's z-score time-series. We keep a ranked list of the n 'best' change-points detected, and print it when we're done.



	pool = multiprocessing.Pool()

	results = pool.map(get_word_change_point, dict_of_z_scores_by_word.keys())

	# for word in dict_of_z_scores_by_word:
	# 	#print('{}: {}\t starting at {}'.format(i, word, datetime.datetime.now()))
	# 	z_scores_dict = dict_of_z_scores_by_word[word]
	# 	change_point = detect_change_point(word, z_scores_dict, options.n_samples, options.p_value_threshold, options.gamma_threshold, options.compare_to)
	# 	if change_point:
	# 		#(word, time_slice, p_value, mean_shift, z_score) = change_point
	# 		results.append(change_point)

	print('got {} results'.format(len(results)))
	results = [r for r in results if r]
	print('got {} not-none results'.format(len(results)))

	if options.rank_by == 'z_score':
		results = sorted(results, key=lambda x:-x[4])
	elif options.rank_by == 'mean_shift':
		results = sorted(results, key=lambda x:-x[3])
	else: # options.rank_by == 'p_value'
		# we'll actually rank by mean-shift first and then p-value, so that words with the same p-value are sorted by the size of the mean-shift.
		results = sorted(results, key=lambda x:-x[3])
		results = sorted(results, key=lambda x:x[2])
	# else:
	# 	raise RunTimeError("Invalid command line argument: Only possible values for option -r (--rank_by) are 'z_score', 'mean_shift', or 'p_value'")
			


	os.makedirs(options.outfiles_dir,exist_ok=True)


	outfile_path = "{}/time_series_analysis_output_{}pc_{}_to_{}__{}_{}_vec_{}_w{}_mc{}_iter{}__c{}_m{}_k{}__s{}_p{}_g{}___pooled.txt".format(options.outfiles_dir, options.vocab_threshold, options.first_timeslice, options.last_timeslice, options.training_mode, options.step_size, options.vector_size, options.window_size, options.min_count, options.no_of_iter, options.compare_to, options.distance_measure, options.k_neighbors, options.n_samples, options.p_value_threshold, options.gamma_threshold)

	#with open(options.outfile_path, 'w') as outfile:
	with open(outfile_path, 'w') as outfile:
		for (i, item) in enumerate(results[:options.n_best]):
			#print(i, ":", item)
			outfile.write('\t'.join([str(s) for s in item])+'\n')

	print("All done at {}. Writing log file...\n".format(datetime.datetime.now()))
	write_logfile(outfile_path, options, start_time)
	print("Written log file.")
