import argparse
import datetime
import os
import multiprocessing
import json
import numpy as np
import gzip
import sys

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



def maybe_return_pseudoword(context_word, year_month_index):

	pseudoword = context_words[context_word]['pseudoword']
	pseudoword_type = context_words[context_word]['pseudoword_type']
	set_number = context_words[context_word]['set_number']

	if pseudoword_type  == 6 or (pseudoword_type  == 7 and set_number == 1):
		dist_index = context_words[context_word]['dist_index']
		p_insert = pseudoword_insert_probs[pseudoword]["p{}_array_series".format(set_number)][year_month_index][dist_index]

	else:
		try:
			p_insert = pseudoword_insert_probs[pseudoword]["p{}_series".format(set_number)][year_month_index]
		except TypeError:
			print(pseudoword)
			print(context_word)
			print(pseudoword_insert_probs[pseudoword])
			raise()


	insert = np.random.RandomState().choice(2,1,p=[1-p_insert, p_insert])

	if insert:
		return pseudoword
	else:
		return False



def create_synthetic(year_month_index, subsampling, subsample_percent):

	year_month = year_months[year_month_index]

	with gzip.open(os.path.join(outfiles_rootdir, year_month[:4], year_month), 'wt') as outfile:

			tweets = TweetYielder([input_filepath])

			for tweet in tweets:

				if subsampling:
					if np.random.RandomState().random_sample() > subsample_percent*0.01:
						continue

				tweet_out = []

		
				for i in range(len(tweet)):


					if tweet[i] in context_words:
						pseudoword = maybe_return_pseudoword(tweet[i], year_month_index)
						if pseudoword:
							tweet_out.append(pseudoword)
						else:
  							tweet_out.append(tweet[i])
					else:
						tweet_out.append(tweet[i])

				outfile.write(' '.join(tweet_out)+'\n')
	return year_month




def success(year_month):
	sys.stdout.write("{} completed at {}\n".format(year_month,datetime.datetime.now()))
	sys.stdout.flush()


def error(a):
	sys.stdout.write("\nError: {}\n\n".format(a))
	sys.stdout.flush()



def write_logfile(outfiles_rootdir, options, start_time):
	logfile_path = '/'.join([outfiles_rootdir, 'corpus_provenance.log'])
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Options used:-\n')
		for (option, value) in vars(options).items():
			logfile.write('{}\t{}\n'.format(option,value))


if __name__ == "__main__":



	parser = argparse.ArgumentParser()

	# parser.add_argument("-i", "--input_filepath", type=str, default='data/clean_001p_local_nodups/2014/2014-04.csv.gz', help = "path to file where data for the month we want to use is stored")
	# parser.add_argument("-o", "--outfiles_rootdir", type=str, default='data/synthetic_evaluation_dataset/', help = "path to directory where synthetic dataset should be written")
	# parser.add_argument("-c", "--context_word_dict_filepath", type=str, default='data/synthetic_evaluation_dataset/context_word_dict.json', help = "path to directory where synthetic dataset should be written")
	# parser.add_argument("-p", "--pseudoword_dict_filepath", type=str, default='data/synthetic_evaluation_dataset/pseudoword_dict.json', help = "path to directory where synthetic dataset should be written")

	parser.add_argument("-i", "--input_filepath", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/2014/2014-12.csv.gz', help = "path to file where data for the month we want to use is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/synthetic_evaluation_dataset/', help = "path to directory where synthetic dataset should be written")
	parser.add_argument("-c", "--context_word_dict_filepath", type=str, default='/data/twitter_spritzer/synthetic_evaluation_dataset/context_word_dict.json', help = "path to directory where synthetic dataset should be written")
	parser.add_argument("-p", "--pseudoword_dict_filepath", type=str, default='/data/twitter_spritzer/synthetic_evaluation_dataset/pseudoword_dict.json', help = "path to directory where synthetic dataset should be written")

	parser.add_argument("-s", "--subsampling", type=int, default=0, help = "whether or not to subsample from original data. 1 = yes, 0 = no.")
	parser.add_argument("-sp", "--subsampling_percent", type=int, default=70, help = "size of sample (percent of original data) e.g. 70")
	parser.add_argument("-sy", "--start_year", type=int, default = 2012, help="start year: integer, e.g. 2012")
	parser.add_argument("-sm", "--start_month", type=int, default = 1, help="start month: integer, e.g. 6")
	parser.add_argument("-ey", "--end_year", type=int, default = 2017, help="end year: integer, e.g. 2014")
	parser.add_argument("-em", "--end_month", type=int, default = 6, help="end month: integer, e.g. 4")

	options = parser.parse_args()


	subsampling = options.subsampling
	subsampling_percent = options.subsampling_percent

	if subsampling:
		options.outfiles_rootdir += '/subsampled_{}/'.format(subsampling_percent)

	input_filepath = options.input_filepath
	outfiles_rootdir = options.outfiles_rootdir
	
	context_word_dict_filepath = options.context_word_dict_filepath
	pseudoword_dict_filepath = options.pseudoword_dict_filepath


	start_year = options.start_year
	end_year = options.end_year
	start_month = options.start_month
	end_month = options.end_month

	start_time = datetime.datetime.now()
	print('Starting at: {}\n\n'.format(start_time))



	year_months = []

	for year in range(start_year, end_year+1):

		os.makedirs(os.path.join(outfiles_rootdir, str(year)),exist_ok=True)

		for month in range(1,13):
			if year == start_year and month < start_month:
				continue
			elif year == end_year and month > end_month:
				break

			year_months.append("{}-{:02}.csv.gz".format(year, month))


	n_timesteps = len(year_months)


	# load the dictionaries.
	with open(context_word_dict_filepath, 'r') as infile:
		context_words = json.load(infile)

	with open(pseudoword_dict_filepath, 'r') as infile:
		pseudoword_insert_probs = json.load(infile)



	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine

	for year_month_index in range(n_timesteps):

		pool.apply_async(create_synthetic, (year_month_index, subsampling, subsampling_percent), callback=success, error_callback=error)

		# print('Starting {} at: {}\n\n'.format(year_months[year_month_index], datetime.datetime.now()))
		# create_synthetic(year_month_index, subsampling, subsampling_percent)
		# print('Finished {} at: {}\n\n'.format(year_months[year_month_index], datetime.datetime.now()))

	try:
		pool.close() 
		pool.join() 
	except:
		pass


	print("All threads finished at {}".format(datetime.datetime.now()))
	write_logfile(outfiles_rootdir, options, start_time)
