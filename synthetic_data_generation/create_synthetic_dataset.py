import argparse
import datetime
import os
import multiprocessing
import json
import numpy as np
import gzip
import sys


# This script expects the input data to be in a gzipped file containing one tweet per line, 
# with tweets consisting of tab-separated fields, the text of the tweet being in the
# last field. To read from a corpus with a different format, modify or replace this class with 
# one of your own.
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
	"""
	Decide whether to replace an instance of a context_word with its corresponding pseudoword.
	"""

	# look up which pseudoword the current context_word is associated with.
	pseudoword = context_words[context_word]['pseudoword']
	
	# look up which schema the associated pseudoword belongs to.
	pseudoword_type = context_words[context_word]['pseudoword_type']
	
	# look up which 'set' the context_word belongs to (i.e. what 'kind' of context_word it is)
	# e.g. Schema C2 pseudowords have one or more context_words which gradually decrease in frequency over
	# time (set 1), and one or more context_words which gradually increase in frequency over time (set 2).
	set_number = context_words[context_word]['set_number']

	# Schema C3 pseudowords and Schema D4 pseudowords each have a set of context_words whose replacement probabilities at 
	# a given time-step are specified by a multinomial distibution drawn with a Dirichlet prior. 
	if pseudoword_type  == 'D4' or (pseudoword_type  == 'C3' and set_number == 1):	
		
		# So, if the current context_word is associated with a C3 or D4 pseudoword, we need to look up which index in 
		# the multinomial distribution corresponds to it.
		dist_index = context_words[context_word]['dist_index']
		
		# We then retrieve the pseudoword-insertion-probability, given the current context_word's set number, the 
		# current time-step, and the current context_word's index in the current time-step's multinomial distribution.
		p_insert = pseudoword_insert_probs[pseudoword]["p{}_array_series".format(set_number)][year_month_index][dist_index]

	else:
		try:
			# retrieve the pseudoword-insertion-probability given the current context_word's set number, and the 
			# current time-step.
			p_insert = pseudoword_insert_probs[pseudoword]["p{}_series".format(set_number)][year_month_index]
		
		except TypeError:
			print(pseudoword)
			print(context_word)
			print(pseudoword_insert_probs[pseudoword])
			raise()


			
			
	# set 'insert' to 1 with probability p_insert, or to 0 with probability 1-p_insert
	insert = np.random.RandomState().choice(2,1,p=[1-p_insert, p_insert])

	# if 'insert' was set to 1, we will replace the current context_word with its associated pseudoword
	if insert:
		return pseudoword
	
	# if 'insert' was set to 0, we will not replace the current context_word.
	else:
		return False



def create_synthetic(year_month_index, subsampling, subsample_percent):
	"""
	Create the synthetic data for a given timestep (year_month_index is simply an integer in the range (0, n_timesteps) 
	specifying which timestep we are at). 
	
	If subsampling is not set to 0, then lines will be randomly subsampled, keeping only
	subsample_percent of them.
	
	For each line that is kept, we identify instances of the words that were selected to represent
	senses of pseudowords, and with the probabilities specified in our pseudoword design dictionaries,
	we replace them with their corresponding pseudowords.
	"""
	

	year_month = year_months[year_month_index]

	# create the output file for the current time-step, and open it for writing
	with gzip.open(os.path.join(outfiles_rootdir, year_month[:4], year_month), 'wt') as outfile:

			# load the input data (which is the same for every time-step)
			tweets = TweetYielder([input_filepath])

			for tweet in tweets: #i.e. for each line of the input file..

				# if subsampling, discard the current line with probability 1 - subsample_percent
				if subsampling:
					if np.random.RandomState().random_sample() > subsample_percent*0.01:
						continue

				# if we didn't discard the current line, we will now go through one token at a time and
				# decide whether to replace each token with a pseudoword.
				tweet_out = []
				for i in range(len(tweet)):

					# if the current token is one of the words we have chosen to represent a pseudoword
					# 'sense'...
					if tweet[i] in context_words:
						
						# ...then we replace it with its associated pseudoword, with a probability 
						# specified by our pseudoword design dictionaries.
						pseudoword = maybe_return_pseudoword(tweet[i], year_month_index)
						if pseudoword:
							tweet_out.append(pseudoword)
						else:
  							tweet_out.append(tweet[i])
					else:
						tweet_out.append(tweet[i])
						
				# finally, we write this potentially-modified line to the synthetic data file for the current time-step.
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

	parser.add_argument("-i", "--input_filepath", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/2014/2014-12.csv.gz', help = "path to file in which data for the month we want to use is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/synthetic_evaluation_dataset/', help = "path to directory where synthetic dataset should be written")
	parser.add_argument("-c", "--context_word_dict_filepath", type=str, default='/data/synthetic_evaluation_dataset/context_word_dict.json', help = "path to file in which context word dict is stored")
	parser.add_argument("-p", "--pseudoword_dict_filepath", type=str, default='/data/synthetic_evaluation_dataset/pseudoword_dict.json', help = "path to file in which pseudoword dict is stored")

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
	
	
	
	
	
	
	# This script assumes you wish to model a corpus in which each timestep consists of data spanning a single month.
	# Hence, the number of timesteps in the synthetic dataset is determined on the basis of a specified start 
	# year & month and a specified end year & month. However, the granularity of the timesteps is not relevant for
	# the actual pseudoword generation procedure; this depends only on the *number* of timesteps you want to have in the 
	# synthetic dataset. So, you may wish to modify the script such that you can specify the number of timesteps directly,
	# and you may wish to modify the directory structure and names of the files that the synthetic data is written to.

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
	
	
	


	# load the pseudoword design dictionaries that were previously created using design_pseudowords.py.
	with open(context_word_dict_filepath, 'r') as infile:
		context_words = json.load(infile)

	with open(pseudoword_dict_filepath, 'r') as infile:
		pseudoword_insert_probs = json.load(infile)


		

	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine

	for year_month_index in range(n_timesteps): # i.e. for each timestep...

		# to run single-threaded, uncomment the following line and comment out the next 6.
		# create_synthetic(year_month_index, subsampling, subsampling_percent)
		
		pool.apply_async(create_synthetic, (year_month_index, subsampling, subsampling_percent), callback=success, error_callback=error)
	try:
		pool.close() 
		pool.join() 
	except:
		pass


	print("All threads finished at {}".format(datetime.datetime.now()))
	write_logfile(outfiles_rootdir, options, start_time)
