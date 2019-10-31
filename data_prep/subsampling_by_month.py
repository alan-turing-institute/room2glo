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
				yield line
			print("Passed through {}".format(filepath))

def subsample_month(year_month_file, subsample_number, subsample_percent):

	os.makedirs(os.path.join(outfiles_rootdir,  str(subsample_number), year_month_file[-14:-10]), exist_ok=True)

	outfile_path = os.path.join(outfiles_rootdir,  str(subsample_number), year_month_file[-14:-10], year_month_file[-14:])

	with gzip.open(outfile_path, 'wt') as outfile:

				tweets = TweetYielder([year_month_file])

				for tweet in tweets:

					if np.random.random_sample() > subsample_percent*0.01:
						continue
					else:
						outfile.write(tweet)

	return year_month_file


def success(year_month_file):
	sys.stdout.write("{} completed at {}\n".format(year_month_file,datetime.datetime.now()))
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


	parser.add_argument("-i", "--infiles_rootdir", type=str, default="/data/twitter_spritzer/synthetic_evaluation_dataset/subsampled_70/", help = "path to directory where original corpus is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/synthetic_evaluation_dataset/subsampled_70/subsamples/', help = "path to directory where subsamples should be written")

	parser.add_argument("-sp", "--subsampling_percent", type=int, default=70, help = "size of sample (percent of original data) e.g. 70")
	parser.add_argument("-n", "--n_subsamples", type=int, default=10, help = "number of subsamples to draw per month")

	parser.add_argument("-sy", "--start_year", type=int, default = 2012, help="start year: integer, e.g. 2012")
	parser.add_argument("-sm", "--start_month", type=int, default = 1, help="start month: integer, e.g. 6")
	parser.add_argument("-ey", "--end_year", type=int, default = 2017, help="end year: integer, e.g. 2014")
	parser.add_argument("-em", "--end_month", type=int, default = 6, help="end month: integer, e.g. 4")

	options = parser.parse_args()


	subsampling_percent = options.subsampling_percent
	n_subsamples = options.n_subsamples

	infiles_rootdir = options.infiles_rootdir
	outfiles_rootdir = options.outfiles_rootdir + '/{}_{}_percent_samples/'.format(n_subsamples,subsampling_percent)
	
	start_year = options.start_year
	end_year = options.end_year
	start_month = options.start_month
	end_month = options.end_month

	start_time = datetime.datetime.now()
	print('Starting at: {}\n\n'.format(start_time))



	year_month_files = []

	for year in range(start_year, end_year+1):

		for month in range(1,13):
			if year == start_year and month < start_month:
				continue
			elif year == end_year and month > end_month:
				break

			year_month_file = os.path.join(infiles_rootdir, str(year), "{}-{:02}.csv.gz".format(year, month))
			if os.path.isfile(year_month_file):
				year_month_files.append(year_month_file)



	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine

	for subsample_number in range(n_subsamples):

		for year_month_file in year_month_files:

			pool.apply_async(subsample_month, (year_month_file, subsample_number, subsampling_percent), callback=success, error_callback=error)

	try:
		pool.close() 
		pool.join() 
	except:
		pass


	print("All threads finished at {}".format(datetime.datetime.now()))
	write_logfile(outfiles_rootdir, options, start_time)
