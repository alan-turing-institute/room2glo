#!/usr/bin/python3

import os
import gzip
import random
import multiprocessing
import sys
import datetime
import glob
from collections import Counter
import argparse
import datetime


def process_yearmonth(year,month, options, start_time):

	year_month = "{}-{:02}".format(year,month)

	word_counts = Counter()

	infilepath = "{}/{}/{}-{:02}.csv.gz".format(infiles_rootdir,year,year,month)
	
	try:
		with gzip.open(infilepath,"rt") as infile:
			for line in infile:

				line = line.strip().split('\t')
				words = line[-1].split()
				word_counts.update(words)
	except:
		pass

	if word_counts:
		with open(outfile_dir+'/word_counts_{}.tsv'.format(year_month), 'w') as outfile:
			for i in word_counts.most_common():
				outfile.write('\t'.join([i[0], str(i[1])])+'\n')
		write_logfile(outfile_dir+'/word_counts_{}.tsv'.format(year_month), options, start_time)

	return year_month

def success(infilepath):
	sys.stdout.write("{} completed at {}\n".format(infilepath,datetime.datetime.now()))
	sys.stdout.flush()

def error(a):
	sys.stderr.write("\nError: {}\n\n".format(a))
	sys.stderr.flush()


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
	parser.add_argument("-i", "--infiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/', help = "path to directory where corpus is stored")
	parser.add_argument("-o", "--outfile_dir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups_stats/word_freqs/', help = "path to directory where where output file containing word frequencies should be written")
	options = parser.parse_args()

	infiles_rootdir = options.infiles_rootdir
	outfile_dir = options.outfile_dir

	os.makedirs(outfile_dir, exist_ok=True)

	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine

	start_time = datetime.datetime.now()
	print("Started at {}\n".format(start_time))
	for year in range(2011,2019):
		for month in range(1,13):
			pool.apply_async(process_yearmonth, (year,month, options, start_time), callback=success, error_callback=error)

	try:
		pool.close() 
		pool.join() 
	except:
		pass	

	print("All threads finished at {}\n".format(datetime.datetime.now()))
