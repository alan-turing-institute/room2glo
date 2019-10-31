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

def write_logfile(outfilepath, options, start_time):
	logfile_path = outfilepath + '.log'
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Options used:-\n')
		for (option, value) in vars(options).items():
			logfile.write('{}\t{}\n'.format(option,value))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--infiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/', help = "path to directory where corpus is stored")
	parser.add_argument("-o", "--outfile_dir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups_stats/', help = "path to directory where where output file containing tweet counts should be written")
	options = parser.parse_args()

	start_time = datetime.datetime.now()
	
	infiles_rootdir = options.infiles_rootdir
	outfile_dir = options.outfile_dir


	os.makedirs(outfile_dir, exist_ok=True)

	total_tweet_count = 0

	with open(outfile_dir+'/tweet_counts.tsv', 'w') as outfile:
		for year in range(2011,2019):
			for month in range(1,13):
				year_month = "{}-{:02}".format(year,month)

				monthly_tweet_count = 0
				

				infilepath = "{}/{}/{}-{:02}.csv.gz".format(infiles_rootdir,year,year,month)
				
				try:
					with gzip.open(infilepath,"rt") as infile:
						for line in infile:

							monthly_tweet_count += 1
							total_tweet_count += 1

				except:
					pass

				else:
					outfile.write('\t'.join(["{}-{:02}".format(year,month),str(monthly_tweet_count)])+'\n')
					print("done {}-{:02}".format(year,month))

		outfile.write('\t'.join(["total",str(total_tweet_count)])+'\n')
	
	write_logfile(outfile_dir+'/tweet_counts.tsv', options, start_time)
