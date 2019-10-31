#!/usr/bin/python3

import os
import gzip
import random
import multiprocessing
import sys
import datetime
import json
from collections import Counter
import argparse

def write_logfile(outfilepath, options, start_time):
	logfile_path = outfilepath + '.log'
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Options used:- \n')
		for (option, value) in vars(options).items():
			logfile.write('{}\t{}\n'.format(option,value))


def process_file(file_input):

	(filename,root) = file_input

	infilepath = os.path.join(root,filename)

	year_month = filename[:7]

	tweet_counts = Counter()
	non_RT_counts = Counter()
	confident_english_counts = Counter()
	confident_english_and_non_RT_counts = Counter()

	cld_eng_percents = Counter()
	user_tweet_counts = Counter()
	
	
	with gzip.open(infilepath, 'rt') as infile:
		for line in infile:
			tweet_counts[year_month] += 1

			tweet = line.strip().split('\t')

			user_id = int(tweet[1])
			user_tweet_counts[user_id] += 1

			cld_eng_percent = int(tweet[4])
			if cld_eng_percent > eng_threshold:
				confident_english = True
			else:
				confident_english = False
			cld_eng_percents[cld_eng_percent] += 1


			if int(tweet[3]):
				RT = True
			else:
				RT = False
			if not RT:
				non_RT_counts[year_month] += 1
				if confident_english:
					confident_english_and_non_RT_counts[year_month] += 1
					confident_english_counts[year_month] += 1
			elif confident_english:
				confident_english_counts[year_month] += 1

	return (tweet_counts, non_RT_counts, confident_english_counts, confident_english_and_non_RT_counts, cld_eng_percents, user_tweet_counts)





if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--infiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/', help = "path to directory where corpus is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups_stats/', help = "path to directory where where output files containing descriptive stats should be written")
	parser.add_argument("-t", "--eng_threshold", type=int, default=90, help = "The minimum CLD confidence score for English that a tweet must have in order for it to be classified as one we are confident is English.")
	options = parser.parse_args()

	
	infiles_rootdir = options.infiles_rootdir
	outfile_dir = options.outfiles_rootdir
	eng_threshold = options.eng_threshold


	start_time = datetime.datetime.now()
	print("Started at {}\n".format(start_time))

	# will aggregate over months:
	master_tweet_counts = Counter()
	master_non_RT_counts = Counter()
	master_confident_english_counts = Counter()
	master_confident_english_and_non_RT_counts = Counter()

	# will aggregate over whole dataset:
	master_cld_eng_percents = Counter()
	master_user_tweet_counts = Counter()


	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine
	for root, subdirs, filenames in os.walk(infiles_rootdir):
		
		
		subdirs[:] = [d for d in subdirs if d != 'duplicates']

		inputs = [(f, root) for f in filenames if f != 'md5sums' and f != '.DS_Store' and f != 'corpus_provenance.log']
		
		try:
			print("Starting year {} at {}".format(inputs[0][0][:4],datetime.datetime.now()))
		except IndexError:
			pass
		
		pool_outputs = pool.map(process_file, inputs)
		
		for p in pool_outputs:
			master_tweet_counts += p[0]
			master_non_RT_counts += p[1]
			master_confident_english_counts += p[2]
			master_confident_english_and_non_RT_counts += p[3]
			master_cld_eng_percents += p[4]
			master_user_tweet_counts += p[5]

	try:
		pool.close() 
		pool.join() 
	except:
		pass	


	print("All threads finished at {}\n".format(datetime.datetime.now()))

	print("Writing output and logfiles...")

	os.makedirs(outfile_dir, exist_ok=True)

	with open(outfile_dir+'/tweet_counts.json', 'w') as outfile:
		json.dump(master_tweet_counts, outfile)
	write_logfile(outfile_dir+'/tweet_counts.json', options, start_time)

	with open(outfile_dir+'/non_RT_counts.json', 'w') as outfile:
		json.dump(master_non_RT_counts, outfile)
	write_logfile(outfile_dir+'/non_RT_counts.json', options, start_time)

	with open(outfile_dir+'/confident_english_counts.json', 'w') as outfile:
		json.dump(master_confident_english_counts, outfile)
	write_logfile(outfile_dir+'/confident_english_counts.json', options, start_time)

	with open(outfile_dir+'/confident_english_and_non_RT_counts.json', 'w') as outfile:
		json.dump(master_confident_english_and_non_RT_counts, outfile)
	write_logfile(outfile_dir+'/confident_english_and_non_RT_counts.json', options, start_time)

	with open(outfile_dir+'/cld_eng_percents.json', 'w') as outfile:
		json.dump(master_cld_eng_percents, outfile)
	write_logfile(outfile_dir+'/cld_eng_percents.json', options, start_time)

	with open(outfile_dir+'/user_tweet_counts.json', 'w') as outfile:
		json.dump(master_user_tweet_counts, outfile)
	write_logfile(outfile_dir+'/user_tweet_counts.json', options, start_time)

	print("All done! Output written at {}\n".format(datetime.datetime.now()))
