#!/usr/bin/env python3

import zlib
import re
import datetime
import os
import sys
import multiprocessing
import glob
import gzip
from collections import defaultdict
import argparse
import string

pattern = re.compile('[\W_]+')

# for ignoring single letters:
singlechars = set(string.ascii_letters)
singlechars -= set('aiu')
singlechars |= set('#_')


def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False


def preprocess_text(tweet_text):
	tweet_text = pattern.sub('', tweet_text) # remove all non-alphanumeric chars (i.e. whitespace, punctuation..)
	if tweet_text[:2] == 'rt':
		tweet_text = tweet_text[2:] # if tweet starts with 'rt', remove that
	return tweet_text


def extra_cleaning(line):
	line = line.strip().split('\t')
	if int(line[4]) > 90:
		tweet = [w for w in line[-1].split() if w not in singlechars]
		for i in range(len(tweet)):
			if is_int(tweet[i]):
				tweet[i] = '<NUM>'
		line = line[:-1]
		line.append(' '.join(tweet))
	line = '\t'.join(line)
	return line

# def test_collisons(tweet_text, fingerprint, tweet_dict, hashtable): 
def test_collisons(tweet_text, fingerprint, hashtable): 
	# tweet_text = tweet_dict[tweet_id]
	is_dup = False


	for tweet_text2 in hashtable[fingerprint]:
		if tweet_text == tweet_text2:
			# tweet_text is exact duplicate of tweet_text2

			# print('\n\nDUPLICATE TWEETS:\n')
			# print(tweet_text)
			# print(tweet_text2)

			is_dup = True

	# for tweet_id2 in hashtable[fingerprint]:
	# 	tweet_text2 = tweet_dict[tweet_id2]
	# 	if tweet_text == tweet_text2:
	# 		# tweet_text is exact duplicate of tweet_text2

	# 		# print('\n\nDUPLICATE TWEETS:\n')
	# 		# print(tweet_text)
	# 		# print(tweet_text2)

	# 		is_dup = True

	# return (is_dup, tweet_dict, hashtable)
	return (is_dup, hashtable)


def deduplicate_month(infiles_rootdir, outfiles_rootdir, year_month, year, month):

	# tweet_dict = {}
	hashtable = defaultdict(set)
	n_dups = 0
	n_tweets = 0

	outfilepath = "{}/{}/{}.csv.gz".format(outfiles_rootdir,year,year_month)
	dups_outfilepath = "{}/duplicates/{}/{}.csv.gz".format(outfiles_rootdir,year,year_month)
	files = glob.glob("{}/{}/{}-*".format(infiles_rootdir,year,year_month))
	if files:
		with gzip.open(outfilepath, 'wt') as outfile:
			with gzip.open(dups_outfilepath, 'wt') as dups_outfile:
				# print(year_month)
				for infilepath in files:
					# print(infilepath)
					with gzip.open(infilepath, 'rt') as infile:
						for line in infile:
							n_tweets += 1
							tweet = line.strip().split('\t')

							#if int(tweet[3]): # if tweet is a retweet:


							# tweet_id = int(tweet[2])
							try:
								tweet_text = preprocess_text(tweet[5])
							except IndexError:
								sys.stderr.write('INDEX ERROR: ' + str(tweet)+'\n')
								continue


							# tweet_dict[tweet_id] = tweet_text
							fingerprint = zlib.adler32(tweet_text.encode('utf-8'))
							if hashtable[fingerprint]:
								# (is_dup, tweet_dict, hashtable) = test_collisons(tweet_text, fingerprint, tweet_dict, hashtable)
								(is_dup, hashtable) = test_collisons(tweet_text, fingerprint, hashtable)
								if is_dup:
									n_dups += 1
									#dups_outfile.write(line)
									print(line, file=dups_outfile)
								else: 
									# hashtable[fingerprint].add(tweet_id)			
									# tweet_dict[tweet_id] = tweet_text	
									hashtable[fingerprint].add(tweet_text)
									# false collision, so not a duplicate, so write out tweet.
									line = extra_cleaning(line)
									#outfile.write(line)
									print(line, file=outfile)
							else:
								# hashtable[fingerprint].add(tweet_id)
								# tweet_dict[tweet_id] = tweet_text	
								hashtable[fingerprint].add(tweet_text)			
								# no collision, so not a duplicate, so write out tweet.
								line = extra_cleaning(line)
								#outfile.write(line)
								print(line, file=outfile)


							# else:
							# 	# not a retweet, so write out tweet.
							# 	outfile.write(line)

	sys.stdout.write('{}: {} duplicates detected out of {} tweets\n'.format(year_month, n_dups, n_tweets))
	return year_month


def success(year_month): 
	sys.stdout.write("{} completed at {}\n".format(year_month, datetime.datetime.now()))
	sys.stdout.flush()


def error(a):
	sys.stderr.write("\nError: {}\n\n".format(a))
	sys.stderr.flush()


def write_logfile(infiles_rootdir, outfiles_rootdir, start_time):
	logfile_path = '/'.join([outfiles_rootdir, 'corpus_provenance.log'])
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Original corpus located at: {}\n\n'.format(infiles_rootdir))
		logfile.write('De-duplicated corpus written to:{}\n\n'.format(outfiles_rootdir))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--infiles_rootdir", type=str, default="/data/twitter_spritzer/clean_001p/", help = "path to directory where original corpus is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/clean_001p_nodups/', help = "path to directory where where de-duplicated corpus should be written")
	options = parser.parse_args()


	pool = multiprocessing.Pool(processes=32) 

	start_time = datetime.datetime.now()
	sys.stdout.write("Started at {}\n".format(start_time))

	for year in range(2011,2019):
	# for year in range(2012,2015):
		os.makedirs('/'.join([options.outfiles_rootdir, str(year)]),exist_ok=True) #Create all needed dirs, no error if they already exist
		os.makedirs('/'.join([options.outfiles_rootdir, 'duplicates', str(year)]),exist_ok=True) #Create all needed dirs, no error if they already exist
		for month in range(1,13):
		# for month in range(1,5):
			year_month = "{}-{:02}".format(year,month)
			pool.apply_async(deduplicate_month, (options.infiles_rootdir, options.outfiles_rootdir, year_month, year, month), callback=success, error_callback=error)

	try:
		pool.close() 
		pool.join() 
	except:
		pass


	print("All threads finished. Writing log file...")
	write_logfile(options.infiles_rootdir, options.outfiles_rootdir, start_time)
	print("Written log. All done!")

	# single threaded, for testing:

	# print("Started at {}\n".format(datetime.datetime.now()))
	# for year in range(2011,2019):
	# # for year in range(2012,2015):
	# 	os.makedirs('/'.join([options.outfiles_rootdir, str(year)]),exist_ok=True) #Create all needed dirs, no error if they already exist
	# 	#for month in range(1,13):
	# 	for month in range(1,5):
	# 		year_month = "{}-{:02}".format(year,month)
	# 		deduplicate_month(options.infiles_rootdir, options.outfiles_rootdir, year_month, year, month)


