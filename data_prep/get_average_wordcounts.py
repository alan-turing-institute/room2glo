#!/usr/bin/python3

import os
import json
from collections import Counter
import argparse
import datetime

# convert json to tsv so can then read line by line instead of loading whole counter to mem.
# for filename in [x for x in os.listdir(infile_dir) if x[-4:] == 'json']:
# 	with open(infile_dir+filename, 'r') as infile:
# 		wcs = Counter(json.load(infile))
# 		with open(infile_dir+filename[:-4]+'tsv', 'w') as outfile:
# 			for i in wcs.most_common():
# 				outfile.write('\t'.join([i[0], str(i[1])])+'\n')
# 	print('done '+filename)

# now average the counts.

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
	parser.add_argument("-i", "--infiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/word_freqs/', help = "path to directory where corpus is stored")
	parser.add_argument("-o", "--outfile_dir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/word_freqs/', help = "path to directory where where output files containing average word counts should be written")
	options = parser.parse_args()

	infiles_rootdir = options.infiles_rootdir
	outfile_dir = options.outfile_dir

	os.makedirs(outfile_dir, exist_ok=True)


	all_wcs = Counter()
	n_time_periods = 0
	for filename in [x for x in os.listdir(infile_dir) if x[-3:] == 'tsv']:
		with open(infile_dir+filename, 'r') as infile:
			n_time_periods += 1
			for line in infile:
				(w, c) = line.strip().split('\t')
				c = int(c)
				if c >= 50:
					all_wcs[w] += c
				else:
					break
		print('done '+filename)

	for w in all_wcs:
	    all_wcs[w] /= n_time_periods

	with open(outfiledir+'/avg_word_counts.tsv', 'w') as outfile:
		for i in all_wcs.most_common():
			outfile.write('\t'.join([i[0], str(i[1])])+'\n')
	write_logfile(outfiledir+'/avg_word_counts.tsv', options, start_time)
