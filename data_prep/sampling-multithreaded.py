#!/usr/bin/python3

import os
import gzip
import random
import multiprocessing
import sys
import datetime
import argparse


def sample_10p(infilepath, filename, outfiles_rootdir):
	outfilepath = '/'.join([outfiles_rootdir, filename[:4], filename])
	with gzip.open(infilepath, 'rt') as infile:
		with gzip.open(outfilepath, 'wt') as outfile:
		#Suggest "w" rather than "a" -- if the file already exists, we want to overwrite (or skip entirely) but probably not append
			for line in infile:
				i = random.random()
				if i < 0.1:
					outfile.write(line)
	return infilepath


def success(infilepath):
	sys.stdout.write("{} completed at {}\n".format(infilepath,datetime.datetime.now()))
	sys.stdout.flush()


def error(a):
	sys.stdout.write("\nError: {}\n\n".format(a))
	sys.stdout.flush()


def write_logfile(infiles_rootdir, outfiles_rootdir, start_time):
	logfile_path = '/'.join([outfile_rootdir, 'corpus_provenance.log'])
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Original corpus located at: {}\n\n'.format(infiles_rootdir))
		logfile.write('10\% sample corpus written to:{}\n\n').format(outfiles_rootdir)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--infiles_rootdir", type=str, default="/data/twitter_spritzer/01p/", help = "path to directory where original corpus is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/001p/', help = "path to directory where 10\% sample corpus should be written")
	options = parser.parse_args()

	start_time = datetime.datetime.now()

	
	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine
	for root, subdirs, filenames in os.walk(options.infiles_rootdir):
		for subdir in subdirs:
			print(subdir)
			os.makedirs('/'.join([options.outfiles_rootdir, subdir]),exist_ok=True) #Create all needed dirs, no error if they already exist
		for filename in [f for f in filenames if f != 'md5sums']:
			infilepath = os.path.join(root,filename)
			#sample_10p(infilepath, filename, options.outfiles_rootdir)
			#print('Done {}'.format(infilepath))
			pool.apply_async(sample_10p, (infilepath, filename, options.outfiles_rootdir), callback=success, error_callback=error)

	try:
		pool.close() 
		pool.join() 
	except:
		pass	

	print("All threads finished. Writing log file...")
	write_logfile(options.infiles_rootdir, options.outfiles_rootdir, start_time)
	print("Written log. All done!")
	
