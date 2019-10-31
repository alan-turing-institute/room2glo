
#!/usr/bin/python3

import os
import gzip
import random
import multiprocessing
import sys
import datetime
import glob
from collections import Counter
import string

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


def process_yearmonth(year,month):

	year_month = "{}-{:02}".format(year,month)
	infilepath = "{}/{}/{}-{:02}.csv.gz".format(infiles_rootdir,year,year,month)
	
	try:
		with gzip.open(infilepath,"rt") as infile:
			with gzip.open("{}/{}/{}-{:02}.csv.gz".format(outfile_dir,year,year,month), 'wt') as outfile:
				for line in infile:

					line = line.strip().split('\t')

					if int(line[4]) > 90:
						tweet = [w for w in line[-1].split() if w not in singlechars]
						for i in range(len(tweet)):
							if is_int(tweet[i]):
								tweet[i] = '<NUM>'


						line = line[:-1]
						line.append(' '.join(tweet))

						outfile.write('\t'.join(line)+'\n')

				
	except FileNotFoundError:
		pass

	return year_month

def success(infilepath):
	sys.stdout.write("{} completed at {}\n".format(infilepath,datetime.datetime.now()))
	sys.stdout.flush()

def error(a):
	sys.stderr.write("\nError: {}\n\n".format(a))
	sys.stderr.flush()


def write_logfile(infiles_rootdir, outfiles_rootdir, start_time):
	logfile_path = '/'.join([outfile_rootdir, 'corpus_provenance.log'])
	with open(logfile_path, 'w') as logfile:
		logfile.write('Script started at: {}\n\n'.format(start_time))
		logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
		logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
		logfile.write('Original corpus located at: {}\n\n'.format(infiles_rootdir))
		logfile.write('Cleaned corpus written to:{}\n\n').format(outfiles_rootdir)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--infiles_rootdir", type=str, default="/data/twitter_spritzer/clean_001p_nodups/", help = "path to directory where original cleaned corpus is stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/', help = "path to directory where where extra-cleaned corpus should be written")
	options = parser.parse_args()

	pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine

	start_time = datetime.datetime.now()
	print("Started at {}\n".format(start_time))
	
	for year in range(2011,2019):
		os.makedirs('/'.join([outfile_dir,str(year)]), exist_ok=True)
		for month in range(1,13):
			#process_yearmonth(year,month)
			pool.apply_async(process_yearmonth, (year,month), callback=success, error_callback=error)

	try:
		pool.close() 
		pool.join() 
	except:
		pass	


	print("All threads finished at {}\n".format(datetime.datetime.now()))
	write_logfile(options.infiles_rootdir, options.outfiles_rootdir, start_time)
	print("Written log. All done!")
