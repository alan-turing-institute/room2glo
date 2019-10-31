import os
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
	parser.add_argument("-i", "--infiles_rootdir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/word_freqs/', help = "path to directory where corpus is stored")
	parser.add_argument("-o", "--outfile_dir", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups_stats/type_and_token_counts/', help = "path to directory where where output files containing type and token counts should be written")
	options = parser.parse_args()

	infiles_rootdir = options.infiles_rootdir
	outfile_dir = options.outfile_dir


	start_time = datetime.datetime.now()

	os.makedirs(outfile_dir, exist_ok=True)


	for fn in os.listdir(infile_dir):
		if fn != 'avg_word_counts.tsv':
			n_tokens = 0
			n_types = 0
			with open(infile_dir + fn, 'r') as infile:
				for line in infile:
					n_types += 1
					token_count = int(line.strip().split('\t')[1])
					n_tokens += token_count
			with open(outfile_dir+fn, 'w') as outfile:
				outfile.write('types:\t{}\n'.format(n_types))
				outfile.write('tokens:\t{}\n'.format(n_tokens))
			write_logfile(outfile_dir+fn, options, start_time)
