from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import Counter
import argparse
import datetime
import gzip
import os



# import string

# stopWords = set(stopwords.words('english'))

# # for ignoring single letters:
# singlechars = set(string.ascii_letters)
# singlechars -= set('aiu')
# singlechars |= set('#_')

# def is_int(s):
# 	try:
# 		int(s)
# 		return True
# 	except ValueError:
# 		return False





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


# class TweetYielder(object):
# 	def __init__(self, filepaths):
# 		self.filepaths = filepaths

# 	def __iter__(self):
# 		for filepath in self.filepaths:									   
# 			for line in gzip.open(filepath,"rt"):
# 				line = line.strip().split('\t')

# 				if int(line[4]) > 90:
# 					tweet = [w for w in line[-1].split() if w not in singlechars and w not in stopWords]
# 					for i in range(len(tweet)):
# 						if is_int(tweet[i]):
# 							tweet[i] = '<NUM>'

# 				yield tweet
# 			print("Passed through {}".format(filepath))


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
	parser.add_argument("-i", "--input_filepath", type=str, default='/data/twitter_spritzer/cleaner_001p_nodups/2014/2014-12.csv.gz', help = "path to file where data for the month we want to use is stored")
	parser.add_argument("-o", "--output_dir", type=str, default='/data/twitter_spritzer/synthetic_evaluation_dataset/', help = "path to file where word frequencies and wordnet stats should be written")

	# parser.add_argument("-i", "--input_filepath", type=str, default='data/clean_001p_local_nodups/2014/2014-04.csv.gz', help = "path to file where data for the month we want to use is stored")
	# parser.add_argument("-o", "--output_dir", type=str, default='data/synthetic_evaluation_dataset/', help = "path to directory where word frequencies and wordnet stats should be written")


	options = parser.parse_args()

	start_time = datetime.datetime.now()
	print('Starting at: {}\n'.format(start_time))


	input_filepath = options.input_filepath
	output_filepath = os.path.join(options.output_dir, 'vocab_stats.tsv')

	os.makedirs(options.output_dir,exist_ok=True)

	tweets = TweetYielder([input_filepath])
	freqs = Counter()

	for tweet in tweets:
		freqs.update(tweet)

	print('Got freqs at: {}\n'.format(datetime.datetime.now()))

	with open(output_filepath, 'w') as outfile:

		for item in freqs.most_common():
			(word, freq) = item
			if freq < 50:
				break
			senses = wn.synsets(word)
			hypernyms = []
			hyponyms = []
			for sense in senses:
				hypernyms.extend(sense.hypernyms())
				hyponyms.extend(sense.hyponyms())

			n_senses = len(senses)
			n_hypernyms = len(set(hypernyms))
			n_hyponyms = len(set(hyponyms))

			senses = ' '.join(str(x) for x in senses)
			hypernyms = ' '.join(str(x) for x in set(hypernyms))
			hyponyms = ' '.join(str(x) for x in set(hyponyms))
			outfile.write('\t'.join([word, str(freq), str(n_senses), senses, str(n_hypernyms), hypernyms, str(n_hyponyms), hyponyms])+'\n')

	print('Done at: {}. Writing log file...\n'.format(datetime.datetime.now()))
	write_logfile(output_filepath, options, start_time)