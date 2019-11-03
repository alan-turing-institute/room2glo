import pandas
import numpy as np
import json
from collections import defaultdict
import argparse
import datetime
import os


def sample_single_context_word(freq_bin_df):

	if len(freq_bin_df) < 1:
		return (freq_bin_df, None)
	else:
		s = freq_bin_df.sample(n=1)
		freq_bin_df = freq_bin_df.drop(s.index)
		return(freq_bin_df, list(s['word']))


def sample_context_word_group(freq_bin_df, n_words):

	if len(freq_bin_df) < n_words:
		return (freq_bin_df, None)
	else:
		stop = False
		while stop == False:
			s = freq_bin_df.sample(n=n_words)
			sense_sets = []
			for i in range(n_words):
				sense_sets.append(set(s.iloc[i]['senses'].split(' ')))
			if not set.intersection(*sense_sets):
				stop = True  
		freq_bin_df = freq_bin_df.drop(s.index)
		return(freq_bin_df, list(s['word']))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




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

	parser.add_argument("-i", "--input_filepath", type=str, default='/data/synthetic_evaluation_dataset/vocab_stats.tsv', help = "path to file where vocab stats for the month we want to use are stored")
	parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/synthetic_evaluation_dataset/', help = "path to directory where pseudoword design info should be written")

	parser.add_argument("-sy", "--start_year", type=int, default = 2012, help="start year: integer, e.g. 2012")
	parser.add_argument("-sm", "--start_month", type=int, default = 1, help="start month: integer, e.g. 6")
	parser.add_argument("-ey", "--end_year", type=int, default = 2017, help="end year: integer, e.g. 2014")
	parser.add_argument("-em", "--end_month", type=int, default = 6, help="end month: integer, e.g. 4")
	parser.add_argument("-mf", "--min_freq", default = 100, help="minimum frequency for context words")
	parser.add_argument("-ms", "--max_n_senses", default = 10, help="maximum number of senses for context words")

	options = parser.parse_args()


	input_filepath = options.input_filepath
	outfiles_rootdir = options.outfiles_rootdir

	os.makedirs(outfiles_rootdir,exist_ok=True)

	start_year = options.start_year
	end_year = options.end_year
	start_month = options.start_month
	end_month = options.end_month
	max_n_senses = options.max_n_senses
	min_freq = options.min_freq

	start_time = datetime.datetime.now()
	print('Starting at: {}\n'.format(start_time))


	year_months = []
	for year in range(start_year, end_year+1):

		for month in range(1,13):
			if year == start_year and month < start_month:
				continue
			elif year == end_year and month > end_month:
				break
			
			year_month = "{}-{:02}".format(year, month)
			year_months.append(year_month)

	n_timesteps = len(year_months)


	df = pandas.read_csv(input_filepath, sep='\t', header=None, names=['word', 'freq', 'n_senses','senses','n_hypernyms','hypernyms','n_hyponyms','hyponyms'])
	

	# only consider words which have at least one sense in WordNet, and which are more than 2 letters long, and which have no more than max_n_senses, and which occur at least min_freq times.
	df = df[df['n_senses'] > 0]
	df = df[df['n_senses'] <= options.max_n_senses]
	df = df[df['word'].str.len() > 2]
	df = df[df['freq'] > options.min_freq]


	# split into 5 equally-sized frequency bins
	freqs_array = df['freq']
	(ser, freq_bins) = pandas.qcut(freqs_array, 5, retbins=True)
	del ser

	bin1_df = df[df['freq'] >= freq_bins[0]]
	bin1_df  = bin1_df[bin1_df['freq'] < freq_bins[1]]
	print(len(bin1_df))

	bin2_df  = df[df['freq'] >= freq_bins[1]]
	bin2_df  = bin2_df[bin2_df['freq'] < freq_bins[2]]
	print(len(bin2_df))

	bin3_df = df[df['freq'] >= freq_bins[2]]
	bin3_df  = bin3_df[bin3_df['freq'] < freq_bins[3]]
	print(len(bin3_df))

	bin4_df  = df[df['freq'] >= freq_bins[3]]
	bin4_df = bin4_df[bin4_df['freq'] < freq_bins[4]]
	print(len(bin4_df))

	bin5_df = df[df['freq'] >= freq_bins[4]]
	print(len(bin5_df))

	freq_bin_dfs = [bin1_df, bin2_df, bin3_df, bin4_df, bin5_df]
	print("got freq_bin_dfs. bins: {} -- {}".format(freq_bins, datetime.datetime.now()))

	
	

	# pseudoword insertion probability time series
	
	constant_prob_array = np.array([0.7]*n_timesteps)
	
	n_zeros = int(np.round(n_timesteps / 5, 0))
	n_ones = n_zeros
	
	increasing_prob_arrays = [np.linspace(0.1,1,n_timesteps), np.logspace(-1,0,n_timesteps), np.concatenate((np.zeros(n_zeros), np.linspace(0.1,1,n_timesteps-n_zeros))), np.concatenate((np.zeros(n_zeros), np.logspace(-1,0,n_timesteps-n_zeros))), np.concatenate((np.linspace(0.1,1,n_timesteps-n_ones), (np.ones(n_ones)))), np.concatenate((np.logspace(-1,0,n_timesteps-n_ones), (np.ones(n_ones))))]
	decreasing_prob_arrays = [np.flip(x,0) for x in increasing_prob_arrays]
	
	spike1 = np.concatenate((np.linspace(0.1,1,3),np.linspace(1,0.1,3)[1:]))
	spike2 = np.concatenate((np.logspace(-1,0,3),np.logspace(0,-1,3)[1:]))
	spiky_arrays = []
	# this will break if n_timesteps is too small...
	for i in [int(i) for i in np.linspace(0, n_timesteps-5, 6)]:
		spiky_arrays.append(np.concatenate((np.array([0.1]*i), spike1, np.array([0.1]*(n_timesteps-5-i)))))
		spiky_arrays.append(np.concatenate((np.array([0.1]*i), spike2, np.array([0.1]*(n_timesteps-5-i)))))

	periodic_arrays = []
	for month in ['01', '03', '05', '07', '09', '11']:
		periodic_array1 = np.array([0.1]*n_timesteps)
		periodic_array2 = np.array([0.1]*n_timesteps)
		for t in range(n_timesteps):
			if year_months[t] [-2:] == month:
				periodic_array1[t] = spike1[2]
				periodic_array2[t] = spike2[2]
				try:
					periodic_array1[t-1] = spike1[1]
					periodic_array2[t-1] = spike2[1]
				except IndexError:
					pass
				try:
					periodic_array1[t-2] = spike1[0]
					periodic_array2[t-2] = spike2[0]
				except IndexError:
					pass
				try:
					periodic_array1[t+1] = spike1[3]
					periodic_array2[t+1] = spike2[3]
				except IndexError:
					pass
				try:
					periodic_array1[t+2] = spike1[4]
					periodic_array2[t+2] = spike2[4]
				except IndexError:
					pass

		periodic_arrays.append(periodic_array1)
		periodic_arrays.append(periodic_array2)

	print("made arrays -- {}".format(datetime.datetime.now()))

	
	
	# now we sample 'context words' to represent different senses of the pseudowords
	# at the moment, for simplicity, we keep all sets of context words to cardinality 1, except for in schemas C3 & D4 where we have have one with cardinality 1 and one with cardinality 10.

	pseudoword_dict = defaultdict(lambda:defaultdict(dict))
	context_word_dict = defaultdict(lambda:defaultdict(dict))

	max_n_pseudowords_per_freq_bin = 6
	pseudoword_number = 1

	for freq_bin_number in range(len(freq_bin_dfs)):

		print("\n\nbin number: {} -- {}".format(freq_bin_number, datetime.datetime.now()))


		for pseudoword_number in range(max_n_pseudowords_per_freq_bin):

			print("\npseudoword number: {} -- {}".format(pseudoword_number,datetime.datetime.now()))

			# Schema D1:
			(new_df, sampled_context_words) = sample_single_context_word(freq_bin_dfs[freq_bin_number])
			if sampled_context_words:
				print(sampled_context_words)
				
				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type1_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':sampled_context_words, 'type': 1, 'p1_series': increasing_prob_arrays[pseudoword_number]}

				context_word_dict[sampled_context_words[0]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 1, 'set_number':1}

			else:
				# bin is empty, so we'll break out of the loop and move on to the next freq_bin
				break

			# Schema C1:
			(new_df, sampled_context_words) = sample_context_word_group(freq_bin_dfs[freq_bin_number], 2)
			if sampled_context_words:
				print(sampled_context_words)

				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type2_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':[sampled_context_words[0]], 'S2_words':[sampled_context_words[1]], 'type': 2, 'p1_series': constant_prob_array, 'p2_series': increasing_prob_arrays[pseudoword_number]}

				context_word_dict[sampled_context_words[0]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 2, 'set_number':1}

				context_word_dict[sampled_context_words[1]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 2, 'set_number':2}

			else:
				# not enough words in bin to take a sample, but perhaps there might be enough to take a smaller sample required for a different regime, so we'll carry on.
				pass

			# Schema D2:
			(new_df, sampled_context_words) = sample_context_word_group(freq_bin_dfs[freq_bin_number], 2)
			if sampled_context_words:
				print(sampled_context_words)

				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type3_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':[sampled_context_words[0]], 'S2_words':[sampled_context_words[1]], 'type': 3, 'p1_series': constant_prob_array, 'p2_series': spiky_arrays[pseudoword_number]}

				context_word_dict[sampled_context_words[0]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 3, 'set_number':1}

				context_word_dict[sampled_context_words[1]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 3, 'set_number':2}

			else:
				# not enough words in bin to take a sample, but perhaps there might be enough to take a smaller sample required for a different regime, so we'll carry on.
				pass

			# Schema D3:
			(new_df, sampled_context_words) = sample_context_word_group(freq_bin_dfs[freq_bin_number], 2)
			if sampled_context_words:
				print(sampled_context_words)
				
				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type4_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':[sampled_context_words[0]], 'S2_words':[sampled_context_words[1]], 'type': 4, 'p1_series': constant_prob_array, 'p2_series': periodic_arrays[pseudoword_number]}

				context_word_dict[sampled_context_words[0]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 4, 'set_number':1}

				context_word_dict[sampled_context_words[1]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 4, 'set_number':2}

			else:
				# not enough words in bin to take a sample, but perhaps there might be enough to take a smaller sample required for a different regime, so we'll carry on.
				pass


			# Schema C2:
			(new_df, sampled_context_words) = sample_context_word_group(freq_bin_dfs[freq_bin_number], 2)
			if sampled_context_words:
				print(sampled_context_words)
				
				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type5_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':[sampled_context_words[0]], 'S2_words':[sampled_context_words[1]], 'type': 5, 'p1_series':decreasing_prob_arrays[pseudoword_number], 'p2_series': increasing_prob_arrays[pseudoword_number]}

				context_word_dict[sampled_context_words[0]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 5, 'set_number':1}

				context_word_dict[sampled_context_words[1]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 5, 'set_number':2}

			else:
				# not enough words in bin to take a sample, but perhaps there might be enough to take a smaller sample required for a different regime, so we'll carry on.
				pass



	for freq_bin_number in range(len(freq_bin_dfs)):

		print("\n\nbin number: {} -- {}".format(freq_bin_number, datetime.datetime.now()))

		for pseudoword_number in range(max_n_pseudowords_per_freq_bin):

			print("\npseudoword number: {} -- {}".format(pseudoword_number,datetime.datetime.now()))

			#Schema C3:
			(new_df, sampled_context_words) = sample_context_word_group(freq_bin_dfs[freq_bin_number], 11)
			if sampled_context_words:
				print(sampled_context_words)
				
				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type7_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':sampled_context_words[1:], 'S2_words':[sampled_context_words[0]], 'type': 7, 'p1_array_series': np.random.dirichlet([0.1]*len(sampled_context_words), n_timesteps), 'p2_series': increasing_prob_arrays[pseudoword_number]}

				context_word_dict[sampled_context_words[0]] = {'pseudoword': pseudoword_name, 'pseudoword_type': 7, 'set_number':2}

				for (i, context_word) in enumerate(sampled_context_words[1:]):
					context_word_dict[context_word] = {'pseudoword': pseudoword_name, 'pseudoword_type': 7, 'set_number':1, 'dist_index': i}

			else:
				# not enough words in bin to take a sample
				break


	for freq_bin_number in range(len(freq_bin_dfs)):

		print("\n\nbin number: {} -- {}".format(freq_bin_number, datetime.datetime.now()))

		for pseudoword_number in range(max_n_pseudowords_per_freq_bin):

			print("\npseudoword number: {} -- {}".format(pseudoword_number,datetime.datetime.now()))

			# Schema D4:
			(new_df, sampled_context_words) = sample_context_word_group(freq_bin_dfs[freq_bin_number], 11)
			if sampled_context_words:
				print(sampled_context_words)
				
				freq_bin_dfs[freq_bin_number] = new_df

				pseudoword_name = "type6_bin{}_pseudoword{}".format(freq_bin_number,pseudoword_number)

				pseudoword_dict[pseudoword_name] = {'S1_words':sampled_context_words, 'type': 6, 'p1_array_series': np.random.dirichlet([0.1]*len(sampled_context_words), n_timesteps)}

				for (i, context_word) in enumerate(sampled_context_words):
					context_word_dict[context_word] = {'pseudoword': pseudoword_name, 'pseudoword_type': 6, 'set_number':1, 'dist_index':i}

			else:
				# not enough words in bin to take a sample
				break



	with open(outfiles_rootdir+'context_word_dict.json', 'w') as outfile:
		json.dump(context_word_dict, outfile, cls=NumpyEncoder)
	write_logfile(outfiles_rootdir+'context_word_dict.json', options, start_time)

	with open(outfiles_rootdir+'pseudoword_dict.json', 'w') as outfile:
		json.dump(pseudoword_dict, outfile, cls=NumpyEncoder)
	write_logfile(outfiles_rootdir+'pseudoword_dict.json', options, start_time)




