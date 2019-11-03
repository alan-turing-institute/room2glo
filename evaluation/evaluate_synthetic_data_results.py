import json
from collections import defaultdict
import argparse



def precision_at_k(k_retrieved,n_correct):
	return n_correct / k_retrieved

def recall_at_k(total_n_correct, n_correct):
	return n_correct / total_n_correct

def average_precision_at_k(average_precision,k_retrieved,n_correct,total_n_correct,current_correct):
	if current_correct:
		change_in_recall = 1 / total_n_correct
		precision = precision_at_k(k_retrieved,n_correct)
		return average_precision + (precision * change_in_recall)
	else: 
		return average_precision




if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--results_dir", type=str, default="/results/change_point_candidates/independent/1/2012-01_2012-01_to_2017-06_2017-06/vec_200_w10_mc500_iter15_sg0/", help = "path to directory where results are stored")
	parser.add_argument("-f", "--results_fn", type=str, default="time_series_analysis_NOT_standardized_output_f2012_01_l2017_06_afirst_cfirst_mcosine_k25_s1000_p0.05_g0_v75.tsv", help = "name of file where results are stored")
	parser.add_argument("-c", "--word_column", type=int, default=1, help = "which column are words in? (for two-step candidates, it's 1; for time series candidates, it's 0)")
	parser.add_argument("-d", "--pseudoword_design_dict", type=str, default="/data/synthetic_evaluation_dataset/pseudoword_dict.json", help = "filepath for pseudoword design dict")
	options = parser.parse_args()


	infilepath = '/'.join([options.results_dir, options.results_fn])
	outfilepath = '/'.join([options.results_dir, options.results_fn[:-4] + '_metrics.tsv'])

	with open(options.pseudoword_design_dict) as infile:
		d = json.load(infile)

	total_n_correct = len([p for p in d if d[p]['type'] in (2,5,7)])
	print('total n correct: {}'.format(total_n_correct))
	n_correct = 0
	k_retrieved = 0
	n_correct_by_type_and_bin = defaultdict(lambda: defaultdict(int))
	n_incorrect_by_type_and_bin = defaultdict(lambda: defaultdict(int))
	n_non_pseudowords = 0

	average_precision = 0

	# open the results file
	with open(infilepath, 'r') as infile:
		with open(outfilepath, 'w') as outfile:
			for line in infile:
				k_retrieved += 1
				word=line.strip().split('\t')[options.word_column]
				if word in d:
					pseudoword_type = d[word]['type']
					freq_bin = word[9]

					if pseudoword_type in ('C1','C2','C3'):
						correct = True
						n_correct += 1
						n_correct_by_type_and_bin[pseudoword_type][freq_bin] += 1
					else:
						correct = False
						n_incorrect_by_type_and_bin[pseudoword_type][freq_bin] += 1
				else:
					correct = False
					n_non_pseudowords += 1


				average_precision = average_precision_at_k(average_precision,k_retrieved,n_correct,total_n_correct,correct)

				if k_retrieved in (10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,1000):

					precision = precision_at_k(k_retrieved,n_correct)
					recall = recall_at_k(total_n_correct, n_correct)

					
					outfile.write("{}\t{}\t{}\t{}\t{}\n".format(k_retrieved, precision, recall, average_precision, n_non_pseudowords))

					print("\n\n\nk = {}".format(k_retrieved))
					print("precision: {}".format(precision))
					print("recall: {}".format(recall))
					print("AP: {}".format(average_precision))
					print("n non-pseudowords: {}".format(n_non_pseudowords))
					print('\n')
					print(n_correct_by_type_and_bin)
					print('\n')
					print(n_incorrect_by_type_and_bin)






	# recall of different types.
	# recall of different freq bins.
	# recall of different pseudoword numbers.

	for type_number in range(1,8):
		print('\n\nTYPE NUMBER: {}\n'.format(type_number))
		outfilepath = '/'.join([options.results_dir, options.results_fn[:-4] + '_recall_type{}.tsv'.format(type_number)])

		with open(options.pseudoword_design_dict) as infile:
			d = json.load(infile)

		total_n_correct = len([p for p in d if d[p]['type'] == type_number])
		print('total n correct: {}'.format(total_n_correct))
		n_correct = 0
		k_retrieved = 0

		# open the results file
		with open(infilepath, 'r') as infile:
			with open(outfilepath, 'w') as outfile:
				for line in infile:
					k_retrieved += 1
					word=line.strip().split('\t')[options.word_column]
					if word in d:
						pseudoword_type = d[word]['type']
						freq_bin = word[9]

						if pseudoword_type == type_number:
							correct = True
							n_correct += 1


					if k_retrieved in (10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,1000):

						recall = recall_at_k(total_n_correct, n_correct)

						
						outfile.write("{}\t{}\n".format(k_retrieved, recall))

						print("\n\n\nk = {}".format(k_retrieved))
						print("recall: {}".format(recall))


	for freq_bin_number in range(5):
		outfilepath = '/'.join([options.results_dir, options.results_fn[:-4] + '_recall_freqbin{}.tsv'.format(freq_bin_number)])

		with open(options.pseudoword_design_dict) as infile:
			d = json.load(infile)

		total_n_correct = len([p for p in d if d[p]['type'] in ('C1','C2','C3') and int(p[9]) == freq_bin_number])
		print('total n correct: {}'.format(total_n_correct))
		n_correct = 0
		k_retrieved = 0
		
		# open the results file
		with open(infilepath, 'r') as infile:
			with open(outfilepath, 'w') as outfile:
				for line in infile:
					k_retrieved += 1
					word=line.strip().split('\t')[options.word_column]
					if word in d:
						pseudoword_type = d[word]['type']
						freq_bin = word[9]

						if pseudoword_type in ('C1','C2','C3') and int(freq_bin) == freq_bin_number:
							correct = True
							n_correct += 1


					if k_retrieved in (10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,1000):

						recall = recall_at_k(total_n_correct, n_correct)

						outfile.write("{}\t{}\n".format(k_retrieved, recall))

						print("\n\n\nk = {}".format(k_retrieved))
						print("recall: {}".format(recall))



	for pseudoword_number in range(6):
		outfilepath = '/'.join([options.results_dir, options.results_fn[:-4] + '_recall_pseudoword{}.tsv'.format(pseudoword_number)])

		with open(options.pseudoword_design_dict) as infile:
			d = json.load(infile)

		total_n_correct = len([p for p in d if d[p]['type'] in ('C1','C2','C3') and int(p[-1]) == pseudoword_number])
		print('total n correct: {}'.format(total_n_correct))
		n_correct = 0
		k_retrieved = 0
		
		# open the results file
		with open(infilepath, 'r') as infile:
			with open(outfilepath, 'w') as outfile:
				for line in infile:
					k_retrieved += 1
					word=line.strip().split('\t')[options.word_column]
					if word in d:
						pseudoword_type = d[word]['type']

						if pseudoword_type in ('C1','C2','C3') and int(word[-1]) == pseudoword_number:
							correct = True
							n_correct += 1


					if k_retrieved in (10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,1000):

						recall = recall_at_k(total_n_correct, n_correct)

						outfile.write("{}\t{}\n".format(k_retrieved, recall))

						print("\n\n\nk = {}".format(k_retrieved))
						print("recall: {}".format(recall))

