import csv
import argparse
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", "--input_file_location", type=str, default="/data2/distance_time_series/synthetic_data/skipgram/independent/neighborhood_last.json", help = "full path of the input file location")
    parser.add_argument("-ofile", "--output_file_location", type=str, default="./results_dec/significant_score_skipgram_independent_neighborhood_last.tsv", help = "location of the file where the results will be stored")
    options = parser.parse_args()

    with open(options.input_file_location) as f:
        data = json.load(f)

    for word, struct_time_cosine in data.items():
        cos_dist =list(struct_time_cosine.values())
        if len(cos_dist) > 0:
                range_cos_dist = max(cos_dist) - min(cos_dist)
                tau, p_value_k = sp.stats.kendalltau(cos_dist, range(len(cos_dist)))
                pcorr, p_value_p = sp.stats.pearsonr(cos_dist, range(len(cos_dist)))
                scorr, p_value_s = sp.stats.spearmanr(cos_dist, range(len(cos_dist)))
                regression_model.fit(np.asarray(list(range(len(cos_dist)))).reshape(-1,1), np.asarray(cos_dist).reshape(-1, 1))
                beta=regression_model.coef_[0][0]
                intercept = regression_model.intercept_[0]
                #stat = stats.ks_2samp(cos_dist, the_shuffle_c)[0]
                #p_value_ks = stats.ks_2samp(cos_dist, the_shuffle_c)[1]
                #print("Genuin Cos dist ", word," in sorted in time series: ", cos_dist, "tau: ", tau, "beta: ",beta)
                with open(options.output_file_location, "a") as output:
                        fieldnames = ['eval_word','kendall_tau','pearson_corr','spearcorr','beta','cosine_dist']
                        writer = csv.writer(output, delimiter='\t', lineterminator='\n')
                        writer.writerow([word, np.absolute(tau),np.absolute(pcorr),np.absolute(scorr), np.absolute(beta), cos_dist])
