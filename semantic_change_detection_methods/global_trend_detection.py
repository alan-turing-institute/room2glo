import csv
import argparse
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()

def convert_dist_dict(dist_dict):
    """
    Dictionaries of distances produced by change_point_detection.py are keyed first by time-slice, then by word.
    This function converts them such that are keyed first by word, then by time-slice.
    """
    dist_dict2 = {}
    for time_slice in dist_dict:
        for word in dist_dict[time_slice]:
            if word in dist_dict2:
                dist_dict2[word][time_slice] = dist_dict[time_slice][word]
            else:
                dist_dict2[word] = {}
                dist_dict2[word][time_slice] = dist_dict[time_slice][word]
    return dist_dict2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", "--input_file_location", type=str, default="/results/change_point_candidates/independent/1/2012-01_2012-01_to_2017-06_2017-06/vec_200_w10_mc500_iter15_sg0/time_series_analysis_distances_f2012_01_l2017_06_afirst_cfirst_mcosine_k25_s1000_p0.05_g0_v75.json", help = "full path of the input file location")
    parser.add_argument("-ofile", "--output_file_location", type=str, default="/results/global_trend_candidates/independent/1/2012-01_2012-01_to_2017-06_2017-06/vec_200_w10_mc500_iter15_sg0/f2012_01_l2017_06_afirst_cfirst_mcosine_k25_s1000_p0.05_g0_v75.tsv", help = "location of the file where the results will be stored")
    options = parser.parse_args()
    
    with open(options.input_file_location) as f:
        dist_dict = json.load(f)
        
    data = convert_dist_dict(dist_dict)

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
                with open(options.output_file_location, "a") as output:
                        fieldnames = ['eval_word','kendall_tau','pearson_corr','spearcorr','beta','cosine_dist']
                        writer = csv.writer(output, delimiter='\t', lineterminator='\n')
                        writer.writerow([word, np.absolute(tau),np.absolute(pcorr),np.absolute(scorr), np.absolute(beta), cos_dist])
