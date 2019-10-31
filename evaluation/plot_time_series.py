#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from time_series_analysis import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")


def latexify(fig_width=None, fig_height=None, columns=1):
        """Set up matplotlib's RC params for LaTeX plotting.
        Call this before plotting a figure.

        Parameters
        ----------
        fig_width : float, optional, inches
        fig_height : float,  optional, inches
        columns : {1, 2}
        """

        # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.png

        assert(columns in [1,2])

        if fig_width is None:
                fig_width = 3.39 if columns==1 else 6.9 # width in inches

        if fig_height is None:
                golden_mean = (np.sqrt(5)-1.0)/2.0 # Aesthetic ratio
                fig_height = fig_width*golden_mean # height in inches

        MAX_HEIGHT_INCHES = 8.0
        if fig_height > MAX_HEIGHT_INCHES:
                print("WARNING: fig_height too large:" + fig_height + 
                  "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

        params = {'backend': 'ps',
                  #'text.latex.preamble': ['\\usepackage{gensymb}'],
                  'axes.labelsize': 8, # fontsize for x and y labels (was 10)
                  'axes.titlesize': 8,
                  'font.size': 10, # was 10
                  'legend.fontsize': 8, # was 10
                  'xtick.labelsize': 5,
                  'ytick.labelsize': 8,
                  #'text.usetex': True,
                  'figure.figsize': [fig_width,fig_height],
                  'font.family': 'serif'
        }

        matplotlib.rcParams.update(params)

def format_axes(ax):

        for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
                ax.spines[spine].set_color(SPINE_COLOR)


        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.grid(which="major", linestyle=':')
        ax.xaxis.grid(which="major", linestyle=':')

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(direction='out', color=SPINE_COLOR)

        return ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--words", type=str, default="snowflake glo stan 💀 lm the", help = "list of  words (separated by spaces) to examine")
    parser.add_argument("-d", "--models_rootdir", type=str, default="/data/twitter_spritzer/clean_001p_nodups_models/monthly/", help = "path to directory where models are stored")
    parser.add_argument("-f", "--first_timeslice", type=str, default='2012_01', help = "path to file where output should be written")
    parser.add_argument("-l", "--last_timeslice", type=str, default='2017_06', help = "path to file where output should be written")
    parser.add_argument("-a", "--align_to", type=str, default="last", help = "which model to align every other model to: 'first' or 'last'")
    parser.add_argument("-c", "--compare_to", type=str, default="first", help = "which model's vector to compare every other model's vector to: 'first', 'last', or 'previous'")
    parser.add_argument("-m", "--distance_measure", type=str, default="cosine", help = "which distance measure to use 'cosine', or 'neighborhood'")
    parser.add_argument("-k", "--k_neighbors", type=int, default=25, help = "Number of neighbors to use for neighborhood shift distance measure")
    parser.add_argument("-s", "--n_samples", type=int, default=1, help = "Number of samples to draw for permutation test")
    parser.add_argument("-v", "--vocab_threshold", type=int, default=75, help = "percent of models which must contain word in order for it to be included")
    options = parser.parse_args()

    words = options.words.split()

    SPINE_COLOR = 'gray'
    dpi = 300
    latexify(fig_width =20, fig_height =0.5)

    print("Starting at {}".format(datetime.datetime.now()))

    
    model_paths = []
    time_slice_labels = []
    vocab_filepath = "{}/time_series_vocab_{}pc_{}_to_{}.txt".format(options.models_rootdir, options.vocab_threshold, options.first_timeslice, options.last_timeslice)

    (first_year, first_month) = (int(i) for i in options.first_timeslice.split('_'))
    (last_year, last_month) = (int(i) for i in options.last_timeslice.split('_'))

    # if we've already stored the common vocab, can just read it, don't have to load all the models and check their vocab
    if os.path.isfile(vocab_filepath):

        vocab = set()
        with open(vocab_filepath, 'r') as infile:
            for line in infile:
                vocab.add(line.strip())

        for year in range(2011,2019):

            if year < first_year:
                continue
            elif year > last_year:
                break

            for month in range(1,13):
                    
                if year == first_year and month < first_month:
                    continue
                elif year == last_year and month > last_month:
                    break

                time_slice = "{}_{:02}".format(year, month)
                model_path = "{}/{}/{:02}/200/saved_model.gensim".format(options.models_rootdir, year, month)
                if os.path.isfile(model_path):
                    model_paths.append(model_path)
                    time_slice_labels.append(time_slice)

    else:
        # if we HAVEN'T already stored the common vocab, we DO need to load all the models and check their vocab

        vocab_counter = Counter()
        for year in range(2011,2019):

            if year < first_year:
                continue
            elif year > last_year:
                break


            for month in range(1,13):

                if year == first_year and month < first_month:
                    continue
                elif year == last_year and month > last_month:
                    break

                time_slice = "{}_{:02}".format(year, month)
                model_path = "{}/{}/{:02}/200/saved_model.gensim".format(options.models_rootdir, year, month)
                try:
                    model = load_model(model_path)
                except FileNotFoundError:
                    pass
                else:
                    print("loaded {} at {}".format(time_slice, datetime.datetime.now()))
                    model_paths.append(model_path)
                    time_slice_labels.append(time_slice)
                    vocab_counter.update(model.vocab.keys())
        
        n_models = len(model_paths)
        print(n_models)
        print(vocab_counter.most_common(10))
        vocab = set([w for w in vocab_counter if vocab_counter[w] >= options.vocab_threshold * 0.01 * n_models])
        del vocab_counter

        with open(vocab_filepath, 'w') as outfile:
            for word in vocab:
                outfile.write(word+'\n')

        
    # vocab = ['glo']

    print("\nGot vocab at {}".format(datetime.datetime.now()))
    print("size of vocab: {}\n".format(len(vocab)))



    dict_of_dist_dicts = {}
    dict_of_z_score_dicts = {}
    time_slice_labels_used = []


    for (i, model_path) in enumerate(model_paths):

        if i == 0 and (options.compare_to == 'previous' or options.align_to =='previous' or options.compare_to == 'first'):
            continue

        elif i == len(model_paths) - 1 and options.compare_to == 'last': 
            continue

        else:

            if options.align_to == 'first':
                alignment_reference_model_path = model_paths[0]
            elif options.align_to == 'last':
                alignment_reference_model_path = model_paths[-1]
            else:
                alignment_reference_model_path = model_paths[i-1]


            if options.compare_to == 'first':
                comparison_reference_model_path = model_paths[0]
            elif options.compare_to =='last':
                comparison_reference_model_path = model_paths[-1]
            else:
                comparison_reference_model_path = model_paths[i-1]


            dist_dict = get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, vocab, options.distance_measure, options.k_neighbors)

            dict_of_dist_dicts[time_slice_labels[i]] = dist_dict


            z_score_dict = get_z_score_dict(dist_dict)

            dict_of_z_score_dicts[time_slice_labels[i]] = z_score_dict

            time_slice_labels_used.append(time_slice_labels[i])



    print("GOT DICT OF DIST DICTS AND DICT OF Z-SCORE DICTS at {}\n".format(datetime.datetime.now()))


    for word in words:

        if word not in vocab:
            print("\n\n{} not in vocab".format(word))
            continue

        print('\n\n'+word)

        dist_series = [dict_of_dist_dicts[time_slice][word] for time_slice in time_slice_labels_used]

        z_score_series = [dict_of_z_score_dicts[time_slice][word] for time_slice in time_slice_labels_used]

        notNone_time_slice_labels = [time_slice_labels_used[i] for i in range(len(time_slice_labels_used)) if z_score_series[i]]
        
        z_score_series = [i for i in z_score_series if i]

        mean_shift_series = get_mean_shift_series(z_score_series, options.compare_to)

        p_value_series = get_p_value_series(word, mean_shift_series, options.n_samples, z_score_series, options.compare_to)


        print(dist_series)
        print(z_score_series)
        print(mean_shift_series)
        print(p_value_series)


        plt.clf()

        x_vals = np.arange(len(notNone_time_slice_labels))

        fig, axes = plt.subplots(1,4)
        axes[0].plot(x_vals, [d for d in dist_series if d])
        axes[0].set_title('Distance')
        axes[1].plot(x_vals, z_score_series)
        axes[1].set_title('Z-score')
        axes[2].plot(x_vals[:-1], mean_shift_series)
        axes[2].set_title('Mean-shift')
        axes[3].plot(x_vals[:-1], p_value_series)
        axes[3].set_title('P-value')
        fig.suptitle("Time series for {}".format(word.encode('utf-8')))

        for ax in axes:
            ax.set_xticks(x_vals)
            ax.set_xticklabels(time_slice_labels)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                format_axes(ax)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        plt.savefig('{}_timeseries_plot_f{}_l{}_a{}_c{}_m{}_k{}_s{}.png'.format(word, options.first_timeslice, options.last_timeslice, options.align_to, options.compare_to, options.distance_measure, options.k_neighbors, options.n_samples), dpi=dpi)
