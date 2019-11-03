#!/usr/bin/env python3

import json
import re
from expressive_lengthening_reduction import remove_repetitions
from dictionary_of_contractions import *
import csv
import os
import html
import glob
import gzip
import multiprocessing
import sys
import datetime
import argparse


def tweet_clean(tweet):
    '''
    Basic cleaning of the raw tweets. Remove handles, non-Latin characters/numbers, and urls)
    '''
    tweet = html.unescape(tweet)

    # remove twitter handles:
    tweet = re.sub(r"@\w+", "", tweet)

    # remove anything that's not an alphanumeric character, an emoji, a whitespace character, or a hash symbol. Also match and remove urls.
    tweet = re.sub("([^\w\s#\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E6-\U0001F1FF])|(\w+:\/\/\S+)", "", tweet)

    # tokenize and re-join with single spaces (this step necesary in order to split apart clusters of emoji)
    # NB: this regex keeps flag emoji as single 'tokens' (i.e. does not split them into their two "regional indicator" letters), 
    # but it *does* split skintone-modified emoji into separate tokens for the base emoji and the skintone modifier.
    # (this is mainly because I couldn't figure out how to keep skintone-modifiable-emoji together with their modifiers but still split apart distinct emoji...
    # however I do think there is an argument for representing the semantics of the base emoji and the skintone-modifiers separately, anyway!)
    tweet = ' '.join(re.findall(u'[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF]|[\U0001F1E6-\U0001F1FF]{2}|\w+|#\w+', tweet))

    return tweet.strip().lower()
 

def create_clean_file(infilepath, filename, outfiles_rootdir):
    outfilepath = '/'.join([outfiles_rootdir, filename[:4], filename])
    with gzip.open(infilepath, 'rt') as infile:
        with gzip.open(outfilepath, 'wt') as outfile:
            tweetwriter = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            date=filename.replace(".csv.gz","")

            for line in infile:
                RT = -1
                d = json.loads(line)
                user_id =  d[u'user'][u'id_str']
                tweet_id =   d[u'id_str']
                tweet  = d[u'text']
                tweet = tweet_clean(tweet)

                if u'retweeted_status' not in d or d["retweeted_status"]==None:
                    if tweet.startswith('rt') is False:
                        RT = 0
                    else:
                        RT = 3
                else:
                    if tweet.startswith('rt') is False:
                        RT = 1
                    else:
                        RT = 2
                if d['cld']['cld_lang1']=="en":
                    cld = d[u'cld'][u'cld_lang1_percent']
                elif d['cld']['cld_lang2']=="en":
                    cld = d[u'cld'][u'cld_lang2_percent']
                else:
                    cld=-1 #Shouldn't get here.
                tweet = remove_repetitions(tweet)
                tweet = contractions(tweet)
                
                tweetwriter.writerow([date, user_id, tweet_id, RT,cld, tweet])
    return infilepath


def success(infilepath):
    sys.stdout.write("{} completed at {}\n".format(infilepath,datetime.datetime.now()))
    sys.stdout.flush()


def error(a):
    sys.stderr.write("\nError: {}\n\n".format(a))
    sys.stderr.flush()
    

def write_logfile(infiles_rootdir, outfiles_rootdir, start_time):
    logfile_path = '/'.join([outfiles_rootdir, 'corpus_provenance.log'])
    with open(logfile_path, 'w') as logfile:
        logfile.write('Script started at: {}\n\n'.format(start_time))
        logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
        logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
        logfile.write('Original corpus located at: {}\n\n'.format(infiles_rootdir))
        logfile.write('Cleaned corpus written to:{}\n\n').format(outfiles_rootdir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str, default="/data/twitter_spritzer/001p/", help = "path to directory where original corpus is stored")
    parser.add_argument("-o", "--outfiles_rootdir", type=str, default='/data/twitter_spritzer/clean_001p/', help = "path to directory where where cleaned corpus should be written")
    options = parser.parse_args()

    start_time = datetime.datetime.now()

    pool = multiprocessing.Pool()  #defaults to the number of Cores on the machine


    print("Started at {}\n".format(datetime.datetime.now()))
    for year in range(2011,2019):
        os.makedirs('/'.join([options.outfiles_rootdir, str(year)]),exist_ok=True) #Create all needed dirs, no error if they already exist
        for month in range(1,13):
            for infilepath in glob.glob("{}/{}/{}-{:02}-*".format(options.infiles_rootdir,year,year,month)):
                filename = infilepath.split('/')[-1]
                pool.apply_async(create_clean_file, (infilepath, filename, options.outfiles_rootdir), callback=success, error_callback=error)

    try:
        pool.close() 
        pool.join() 
    except:
        pass    

    print("All threads finished. Writing log file...")
    write_logfile(options.infiles_rootdir, options.outfiles_rootdir, start_time)
    print("Written log. All done!")
