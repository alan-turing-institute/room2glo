#
# Requirements
# $ sudo apt-get install enchant
# $ sudo python3 -m pip install pyenchant
#

import enchant
import itertools

d_US = enchant.Dict('en_US')
d_GB = enchant.Dict('en_GB')
d_AU = enchant.Dict('en_AU')

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def remove_repetitions(tweet):
    """
    Adapted from the reduce_repetitions function in preprocess.py from https://github.com/Wronskia/Sentiment-Analysis-on-Twitter-data

    Wronskia's function simply reduced the max length of consecutive character repetitions to 2, and then checked against the dictionary.
    If not in the dictionary, all character repetitions were removed.

    Here, we also try shortening SOME but not ALL groups of repeated characters.

    We capitalize the last letter of normalized tokens to distinguish them from non-lengthened versions.

    """
    tweet=tweet.split()

    # for each token in the tweet:
    for i in range(len(tweet)):


        #iff the token is not in the dictionary:
        if not d_US.check(tweet[i]) and not d_GB.check(tweet[i]) and not d_AU.check(tweet[i]):

            # split the token into groups of identical consecutive characters, and reduce all groups to a maximum length of 2:
            # e.g. 'ssmiiilllleeeee' --> ['ss', 'm', 'ii', 'll', 'ee']
            #      'hellooooo' --> ['h', 'e', 'll', 'oo']
            chargroups = [''.join(s)[:2] for _, s in itertools.groupby(tweet[i])]
            charpair_indices = [j for j in range(len(chargroups)) if len(chargroups[j]) > 1 and not is_int(chargroups[j][0])]

            # iff there are some groups of repeated characters...
            if len(charpair_indices)>0 and len(charpair_indices)<=10:

                # check if our reduced token (i.e. with no character repetitions of length > 2) is in the dictionary.
                if d_US.check(''.join(chargroups)) or d_GB.check(''.join(chargroups)) or d_AU.check(''.join(chargroups)):
                    # (if it is in the dictionary, accept this as the canonical form, and stop here.)
                    tweet[i]=(''.join(chargroups))[::-1].capitalize()[::-1]
                
                else:

                    # we can begin by checking whether the word is in the dictionary when we remove ALL consecutive character repetitions.
                    # e.g. 'ssmiiilllleeeee' --> 'smile' . 'smile' is in the dictionary, yay!
               
                    # but: 'hellooooo' --> ['h', 'e', 'll', 'oo'] --> 'helo' . 'helo' is NOT in the dictionary!
                    # In cases like this, we next check if we can find a dictionary word by reducing just SOME of the repeated-character groups to single characters.
                    # e.g. we can check in the dictionary for 'hello' or 'helloo'.
                    # For 'sschooll',  we would consider 'sschol, 'school', 'scholl', 'sschool', 'sscholl', and 'schooll'.

                    # First, we create an iterator over all the possible subsets of groups of repeated characters
                    charpair_index_combos = []
                    for n in range(len(charpair_indices),0,-1):
                        charpair_index_combos.extend(itertools.combinations(charpair_indices,n))

                    # for each subset of the groups of repeated characters:
                    for index_combo in charpair_index_combos:

                        # copy the original list of character groups
                        chargroups2 = [c for c in chargroups]

                        # replace the groups in the subset with single characters
                        for j in index_combo:
                            chargroups2[j] = chargroups[j][:1]

                        # check if this version of the token is in the dictionary.
                        if d_US.check(''.join(chargroups2)) or d_GB.check(''.join(chargroups2)) or d_US.check(''.join(chargroups2)):
                            # (if it is in the dictionary, accept this as the canonical form, and stop here.)
                            tweet[i]=(''.join(chargroups2))[::-1].capitalize()[::-1]
                            break 

    tweet=' '.join(tweet)
    return tweet
