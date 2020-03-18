import nltk
from nltk.corpus import stopwords
import string
import csv
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

# Category short hands and mappings for easier coding.
# Category and subset or set are used interchangeably.
REF = "Reference"
NON = "Nonsarcastic"
SAR = "Sarcastic"

main_cates = [REF, NON, SAR]

# Source files for the most frequent words in each category of data.
key_word_src_files = {
    REF: 'most_freq_words.csv',
    NON: 'most_freq_words_non_sarc.csv',
    SAR: 'most_freq_words_sarc.csv',
}


# Source files for labeled sarcastic and nonsarcastic data and metadata.
sarc_file_name = "SarcComments.csv"
nonsarc_file_name = "NonSarcComments.csv"


# From file pull the corpus. Expects a CSV file with tokenized comments in first column followed by metadata.
# [str] where each string is a post made of only lower case letters.
def get_corpus(file_name):
    with open(file_name, "r") as file_handle:
        #  For each line in the file remove ending whitespace, convert it to a list, get only the post and no metadata,
        #  convert all letters in the post to lowercase, and add it to "the_text".
        #  The csv header file will be in the 0th position, remove it.
        the_text = [line.strip().split("\t")[0].lower() for line in file_handle.readlines()][1:]

    return the_text


# Get the corpus for later processing. Important that these data go unchanged and are copied rather than altered.
sarc_corpus = get_corpus(sarc_file_name)  # [str]
nonsarc_corpus = get_corpus(nonsarc_file_name)
reference_corpus = sarc_corpus + nonsarc_corpus

# {str: [str]} mapping short hand to corpus.
cate_to_corpus = {
    SAR: sarc_corpus,
    NON: nonsarc_corpus,
    REF: reference_corpus,
}


# Pulls data from files that contain the most frequent keywords for each set.
# Returns list of keywords without counts.
def most_freq_words_from_file(the_file):
    with open(the_file, newline='') as the_csv_file:
        the_csv_file = list(csv.reader(the_csv_file, delimiter=','))[1:]  # Remove header.
    return [word[0] for word in the_csv_file]  # Loose keyword count in file ('the_file'); if need, need to recalculate.


# Creates a Venn diagram that shows the overlap of keywords between the Reference, Sarcastic, and Nonsarcastic sets.
def graph_venn(setoverlaps):
    goodanswers = ['y', 'n']
    userchoice = input("Do you want to make a Venn diagram of how the sets over lap?\n" +
                   "good answers are " + " or ".join(goodanswers) + ".\n" +
                   "Your selection: ")
    a, b, c = main_cates

    if userchoice.strip() not in goodanswers:
        print("Invalid selection. Try again.")
        graph_venn(setoverlaps)

    elif userchoice.strip() == 'y':
        userchoice = input("Enter file name (the code will add the .png ending for you): ")
        print('\n')
        venn3(subsets=(len(setoverlaps[a]), len(setoverlaps[b]), len(setoverlaps[a+b]), len(setoverlaps[c]),
                       len(setoverlaps[a+c]), len(setoverlaps[b+c]), len(setoverlaps[a+b+c])),
              set_labels=(main_cates[0], main_cates[1], main_cates[2]))
        plt.title(userchoice.strip())
        plt.savefig(userchoice + ".png")
        plt.close()


# Finds how keywords appear in multiple subsets or none at all.
def set_overlaps(all_sets):
    # ABC
    to_ret = {}
    a, b, c = main_cates
    to_ret[a] = [x for x in all_sets[a] if x not in all_sets[b] + all_sets[c]]  # Abc
    to_ret[b] = [x for x in all_sets[b] if x not in all_sets[a] + all_sets[c]]  # aBc
    to_ret[c] = [x for x in all_sets[c] if x not in all_sets[a] + all_sets[b]]  # abC

    to_ret[a+b] = [x for x in all_sets[a] if x in all_sets[b] and x not in all_sets[c]]  # ABc
    to_ret[a+c] = [x for x in all_sets[a] if x in all_sets[c] and x not in all_sets[b]]  # AbC
    to_ret[b+c] = [x for x in all_sets[b] if x in all_sets[c] and x not in all_sets[a]]  # aBC

    to_ret[a+b+c] = [x for x in all_sets[a] if x in all_sets[b] and x in all_sets[c]]  # ABC

    graph_venn(to_ret)

    return to_ret


# {str: [str]}
# Maps the categories to lists of their most frequent words.
most_freq_words = {
    cate: most_freq_words_from_file(freq_words) for cate, freq_words in key_word_src_files.items()
}


# Map the various union subsets to their categorical names.
key_words = set_overlaps(most_freq_words)  # {str: [str]}


# Orientation is your right or left, not the posts's.
# Returns a list of strings, [str].
# Biased in that if choosing center orientation with an even window size, the keyword appears in the window's left side.
def get_window(key_word, the_post, orientation, window_size):
    to_return = []
    locus = the_post.index(key_word)
    right = lambda arr, loc, win: arr[loc: loc+win]
    left = lambda arr, loc, window: arr[0 if locus+1 - window < 0 else locus+1 - window: loc+1]

    if orientation == 'right':
        to_return = right(the_post, locus, window_size)

    if orientation == 'left':
        to_return = left(the_post, locus, window_size)

    if orientation == 'center':
        if window_size % 2 == 0:
            amt_left = window_size // 2
            amt_right = window_size - amt_left + 1
        else:
            amt_left = (window_size // 2) + 1
            amt_right = amt_left

        to_return = left(the_post, locus, amt_left) + right(the_post, locus, amt_right)[1:]

    return to_return


# Allows user to choose which sets of keywords are to undergo tagged context search.
def user_get_subsets():
    tar_subsets = input("Specify which union subsets to analyze or 'exit' to leave; "
                        "space delineated. the options are: \n" + ", ".join(key_words.keys()) + "\nselect: ")
    tar_subsets = list(set(tar_subsets.strip().split(" ")))

    if tar_subsets and tar_subsets[0] == 'exit':
        exit()

    # Ensure all user inputs are valid cate.
    invalid_input = not bool(tar_subsets) or bool(sum(map(lambda x: 1 if x not in key_words.keys() else 0, tar_subsets)))
    if invalid_input:
        print("Incorrect entry. Try again: \n")
        tar_subsets = user_get_subsets()

    return tar_subsets


# Allows the user to dictate the hyper-parameters of the keyword in tagged context search.
def user_get_window_params():
    user_response = input("\nType window size and if 'center' on keyword, to 'left', or to the 'right' of keyword.\n"
                          "Size is specified first than window centeredness; space delineated.\n"
                          "Window size is between 1 and 100, inclusive.\n"
                          "Select: ")
    user_response = user_response.strip().split(" ")

    if user_response and user_response[0] == 'exit':
        exit()

    bad_input_flags = True if len(user_response) != 2 else False or \
                      True if not user_response[0].isdigit() else False or \
                      True if int(user_response[0]) < 1 or int(user_response[0]) > 100 else False or \
                      True if not user_response[1] else False or \
                      True if user_response[1] not in ['right', 'left', 'center'] else False

    if bad_input_flags:
        print("Incorrect entry. Try again: \n")
        user_response = user_get_window_params()

    return int(user_response[0]), user_response[1]


# For each category of keyword, get the tagged context of each keyword in the category.
def get_tagged_keyword_context(subsets, window_size, window_center, stop_words):
    context_by_subset = dict()

    for some_subsets in subsets:
        working_keywords = list()
        working_corporas = list()
        working_keywords += key_words[some_subsets]

        for cate in main_cates:
            if cate in some_subsets:
                working_corporas += cate_to_corpus[cate]

        contexts = list()
        for a_post in working_corporas:
            for a_keyword in working_keywords:
                if a_keyword in [word for word in a_post.split(" ") if word not in stop_words]:
                    tagged_post = [a_word for a_word in nltk.pos_tag(a_post.split(" ")) if a_word[0] not in stop_words]
                    tagged_keyword = [a_word for a_word in tagged_post if a_word[0] == a_keyword][0]
                    contexts += get_window(tagged_keyword, tagged_post, window_center, window_size)

        context_by_subset[some_subsets] = contexts

    return context_by_subset


# Gets a count for each distinct (keyword, POS tag) tuple by category.
# Gets total number of (keyword, POS tag) for each category.
# Gets counts of how often each POS appears inn each category.
def process_tagged_contexts(contexts_by_subset):
    counts = {k: Counter(v) for k, v in contexts_by_subset.items()}
    tots = {k: sum([value for key, value in v.items()]) for k, v in counts.items()}
    tots_by_POS = dict()

    for key, values in counts.items():
        accum = dict()
        for k, v in values.items():
            if k[1] in accum:
                accum[k[1]] += v
            else:
                accum[k[1]] = v
        tots_by_POS[key] = accum

    return counts, tots, tots_by_POS


# Write tagged keyword context to file.
def keyword_in_tagged_context_tofile(data, file):
    with open(file, "w") as file_handle:
        file_handle.write("Subset,Word,POS,CountInSubset\n")
        for category, taggedkeyword_counts in data.items():
            for taggedkeyword, count in taggedkeyword_counts.items():
                file_handle.write(category+','+taggedkeyword[0]+','+taggedkeyword[1]+','+str(count)+'\n')


# determines if user wants to save tagged context data to file.
def send_tagged_context_to_file(data):
    goodanswers = ['y', 'n']
    userchoice = input("Do you want to ship the tagged contextual words and counts by category to file?.\n" +
                       "Good answers are " + " or ".join(goodanswers) + ".\n" +
                       "Your selection: ")
    if userchoice.strip() not in goodanswers:
        print("Invalid selection. Try again.")
        send_tagged_context_to_file(data)
    elif userchoice.strip() == 'y':
        userchoice = input("Enter file name (the code will add the .csv tag for you): ")
        keyword_in_tagged_context_tofile(data, userchoice + ".csv")


# Get hyper-parameters from user.
subsets = user_get_subsets()
window_size, window_center = user_get_window_params()

# Perform keyword in tagged context search.
stop_words = set([word.lower().translate(str.maketrans('', '', string.punctuation)) for word in stopwords.words('english')])
context_by_subset = get_tagged_keyword_context(subsets, window_size, window_center, stop_words)
taggedwords_by_cate, totalltaggedcount_by_cate, POScounts_by_cate = process_tagged_contexts(context_by_subset)

# Save keyword in tagged context search to file.
send_tagged_context_to_file(taggedwords_by_cate)
