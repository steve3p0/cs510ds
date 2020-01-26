# Use this in the future for reading corpora from a text file
# from nltk.corpus import PlaintextCorpusReader
# corpus_root = 'corpora/'  # Mac users should leave out C:
# corpus = PlaintextCorpusReader(corpus_root, '.*txt')  # all files ending in 'txt'

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict

nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('tagsets')
# tagset_upenn = nltk.help.upenn_tagset()

# Function Words: We explored using NLTK stop words, but ultimately we did not use it
# We combined ADP, PRON, DET, CONJ into Inserts
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stops = set(stopwords.words("english"))

insert_words = ( 'yeah', 'Ok', 'ahh')

messages = ['Gym?',
            'yeah be there in about a half',
            'Ok see you when you get here!',
            'Seconds away',
            'Meet me between smith and cramer asap',
            'I got you and Taylor tix in pit section.',
            'Get some milk please',
            'Chk email',
            'Made it',
            'Do u know where u saved that movie on my compute',
            'Im meeting some dude from the internet for happy hour ahh!',
            'Wed is dinner for renetta call us soon',
            'where r u???',
            'pinball']

# Iterate thru each message in our 2007 Text Message Corpus
# NOTE: We could have written this code to simply get the counts on the whole corpus,
# but for this assignment, message level analysis made it easier to confirm with manual conounts
# This is how it looks if we use the upenn tagset:
# Counter({'NOUN': 29, 'VERB': 12, 'ADP': 9, 'ADV': 8, 'PRON': 8, '.': 7, 'DET': 4, 'ADJ': 3, 'CONJ': 2})
# But we are going to modify the tagging to:
#   1. Inserts: Check if a word is in our inserts set
#   2. Function Words: Combine ADP, PRON, DET, CONJ into FUNCTOR
#   3. Remove Punctuation

counter_list = []
words = defaultdict(list)

for msg in messages:
    tokens = nltk.word_tokenize(msg)
    word_tag_pairs = nltk.pos_tag(tokens, tagset='universal')
    print(f"\nRaw Message: {msg}")
    print(f"Words with POS Tags: {word_tag_pairs}")

    # Build a dictionary of words, grouped by POS
    for w, tag in word_tag_pairs:
        if w in insert_words:
            tag = "Inserts"
            words[tag].append(w)
        elif tag in ('PRON', 'DET', 'ADP', 'CONJ'):
            tag = "FUNCTOR"
            words[tag].append(w)
        elif tag != '.':
            words[tag].append(w)

counter_pos = {k: len(v) for k,v in words.items()}
total_words = sum(counter_pos.values())

# List of words by POS
print(f"\nSummary - POS Tagging")
print(f"counter_pos: {counter_pos}")
print(f"Nouns: {words['NOUN']}")
print(f"Verbs: {words['VERB']}")
print(f"Adjectives: {words['ADJ']}")
print(f"Adverbs: {words['ADV']}")
print(f"Function Words: {words['FUNCTOR']}")
print(f"Inserts: {words['Inserts']}")

# Gather Counts
raw_counts_nouns = counter_pos['NOUN']
raw_counts_verbs = counter_pos['VERB']
raw_counts_adverbs = counter_pos['ADV']
raw_counts_adjectives = counter_pos['ADJ']
raw_counts_functors = counter_pos['FUNCTOR']
raw_counts_inserts = counter_pos['Inserts']

percent_nouns = raw_counts_nouns / total_words
percent_verbs = raw_counts_verbs / total_words
percent_adjectives = raw_counts_adjectives / total_words
percent_adverbs = raw_counts_adverbs / total_words
percent_functors = raw_counts_functors / total_words
percent_inserts = raw_counts_inserts / total_words

norm_counts_nouns = percent_nouns * 1000
norm_counts_verbs = percent_verbs * 1000
norm_counts_adjectives = percent_adjectives * 1000
norm_counts_adverbs = percent_adverbs * 1000
norm_counts_functors = percent_functors * 1000
norm_counts_inserts = percent_inserts * 1000

# Print Counts
print(f"\nRaw Counts:")
print(f"Nouns: {raw_counts_nouns}")
print(f"Verbs: {raw_counts_verbs}")
print(f"Adjectives: {raw_counts_adjectives}")
print(f"Adverbs: {raw_counts_adverbs}")
print(f"Function Words: {raw_counts_functors}")
print(f"Inserts: {raw_counts_inserts}")
print(f"Total Words: {total_words}")

print(f"\nPercentages:")
print(f"Nouns: {percent_nouns:.1%}")
print(f"Verbs: {percent_verbs:.1%}")
print(f"Adjectives: {percent_adjectives:.1%}")
print(f"Adverbs: {percent_adverbs:.1%}")
print(f"Function Words: {percent_functors:.1%}")
print(f"Inserts: {percent_inserts:.1%}")
total_percentages = sum([percent_nouns, percent_verbs, percent_adjectives, percent_adverbs,
                        percent_functors, percent_inserts])
print(f"Total Percentages: {total_percentages:.1%}")

print(f"\nNormed Counts Per 1000:")
print(f"Nouns: {norm_counts_nouns:0.1f}")
print(f"Verbs: {norm_counts_verbs:0.1f}")
print(f"Adjectives: {norm_counts_adjectives:0.1f}")
print(f"Adverbs: {norm_counts_adverbs:0.1f}")
print(f"Function Words: {norm_counts_functors:0.1f}")
print(f"Inserts: {norm_counts_inserts:0.1f}")
total_norm_counts = sum([norm_counts_nouns, norm_counts_verbs, norm_counts_adjectives, norm_counts_adverbs,
                        norm_counts_functors, norm_counts_inserts])
print(f"Total Norm Counts: {total_norm_counts:0.1f}")

# Graph it!
width = 0.7
p1 = plt.bar(width=width, x=1, height=norm_counts_nouns)
p2 = plt.bar(width=width, x=1, height=norm_counts_verbs, bottom=norm_counts_nouns)
p3 = plt.bar(width=width, x=1, height=norm_counts_adverbs, bottom=norm_counts_nouns + norm_counts_verbs)
p4 = plt.bar(width=width, x=1, height=norm_counts_adjectives, bottom=norm_counts_nouns + norm_counts_verbs + norm_counts_adverbs)

plt.ylabel('Normed Counts Per 1000 Words')
plt.title('2007 Text Messages: Frequency of Lexical Word Classes')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

actual_last_value = norm_counts_nouns + norm_counts_verbs + norm_counts_adverbs
max_y_value = total_words * 1000
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=(p1[0], p2[0], p3[0], p4[0]), labels=('Nouns', 'Verbs', 'Adverbs', 'Adjectives'))
plt.autoscale(False)
plt.show()
