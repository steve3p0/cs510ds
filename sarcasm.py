import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.corpus.reader.api import CorpusReader
from nltk.tokenize import RegexpTokenizer

# tokenizer that removes punctuation
tokenizer = RegexpTokenizer(r'\w+')

#file = 'sarc_s_meta_10.csv'
#file = 'sarc_s_meta_100.csv'
#file = 'sarc_s_meta_1000.csv'
file = 'sarc_s_meta_10k.csv'
#file = 'sarc_s_meta_100k.csv'
col_headers = ['label', 'comment', 'author', 'subreddit', 'up_down_votes', 'up_votes', 'down_votes',
               'comment_date', 'unix_time', 'parent_comment', 'parent_comment_id', 'link_id']
comments_df = pd.read_csv(file, names=col_headers, sep='\t', encoding="utf-8", error_bad_lines=False, quoting=3) # quotechar=None)

# Remove Rows with Null Column Values
for col in col_headers:
    # if you want to allow null colunmn values for parent comments
    #if col != 'parent_comment' or col != 'parent_comment_id':
    comments_df = comments_df[pd.notnull(comments_df[col])]

comments = comments_df['comment'].str.cat()
tokens = nltk.word_tokenize(comments)
tokens_no_punct = tokenizer.tokenize(comments)
text = nltk.Text(tokens_no_punct)
stop_words = set(stopwords.words('english'))
text_filtered = nltk.Text([w.lower() for w in text.tokens if w.lower() not in stop_words])

print(len(text))                  # Total number of tokens
print(len(set(text)))             # number of unique tokens
print(len(set(text)) / len(text)) # lexical diversity
print(text.count("oh"))           # occurances of a word
print(100 * text.count("oh") / len(text)) # % of text


# Show 10 most common words
fdist = nltk.FreqDist(text_filtered)
print(fdist.most_common(20))

# Concordance Search of the word 'right'
print(text.concordance('right'))

# Plot a Fequency Distribution of the 10 most common words
fdist.plot(20, cumulative=False)

# Display a lexical Dispersion Plot
# Show words importance weighte by it's lexical disperson in a corpus
word_list = ['people', 'yeah', 'right', 'god', 'never', 'way', 'really']
text.dispersion_plot(word_list)

# Similar words
# Search of words that appear in a similar range of contexts as 'never'
print(text.similar('never'))

# Common Contexts
text.common_contexts(["never", "really"])

# Collocations
# print(text.collocation_list())  # <-- error ValueError: too many values to unpack (expected 2)
print('; '.join(text.collocation_list()))




# Other metrics
# Number of comments in each subreddit
# Comment Scores
# Comment Length: How many words
# https://andrewpwheeler.com/2016/06/08/sentence-length-in-academic-articles/




# Playing with the data
# print(comments)
# print(comments_df.shape)
# print(comments_df['comment'])
# print(text)
# text.concordance("OMG")


