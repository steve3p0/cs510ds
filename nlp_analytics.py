import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#import sys
#sys.modules.pop('matplotlib')
import matplotlib
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
# from collections import defaultdict
from IPython.display import display, HTML
display(HTML("<style>pre { white-space: pre !important; }</style>"))
plt.rcdefaults()
#%matplotlib inline
#%matplotlib notebook

print(f"NLTK: {nltk.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"matplotlib: {matplotlib.__version__}")


# Load Sarcastic Content and Meta Data from CSV file

#file = 'sarc_s_meta_100k.csv'
file = 'sarc_s_meta_10k.csv'
#file = 'sarc_s_meta_10.csv'
col_headers = ['label', 'comment', 'author', 'subreddit', 'rank', 'up', 'down',
               'date', 'Unix Time', 'Parent Comment', 'Parent ID', 'Link ID']
comments_df = pd.read_csv(file, names=col_headers, sep='\t', error_bad_lines=False, quoting=3) # quotechar=None)

# Remove Rows with Null Column Values
for col in col_headers:
    # if you want to allow null colunmn values for parent comments
    #if col != 'parent_comment' or col != 'parent_comment_id':
    comments_df = comments_df[pd.notnull(comments_df[col])]

pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 12)

df = pd.DataFrame(comments_df[0:9])

# Set table styles
styles = [ dict(selector="th", props=[('text-align', 'center')]),
           dict(selector="th", props=[('white-space', 'nowrap')]),
           dict(selector="td", props=[('text-align', 'left')]) ]
styled_df = (df.style
             .set_properties(subset=df.columns[0],  **{'white-space':'nowrap'})
             .set_properties(subset=df.columns[1],  **{'width': '300px'})
             .set_table_styles(styles))

html = styled_df.hide_index().render()
display(HTML(html))

# comments = comments_df['comment'].str.cat()

# Tokenization
# Create a custom tokenizer to remove punctuation and stopwords.
# Stopwords are common words like 'the', 'and', 'an', etc.


comments = comments_df['comment'].str.cat()

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

tokens = nltk.word_tokenize(comments)
tokens_no_punct = tokenizer.tokenize(comments)
text = nltk.Text(tokens_no_punct)

text_filtered = nltk.Text([w.lower() for w in text.tokens if w.lower() not in stop_words])


### Counting Words, Frequency Distributions, and Collocations
# We can simply count the number of tokens in a corpus, the number of unique tokens (word types), the
# lexical diversity of a corpus, the occurrences of a specific word, and the percentage that a word takes up
# in a corpus, among many other calculations we can make. This will give us some sense of how large the
# corpus is, how lexically rich it is and allows us to start poking around the corpus.

comments = comments_df['comment'].str.cat()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
tokens = nltk.word_tokenize(comments)
tokens_no_punct = tokenizer.tokenize(comments)
text = nltk.Text(tokens_no_punct)
text_filtered = nltk.Text([w.lower() for w in text.tokens if w.lower() not in stop_words])

print(len(text))                  # Total number of tokens
print(len(set(text)))             # number of unique tokens
print(len(set(text)) / len(text)) # lexical diversity
print(text.count("oh"))           # occurances of a word
print(100 * text.count("oh") / len(text)) # % of text


### Common Words, Concordance Searching and Frequency Distributions
# We can display the most common words, perform condordance searches and display frequency distributions.

fdist = nltk.FreqDist(text_filtered)
print(fdist.most_common(20))

# Concordance Search of the word 'right'
print(text.concordance('right'))

# Plot a Fequency Distribution of the 10 most common words
#fdist.plot(20, cumulative=False)
fdist.plot(20)

### Display a lexical Dispersion Plot
# Show a word's importance weighted by it's lexical disperson in a corpus

word_list = ['people', 'yeah', 'right', 'god', 'never', 'way', 'really']
text.dispersion_plot(word_list)

### Similar words, Common Contexts, and Collocations
# Show a word's importance weighted by it's lexical disperson in a corpus

# Search of words that appear in a similar range of contexts as 'never'
print(text.similar('never'))
# Common Contexts
text.common_contexts(["never", "really"])
# Collocations
# print(text.collocation_list())  # <-- error ValueError: too many values to unpack (expected 2)
print('; '.join(text.collocation_list()))

### Other metrics
# * Number of comments in each subreddit
# * Comment Scores
# * Comment Length: How many words
# * Sentence Length
#https://andrewpwheeler.com/2016/06/08/sentence-length-in-academic-articles/
