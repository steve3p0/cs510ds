import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import csv

#save file name to variable - make sure the file is in the path to be opened
file = 'SarcComments.csv'

#add column headers to each column in the file - makes it easier to manipulate columns
col_headers = ['post', 'label', 'author', 'subreddit', 'score', 'ups', 'downs',
               'date', 'Unix Time', 'Parent Comment', 'Link ID', 'Parent ID']

#read the file with the headers
comments_df = pd.read_csv(file, names = col_headers, sep = '\t', error_bad_lines = False)

for col in col_headers:
    comments_df = comments_df[pd.notnull(comments_df[col])]

#tokenize the posts and get rid of stopwords
comments = comments_df['post'].str.cat()

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

tokens = nltk.word_tokenize(comments)
tokens_no_punct = tokenizer.tokenize(comments)
text = nltk.Text(tokens_no_punct)

#filter out the stopwords
text_filtered = nltk.Text([w.lower() for w in text.tokens if w.lower() not in stop_words])

#grab the most common words in the dataset
freq_dist = nltk.FreqDist(text_filtered)
common_words = freq_dist.most_common(50)
print(common_words)

#save the common words to csv file to use later in tableau
with open('most_freq_words_sarc.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['word','count'])
    csv_out.writerows(common_words)