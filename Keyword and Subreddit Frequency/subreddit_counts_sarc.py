import nltk
import pandas as pd
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

#grab the subreddit information
subreddits = comments_df['subreddit']

#grab the most common subreddits in the dataset
freq_dist = nltk.FreqDist(subreddits)
common_topics = freq_dist.most_common(50)
print(common_topics)

#save the common subreddits to csv file to use later in tableau
with open('subreddits_category_count_sarc.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['category','count'])
    csv_out.writerows(common_topics)
