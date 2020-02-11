import nltk
import csv
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.corpus.reader.api import CorpusReader

# https://stackoverflow.com/questions/52686640/how-to-incorporate-metadata-into-nltk-corpus-for-efficient-processing
class MetadataCSVCorpusReader(CorpusReader):
    def __init__(self, root, fileids, encoding='utf8', tagset=None):
        super().__init__(root, fileids, encoding='utf8', tagset=None)
        self._parsed_metadata = {}
        metadata = self.open('metadata.csv')
        reader = csv.DictReader(metadata)
        for row in reader:
            self._parsed_metadata[row['fileid']] = row

    @property
    def metadata(self):
        """
        Return the contents of the corpus metadata.csv file, if it exists.
        """
        return self.open("metadata.csv").read()

    @property
    def parsed_metadata(self):
        """
        Return the contents of the metadata.csv file as a dict
        """
        return self._parsed_metadata


#meta_corpus = MetadataCSVCorpusReader()

# with open('sarc_s_meta_10.csv', 'r' ) as theFile:
#     reader = csv.DictReader(theFile)
#     for line in reader:
#         # line is { 'workers': 'w0', 'constant': 7.334, 'age': -1.406, ... }
#         # e.g. print( line[ 'workers' ] ) yields 'w0'
#         print(line)


#comments_df = pd.read_csv('sarc_s_meta_10.csv', error_bad_lines=False)
col_headers = ['label', 'comment', 'author', 'subreddit', 'up_down_votes', 'up_votes', 'down_votes', 'comment_date', 'unix_time', 'parent_comment', 'parent_comment_id', 'link_id']
comments_df = pd.read_csv('sarc_s_meta_10.csv', names=col_headers, sep='\t', encoding="utf-8")
#comments_df['comment_tok'] = comments_df['comment'].apply(nltk.word_tokenize)
#comments_df['comment_tok'] = comments_df['comment'].apply(nltk.tokenize.)

comments = comments_df['comment'].str.cat()
print(comments)
print(comments_df.shape)
print(comments_df['comment'])


tokens = nltk.word_tokenize(comments)
text = nltk.Text(tokens)
print(text)




#print(comments_df['comment_tok'])

