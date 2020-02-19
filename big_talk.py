import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib
from IPython.display import display, HTML

class BigTalk:
    def __init__(self, file):
        # Load Sarcastic Content and Meta Data from CSV file
        self.filename = file
        self.col_headers = ['label', 'comment', 'author', 'subreddit', 'rank', 'up', 'down',
                            'date', 'Unix Time', 'Parent Comment', 'Parent ID', 'Link ID']
        self.comments_df = pd.read_csv(file, names=self.col_headers, sep='\t', error_bad_lines=False,  quoting=3)

        # Remove Rows with Null Column Values
        for col in self.col_headers:
            # if you want to allow null colunmn values for parent comments
            # if col != 'parent_comment' or col != 'parent_comment_id':
            self.comments_df = self.comments_df[pd.notnull(self.comments_df[col])]

        self.tokens, self.text_no_punct, self.text_no_stop = self._filter()
        self._load_basic_stats()
        self.freq_dist = nltk.FreqDist(self.text_no_stop)

    def _filter(self):
        # Tokenization
        # Create a custom tokenizer to remove punctuation and stopwords.
        # Stopwords are common words like 'the', 'and', 'an', etc.

        comments = self.comments_df['comment'].str.cat()

        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))

        tokens = nltk.word_tokenize(comments)
        tokens_no_punct = tokenizer.tokenize(comments)
        text_no_punct = nltk.Text(tokens_no_punct)
        text_no_stop = nltk.Text([w.lower() for w in text_no_punct.tokens if w.lower() not in stop_words])

        return tokens, text_no_punct, text_no_stop

    def _load_basic_stats(self):
        self.token_count = len(self.text_no_punct)  # Total number of tokens
        self.token_unique = len(set(self.text_no_punct))  # number of unique tokens
        self.lexical_diversity = len(set(self.text_no_punct)) / len(self.text_no_punct)  # lexical diversity

    def _about(self):
        print(f"NLTK: {nltk.__version__}")
        print(f"Pandas: {pd.__version__}")
        print(f"matplotlib: {matplotlib.__version__}")

    def collocation_list(self):
        # print(text.collocation_list())  # <-- error ValueError: too many values to unpack (expected 2)
        return '; '.join(self.text_no_stop.collocation_list())

    def display_rawdata(self, x, y):
        pd.set_option('display.max_rows', x)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_columns', y)

        df = pd.DataFrame(self.comments_df[0:x])

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

    def display_basic_stats(self):
        print(f"\n\nBasic NLP Statistics for Corpus '{self.filename}:")
        print(f"\tToken Count: {self.token_count}")
        print(f"\tUnique Count: {self.token_unique}")
        print(f"\tLexical Diversity: {self.lexical_diversity}")

    def display_most_common(self, n):
        data = pd.DataFrame(self.freq_dist.most_common(n), columns=['Word', 'Count']).to_string(index=False)
        print(f"\n\n{n} Common Words for Corpus '{self.filename}:")
        print(data)

    def percent_of_text(self, word):
        return 100 * self.text_no_punct.count("oh") / len(self.text_no_punct)


def main():
    # Counting Words, Frequency Distributions, and Collocations
    # We can simply count the number of tokens in a corpus, the number of unique tokens (word types), the
    # lexical diversity of a corpus, the occurrences of a specific word, and the percentage that a word takes up
    # in a corpus, among many other calculations we can make. This will give us some sense of how large the
    # corpus is, how lexically rich it is and allows us to start poking around the corpus.

    # # file = 'sarc_s_meta_100k.csv'
    file = 'sarc_s_meta_10k.csv'
    # # file = 'sarc_s_meta_10.csv'

    bt = BigTalk(file)

    bt.display_rawdata(10, 12)
    bt.display_basic_stats()
    bt.display_most_common(20)

    # Common Words, Concordance Searching and Frequency Distributions
    # We can display the most common words, perform concordance searches and display frequency distributions.
    #print(bt.freq_dist.most_common(20))

    # Some specific word metrics
    word = 'oh'
    print(f"\n\nSome specific metrics for the word '{word}':")
    print(f"\tCount: {bt.text_no_punct.count(word)}")
    print(f"\tPercentage of Text: {bt.percent_of_text(word)}")

    # Concordance Search of the word 'right'
    word = 'right'
    print(f"\n\nConcordance Search of the word '{word}':")
    print(bt.text_no_punct.concordance(word))

    # Plot a Frequency Distribution of the 20 most common words
    # bt.freq_dist.plot(20, cumulative=False)
    n = 20
    print(f"\n\nPlot a Frequency Distribution of the {n} most common words:")
    bt.freq_dist.plot(20)
    # TODO: Add x and y axis labels

    # Display a lexical Dispersion Plot
    # Show a word's importance weighted by it's lexical dispersion in a corpus
    # word_list = ['people', 'yeah', 'right', 'god', 'never', 'way', 'really']
    print(f"\n\nDisplay a lexical Dispersion Plot:")
    print(f"(Show a word's importance weighted by it's lexical dispersion in a corpus)")
    n = 10
    word_list = [w[0] for w in bt.freq_dist.most_common(n)]
    bt.text_no_punct.dispersion_plot(word_list)

    # Similar words, Common Contexts, and Collocations
    word = 'never'
    print(f"\n\nSearch of words that appear in a similar range of contexts as the word '{word}':")
    print(bt.text_no_stop.similar(word))

    # Common Contexts
    word_list = ["thank", "god"]
    print(f"\n\nShow common contexts as the words '{word_list[0]}' and '{word_list[1]}':")
    print(bt.text_no_stop.common_contexts(word_list))

    # Collocations
    n = 20
    print(f"\n\nShow the {n} most common collocations:")
    print(bt.collocation_list())

    # Other metrics
    # * Number of comments in each subreddit
    # * Comment Scores
    # * Comment Length: How many words
    # * Sentence Length
    # https://andrewpwheeler.com/2016/06/08/sentence-length-in-academic-articles/


if __name__ == '__main__':
    # logging.debug("__main__")
    main()