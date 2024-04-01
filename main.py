from utils.build_evaluation_sets import buildSet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('brown')


corpus_raw, corpora = buildSet.getCorpus(corpus_list = ['news', 'romance'])

