from nltk.corpus import brown
from nltk.tokenize import word_tokenize
                                                    
class buildSet:
  
    def getCorpus(corpus_list = ['news', 'romance']):

        corpora = {} 
        corpus_raw = {}
        for corpus_name in corpus_list:
        # Assuming you've downloaded the necessary NLTK data
            corpus = [" ".join(brown.words(fileids=field)) for field in brown.fileids(categories=corpus_name)]
        # Preprocess and tokenize as before
            corpus_tokenized = [word_tokenize(doc.lower()) for doc in corpus]
            corpora[corpus_name] = corpus_tokenized
            corpus_raw[corpus_name] = corpus

        return corpus_raw, corpora

