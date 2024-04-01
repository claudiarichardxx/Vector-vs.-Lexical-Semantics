from utils.build_evaluation_sets import buildSet
from utils.build_vocab import simLex
from utils.word_2_vec import Word2Vec
from utils.tfidf import tfidf
from utils.evaluation import Evaluation
from utils.plot import plot
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('brown')


corpus_raw, corpora = buildSet.getCorpus(corpus_list = ['news', 'romance'])
similarity_dict = simLex.getSimDict()
top_k_g = simLex.formTop_k_g(similarity_dict, k = 10)
w2v_models = Word2Vec.buildWord2vecModels(corpora, context_windows = [1, 2, 5, 10], vector_sizes = [10, 50, 100, 300], iterations = 10)
top_k_g_w2v = Evaluation.evaluateWord2Vec(similarity_dict, w2v_models)
top_k_g_tfidf = tfidf.tfidf_models(corpus_raw, similarity_dict)
nDCG_w2v = Evaluation.get_nDCG(top_k_g_w2v, top_k_g)
nDCG_tfidf = Evaluation.get_nDCG(top_k_g_tfidf, top_k_g)
best = plot.compareWithBestModels(nDCG_w2v, nDCG_tfidf)

