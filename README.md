# Vector-vs.-Lexical-Semantics
The aim of this project is to find the correlation between lexical and vector representation of words.

# To setup the environment:
To  install the required libraries, run this:
```
pip install -r requirements.txt
```
# Quickstart:
Run this for a quick start on this repo. A copy of this can be found in the example run at /notebook_run.ipynb. 
```
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

#A sample run using 100 iterations
corpus_raw, corpora = buildSet.getCorpus(corpus_list = ['news', 'romance'])
similarity_dict = simLex.getSimDict()
top_k_g = simLex.formTop_k_g(similarity_dict, k = 10)
w2v_models = Word2Vec.buildWord2vecModels(corpora, context_windows = [1, 2, 5, 10], vector_sizes = [10, 50, 100, 300], iterations = 100)
top_k_g_w2v = Evaluation.evaluateWord2Vec(similarity_dict, w2v_models)
top_k_g_tfidf = tfidf.tfidf_models(corpus_raw, similarity_dict)
nDCG_w2v = Evaluation.get_nDCG(top_k_g_w2v, top_k_g)
nDCG_tfidf = Evaluation.get_nDCG(top_k_g_tfidf, top_k_g)
best = plot.compareWithBestModels(nDCG_w2v, nDCG_tfidf)

plot.visualize_nDCG_scores(best, graph_label = 'Word2Vec and tfidf')
```

# Datasets
The SimLex-999 corpus is chosen as the golden standard for the evaluation. The corpora selected for evaluations are the 'news' and 'romance' categories from the 'Brown' corpus. It was downloaded from the 'nltk' library.

# Steps Involved
1. Download the SimLex-999 dataset as a text file
2. Download the 'Brown' corpus from nltk
3. Create variations of Word2Vec models and Tfidf model from categories in the 'Brown' corpus
4. Find top 10 words using the models for each word in SimLex-999
5. Consider SimLex-999 as the golden truth and evaluate the retrieved top 10 using the average nDCG metric
6. Plot the results

# Configurations
The corpora can be changes to other categories of the 'Brown Corpus' by changing the parameters of the 'getCorpus' function.
1. corpus_list = []  ;  default = ['news', 'romance'] 
The configurations to the Word2Vec model can be changed with the parameters to the 'buildWord2vecModels' function. The parameters are:
1. corpora 
2. context_windows = [] ; default = [1, 2, 5, 10]
3. vector_sizes = [], ; default = [10, 50, 100, 300]
4. iterations = int ; default = 100
