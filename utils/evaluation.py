import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pytrec_eval

class Evaluation:
    
    def prepare_data_for_pytreceval(top_k_G, top_k_v):
        qrels = {}
        runs = {}

        for word, sim_words_scores in top_k_G.items():
            # Ground truth: Word's similar words from SimLex-999 with binary relevance (1)
            qrels[word] = {sim_word: 1 for sim_word, _ in sim_words_scores}

            # Predictions: Word's similar words from Word2Vec model with similarity scores
            if word in top_k_v:
                runs[word] = {sim_word: score for sim_word, score in top_k_v[word]}

        return qrels, runs
    
    def get_nDCG(top_k_v_models, top_10_G):

        ndcg_scores = {}

        for model_key, top_k_v in top_k_v_models.items():

            #qrels, runs = self.prepare_data_for_pytreceval(top_10_G, top_k_v)
            qrels = {}
            runs = {}

            for word, sim_words_scores in top_10_G.items():
                # Ground truth: Word's similar words from SimLex-999 with binary relevance (1)
                qrels[word] = {sim_word: 1 for sim_word, _ in sim_words_scores}

                # Predictions: Word's similar words from Word2Vec model with similarity scores
                if word in top_k_v:
                    runs[word] = {sim_word: score for sim_word, score in top_k_v[word]}

            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg'})
            results = evaluator.evaluate(runs)

            # Calculate average nDCG for this model
            avg_ndcg = sum(measures['ndcg'] for measures in results.values()) / len(results)
            ndcg_scores[model_key] = avg_ndcg

            print(f"Average nDCG for {model_key}: {avg_ndcg}")

        return ndcg_scores

    def evaluateWord2Vec(similarity_dict, word2vec_models):
    
        top_k_v_models = {}  # Dictionary to store top-10 similar words from each model for each word in SimLex-999

        for model_key, model in word2vec_models.items():

            print(f"Processing model: {model_key}")
            top_k_v = {}  # Dictionary for the current model

            for word in similarity_dict.keys():
                if word in model.wv.key_to_index:
                    # Word is in the model's vocabulary, find the top-10 most similar words

                    similar_words = model.wv.most_similar(word, topn=10)
                    top_k_v[word] = similar_words  # Store the similar words and their similarity scores

                else:
                    # Word is OOV (out-of-vocabulary) for this model, ignore it
                    #print(f"'{word}' is not in the vocabulary of {model_key} and will be ignored.")
                    continue

            top_k_v_models[model_key] = top_k_v  # Store the top-k results for this model

        return top_k_v_models
        
