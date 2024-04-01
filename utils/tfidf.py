from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class tfidf:
    
    def tfidf_models(corpus_raw, similarity_dict):
  
        top_k_v_tfidf_models = {}

        for corpus in corpus_raw.keys():
            # Initialize the TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer()

            # Fit and transform the corpus to compute TF-IDF values for words
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_raw[corpus])

            # Get the vocabulary from the TF-IDF vectorizer
            vocabulary = tfidf_vectorizer.get_feature_names_out()
            # Calculate cosine similarity between word vectors
            word_similarity_matrix = cosine_similarity(tfidf_matrix.T, tfidf_matrix.T)

            # Create a dictionary to store the top 10 similar words for each word
            top_10_similar_words = {}

            # Iterate over each word in the vocabulary
            for idx, word in enumerate(similarity_dict.keys()):
                # Get the cosine similarity scores for the current word
                
                cosine_similarities = word_similarity_matrix[idx]

                # Sort the cosine similarity scores in descending order and get the indices of the top 10 similar words
                top_indices = cosine_similarities.argsort()[:-11:-1]

                # Get the top 10 similar words and their cosine similarity scores
                similar_words = [(vocabulary[i], cosine_similarities[i]) for i in top_indices if i != idx]

                # Store the top 10 similar words for the current word
                top_10_similar_words[word] = similar_words

            top_k_v_tfidf_models['tfidf_'+ corpus] = top_10_similar_words
            
        return top_k_v_tfidf_models