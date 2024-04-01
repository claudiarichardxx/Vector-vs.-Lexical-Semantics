from gensim.models import Word2Vec
import itertools

class Word2Vec:

    def buildWord2vecModels(corpora):
        # Define the parameters for experimentation
        context_windows = [1, 2, 5, 10]
        vector_sizes = [10, 50, 100, 300]
        iterations = 100  # Fixed number of iterations

        # Store the models in a dictionary for easy access
        word2vec_models = {}

        for corpus in corpora.keys():

            print('Training on corpus: ', corpus)
            for window, size in itertools.product(context_windows, vector_sizes):
                model_key = "Word2Vec_window_"+ str(window) +"_vector_size_"+ str(size) + "_" + corpus
                print("Training Word2Vec with window : "+ str(window) +" and vector size : "+ str(size))

                model = Word2Vec(sentences = corpora[corpus], vector_size = size, window = window,
                                min_count = 1, workers = 4, epochs = iterations)

                word2vec_models[model_key] = model

                print(f"{model_key} is done.")
                
        return word2vec_models
    
