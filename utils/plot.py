import matplotlib.pyplot as plt

class plot:

    def visualize_nDCG_scores(ndcg_scores):
  
        # Data for plotting
        model_keys = list(ndcg_scores.keys())
        avg_ndcg_values = [ndcg_scores[key] for key in model_keys]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(model_keys, avg_ndcg_values, color='skyblue')
        plt.xlabel('Average nDCG')
        plt.title('Average nDCG Scores by tfidf Model Configuration')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.show()

    def compareWithBestModels(nDCGsWord2Vec, nDCGstfidf):
        
        offset = 0
        lim = int(len(nDCGsWord2Vec)/len(nDCGstfidf))
        while(offset <= lim):
            max_key = max(list(nDCGsWord2Vec.keys())[offset: offset+ lim], key=nDCGsWord2Vec.get)
            offset = offset + lim
            nDCGstfidf[max_key] = nDCGsWord2Vec[max_key]

        return nDCGstfidf