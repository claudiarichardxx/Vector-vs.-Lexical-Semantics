import nltk
from nltk.tokenize import word_tokenize

import string
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


class simLex:

    def getSimDict(golden_truth_path = 'data\EN-SIMLEX-999.txt'):

        similarity_dict = {}
        with open(golden_truth_path, 'r') as file:
            next(file)  # Skip header line if there is one
            for line in file:
                contents = line.strip().split()
                first, second = contents[:2]  # Words are typically the first two columns
                score = float(contents[2])  # Assuming the similarity score is the fourth column

                if first not in similarity_dict:
                    similarity_dict[first] = []
                similarity_dict[first].append([second, score])

                if second not in similarity_dict:
                    similarity_dict[second] = []
                similarity_dict[second].append([first, score])

        return similarity_dict


    def formTop_k_g(similarity_dict, k = 10):

        top_k_g = {}

        for word, sim_list in similarity_dict.items():
            # Sort based on similarity score in descending order
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            # Apply transitivity rule if the list has fewer than 10 items
            if len(sorted_sim_list) < k:
                expanded_list = sorted_sim_list.copy()

                for wordd, score in sim_list:
                        # If we have enough words, stop
                        if len(expanded_list) >= 10:
                            break
                        # Add words similar to 'word' that are not already in the list
                        for sim_word, sim_score in similarity_dict.get(wordd, []):

                            if sim_word not in [w for w, score in expanded_list]:
                                expanded_list.append([sim_word, sim_score])
                                if len(expanded_list) >= k:
                                    break

            top_k_g[word] = expanded_list[:k]

        return top_k_g