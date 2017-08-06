import warnings
import math
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    input_word_list = test_set.get_all_sequences()
    hwords = test_set.get_all_Xlengths()

    for word_id in range(len(input_word_list)):
        best_score = -math.inf
        best_word = ""
        X, lengths = hwords[word_id]
        probability_dict = {}
        for word in models:
            model = models[word]
            try:
                score = model.score(X, lengths)
                if score > best_score:
                    best_score = score
                    best_word = word
                probability_dict[word] = score
            except:
                print("failure to score word: {} using model".format(word))
        probabilities.append(probability_dict)
        guesses.append(best_word)
    return (probabilities, guesses)
