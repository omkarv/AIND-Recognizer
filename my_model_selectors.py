import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def base_selector(self, custom_scoring_method):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        split_method = KFold()
        best_score_sum = -math.inf
        best_hmm_model = None
        for num_states in range(self.min_n_components, self.max_n_components):
            hmm_model_score_sum = 0
            if len(self.sequences) is 1:
                return GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            elif len(self.sequences) is 2:
                normalized_X, normalized_lengths = combine_sequences([0], self.sequences)
                test_X, test_lengths = combine_sequences([1], self.sequences)
                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(normalized_X, normalized_lengths)
                    hmm_model_score_sum += self.custom_scoring_method(hmm_model, test_X, test_lengths, num_states)
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))
                        print("model score is {}".format(hmm_model_score_sum))
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
            else:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    normalized_X, normalized_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(normalized_X, normalized_lengths)
                        hmm_model_score_sum += self.custom_scoring_method(hmm_model, test_X, test_lengths, num_states)
                        if self.verbose:
                            print("model created for {} with {} states".format(self.this_word, num_states))
                            print("model score is {}".format(hmm_model_score_sum))
                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(self.this_word, num_states))
            if best_hmm_model is None:
                best_hmm_model = hmm_model
            if hmm_model_score_sum > best_score_sum:
                best_score_sum = hmm_model_score_sum
                best_n = num_states
                best_hmm_model = hmm_model
        self.print_best_score()
        return best_hmm_model

    def print_best_score(self):
        if self.verbose:
            print("best score is {}".format(best_score_sum))
            print("best n is {}".format(best_n))

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def custom_scoring_method(self, hmm_model, test_X, test_lengths, num_params):
        return -2 * hmm_model.score(test_X, test_lengths) + len(test_X) * math.log10(num_params)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        return self.base_selector(self.custom_scoring_method)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def custom_scoring_method(self, hmm_model, test_X, test_lengths, num_params):
        antiLogL = 0
        for word in self.hwords:
            if word == self.this_word:
                continue
            X, lengths = self.hwords[word]
            antiLogL += hmm_model.score(X, lengths)
        return hmm_model.score(test_X, test_lengths) - 1 / ( len(self.words) * antiLogL )

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return self.base_selector(self.custom_scoring_method)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def custom_scoring_method(self, hmm_model, test_X, test_lengths, num_params):
        return hmm_model.score(test_X, test_lengths)

    def select(self):
        return self.base_selector(self.custom_scoring_method)
