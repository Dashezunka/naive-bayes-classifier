import math

from terms_dictionary import DictionaryTrie

class ClassDictionary:
    def __init__(self):
        self.samples = 0
        self.trie = DictionaryTrie()


class BayesModel:
    def __init__(self):
        self.classes_dict = {}
        self.corpus_vocabulary = set()

    def fit(self, X, y):
        for i in range(len(X)):
            tokens = X[i]
            class_label = y[i]
            if not self.classes_dict.get(class_label):
                self.classes_dict[class_label] = ClassDictionary()
            self.classes_dict[class_label].samples += 1
            self.classes_dict[class_label].trie.add_words(tokens)
            self.corpus_vocabulary.update(tokens)


    def predict(self, test_sample):
        all_samples_counter = sum([d.samples for d in self.classes_dict.values()])
        unique_words = len(self.corpus_vocabulary)
        classes_likelihood_logs = {}
        for class_label in self.classes_dict.keys():
            class_dict = self.classes_dict[class_label]
            class_probability = class_dict.samples / all_samples_counter
            class_likelihood_log = math.log(class_probability)
            for token in test_sample:
                token_entries_in_class_counter = class_dict.trie.get(token, 0)
                class_likelihood_log += math.log((1 + token_entries_in_class_counter) /
                                                 (unique_words + class_dict.trie.size))
            classes_likelihood_logs[class_label] = class_likelihood_log

        class_probability_list = []
        for cur_class, cur_class_likelihood_log in classes_likelihood_logs.items():
            other_classes_likelihood_logs = list(
                [class_likelihood_log for class_label, class_likelihood_log in classes_likelihood_logs.items()
                 if class_label != cur_class])
            cur_class_probability = 1 / ( 1 + sum([math.exp(class_likelihood_log - cur_class_likelihood_log)
                                                   for class_likelihood_log in other_classes_likelihood_logs]))
            class_probability_list.append((cur_class, cur_class_probability))
        return max(class_probability_list, key = lambda x: x[1])