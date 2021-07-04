import ast
import math
from collections import Counter
import spacy
from sklearn.cluster import KMeans

from GeneralClassifier import Classifier


class NLP:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.nlp = spacy.load('en_core_web_md')

    def _fill_diff_dictionary(self, text_list):
        """
        :param text_list:
        :return: mlist & my dict


        IMPORTANT::: this function is a copy of the original function written in CustomClassifier2
        """

        my_dict = {}
        counter = 0
        mlist = []
        for index, paragraph in enumerate(text_list):

            try:
                words = self.nlp(paragraph)
            except Exception:
                continue
            for word in words:
                if str(word) not in my_dict:
                    my_dict.setdefault(str(word), counter)
                    counter += 1
                    mlist.append([index])

                else:
                    mlist[my_dict.get(str(word))].append(index)

        return my_dict, mlist

    def _clustering(self, my_list, diff_dict, number_of_clusters):
        """
        clustering over different words using word2vec and KMeans algorithm
        """

        vector_list = []
        for key in diff_dict.keys():
            vector = self.nlp(key)[0].vector
            vector_list.append(vector)

        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(vector_list)

        print(kmeans.labels_)
        print(set(kmeans.labels_))

        my_dict = {}
        for i in list(set(kmeans.labels_)):
            tmp_list = []
            print("\n\n\n")
            for index, label in enumerate(kmeans.labels_):
                if label == i:
                    word = str(list(diff_dict.keys())[index])
                    print(word)
                    tmp_list.extend(my_list[diff_dict.get(word)])
            my_dict.setdefault(i, set(tmp_list))

        print("my dict is: " + str(my_dict))
        return kmeans, my_dict

    def _idf(self, custom_dict: dict, k_means, word: str, number_of_paragraphs):
        word_obj = self.nlp(word)[0]
        predicted_val = k_means.predict([word_obj.vector])[0]
        df = len(custom_dict.get(predicted_val))
        idf = math.log(number_of_paragraphs / df)
        return idf

    def _prepare_data_for_classifier(self, mlist, different_genres_dict, custom_dict, k_means):

        input_x = []
        labels = []
        for index, row in enumerate(mlist):
            if index % 100 == 0:
                print(index)
            tmp_list = [0] * len(custom_dict)
            try:
                for word in self.nlp(row[1]):
                    predicted_val = k_means.predict([word.vector])[0]
                    idf = self._idf(custom_dict, k_means, str(word), len(mlist))
                    tmp_list[predicted_val] += idf

            except Exception:
                pass
            input_x.append(tmp_list)

            tmp_list = []
            for genre_id in different_genres_dict.keys():
                sw = True
                for dictionary in row[0]:

                    if genre_id in dict(dictionary).values():
                        tmp_list.append(1)
                        sw = False
                        break
                if sw:
                    tmp_list.append(0)

            labels.append(tmp_list)

        return input_x, labels

    def _find_different_genres(self, mlist):
        outcome = {}
        for i in mlist:
            for j in i[0]:
                outcome.setdefault(j['id'], j['name'])

        return outcome

    def _get_mlist(self, df):
        mlist = []
        overview_list = []
        for index, row in df.iterrows():
            # if index > 10: break  # DELETE THIS LINE
            if index % 100 == 0:
                print(index)
            genres_list = row['genres']
            overview = row['overview']
            x = ast.literal_eval(genres_list)
            overview_list.append(overview)
            mlist.append([x, overview])

        return mlist, overview_list

    def run(self):

        train_mlist, train_overview_list = self._get_mlist(self.train_df)

        different_words_dict, frequency_list = self._fill_diff_dictionary(train_overview_list)
        print("K-Means algorithm just started!")
        k_means, my_dict = self._clustering(frequency_list, different_words_dict,
                                            len(different_words_dict.keys()) // 100)

        x_train, y_train = self._prepare_data_for_classifier(train_mlist, self._find_different_genres(train_mlist),
                                                             my_dict, k_means)

        test_mlist, test_overview_list = self._get_mlist(self.test_df)
        x_test, y_test = self._prepare_data_for_classifier(test_mlist, self._find_different_genres(train_mlist),
                                                           my_dict, k_means)

        classifier = Classifier(x_train, y_train, x_test, y_test)
        classifier.run()
