import ast
import numpy as np
import spacy
from sklearn.cluster import KMeans


class Classifier:

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

    def _clustering(self, diff_dict, number_of_clusters):
        """
        clustering over different words using word2vec and KMeans algorithm
        """

        vector_list = []
        for key in diff_dict.keys():
            vector = self.nlp(key).vector
            vector_list.append(vector)

        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(vector_list)

        print(kmeans.labels_)

    def run(self):
        overview_list = []
        for index, row in self.train_df.iterrows():
            if index > 10: break  # DELETE THIS LINE
            if index % 100 == 0:
                print(index)
            genres_list = row['genres']
            overview = row['overview']
            overview_list.append(overview)
            x = ast.literal_eval(genres_list)

        diff_dict, mlist = self._fill_diff_dictionary(overview_list)
        self._clustering(diff_dict, len(diff_dict.keys()) // 5)
