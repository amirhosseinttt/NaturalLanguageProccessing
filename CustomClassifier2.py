import ast
from collections import Counter
import math

import spacy


class Classifier:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.max_length = 1500000  # or any large value, as long as you don't run out of RAM

    def _fill_diff_dictionary(self, text_list):
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

    def _tf_idf(self, diff_dict, mlist, word: str, paragraph_index, number_of_paragraphs):
        tmp_list = mlist[diff_dict.get(word)]
        tf = 0
        values = list(Counter(tmp_list).values())
        keys = Counter(tmp_list).keys()
        for index, key in enumerate(keys):
            if key == paragraph_index:
                tf = values[index]
                break

        df = len(keys)
        idf = math.log(number_of_paragraphs / df)
        return tf * idf


    def run(self):
        overview_list = []
        for index, row in self.train_df.iterrows():
            # if index > 10: break  # DELETE THIS LINE
            if index % 100 == 0:
                print(index)
            genres_list = row['genres']
            overview = row['overview']
            overview_list.append(overview)
            x = ast.literal_eval(genres_list)

        diff_dict, mlist = self._fill_diff_dictionary(overview_list)
        print(len(diff_dict))
        print(self._tf_idf(diff_dict,mlist,"is",0,2000))
