import ast
import pickle
from collections import Counter
import math

import spacy

from GeneralClassifier import Classifier


class NLP:

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

    def _tf_idf(self, diff_dict, custom_list, word: str, paragraph_index, mlist):
        tmp_list = custom_list[diff_dict.get(word)]
        tf = 0
        # paragraph = mlist[paragraph_index][1]
        # try:
        #     words = self.nlp(paragraph)
        #     for w in words:
        #         if str(w) == word:
        #             tf += 1
        # except Exception:
        #     print(paragraph_index)
        #     print("there is an err in this Paragraph: "+str(paragraph))

        counter = Counter(tmp_list)
        values = list(counter.values())
        keys = counter.keys()
        for index, key in enumerate(keys):
            if key == paragraph_index:
                tf = values[index]
                break

        df = len(set(custom_list[diff_dict.get(word)]))
        idf = math.log(len(mlist) / df)
        return tf * idf

    def _prepare_train_data_for_classifier(self, mlist, different_words_dict, custom_list, different_genres_dict):

        input_x = []
        labels = []
        for index, row in enumerate(mlist):
            if index % 100 == 0:
                print(index)
            tmp_list = [0] * len(different_words_dict.keys())
            try:
                for word in self.nlp(row[1]):
                    key = int(different_words_dict.get(str(word)))
                    tf_idf = self._tf_idf(different_words_dict, custom_list, str(word), index, mlist)
                    tmp_list[key] = tf_idf

            except Exception:
                print("index is: "+str(index))

            input_x.append(tmp_list.copy())

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

    def _idf(self, N, df):
        return math.log(N / df)

    def _prepare_test_data_for_classifier(self, mlist1, diff_dict, custom_list, different_genres_dict, N):
        input_x = []
        labels = []
        for index, row in enumerate(mlist1):
            if index % 100 == 0:
                print(index)
            tmp_list = [0] * len(diff_dict.keys())
            try:
                for word in self.nlp(row[1]):
                    word_str = diff_dict.get(str(word))
                    if word_str is None:
                        continue
                    key = int(word_str)
                    df = len(set(custom_list[key]))
                    # print("key is: "+str(str(word))+"    df is: "+str(df))
                    idf = self._idf(N, df)
                    if idf < 0:
                        print(str(df) + " " + str(len(mlist1)))

                    tmp_list[key] += idf

            except Exception as e:
                pass

            input_x.append(tmp_list.copy())

            tmp_list1 = []
            for genre_id in different_genres_dict.keys():
                sw = True
                for dictionary in row[0]:

                    if genre_id in dict(dictionary).values():
                        tmp_list1.append(1)
                        sw = False
                        break
                if sw:
                    tmp_list1.append(0)

            labels.append(tmp_list1.copy())

        return input_x, labels

    def _find_different_genres(self, my_special_list):
        outcome = {}
        for i in my_special_list:
            for j in i[0]:
                outcome[j['id']] = j['name']

        return outcome

    def _save_data(self, x_train, y_train, x_test, y_test, path="algorithm2"):
        with open(path + "/x_train.pickle", "wb") as output_file:
            pickle.dump(x_train, output_file)

        with open(path + "/y_train.pickle", "wb") as output_file:
            pickle.dump(y_train, output_file)

        with open(path + "/x_test.pickle", "wb") as output_file:
            pickle.dump(x_test, output_file)

        with open(path + "/y_test", "wb") as output_file:
            pickle.dump(y_test, output_file)

    def _load_data(self, path="algorithm2"):
        with open(path + "/x_train.pickle", "rb") as input_file:
            x_train = pickle.load(input_file)

        with open(path + "/y_train.pickle", "rb") as input_file:
            y_train = pickle.load(input_file)

        with open(path + "/x_test.pickle", "rb") as input_file:
            x_test = pickle.load(input_file)

        with open(path + "/y_test", "rb") as input_file:
            y_test = pickle.load(input_file)

        return x_train, y_train, x_test, y_test

    def run(self):

        mlist, overview_list = self._get_mlist(self.train_df)
        diff_dict, custom_list = self._fill_diff_dictionary(overview_list)
        print(len(diff_dict))
        print(self._tf_idf(diff_dict, custom_list, "is", 0, mlist))
        print("creating x_train & y_train just strated...")
        different_genres_dict = self._find_different_genres(mlist)
        x_train, y_train = self._prepare_train_data_for_classifier(mlist, diff_dict, custom_list, different_genres_dict)
        print(x_train[0])
        print(x_train[1])
        print(x_train[2])
        print(y_train[0])
        mlist1, overview_list1 = self._get_mlist(self.test_df)
        print("creating x_test & y_test just strated...")
        x_test, y_test = self._prepare_test_data_for_classifier(mlist1, diff_dict, custom_list, different_genres_dict,
                                                                len(mlist))
        print(x_test[0])
        print(x_test[1])
        print(x_test[2])
        print(y_test[0])

        self._save_data(x_train, y_train, x_test, y_test, "algorithm2")

        # x_train, y_train, x_test, y_test = self._load_data(path="algorithm2")

        classifier = Classifier(x_train, y_train, x_test, y_test)
        classifier.run()
