import ast
import spacy
from GeneralClassifier import Classifier


class NLP:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.nlp = spacy.load('en_core_web_md')

    def _get_paragraph_mean_vector(self, paragraph: str):
        string = paragraph.replace('\n', ' ')

        doc = self.nlp(string)

        mlist = [0] * 300

        for word in doc:
            # print(word)
            mlist += word.vector

        mlist /= len(doc)

        return mlist

    def _find_different_genres(self, mlist):
        outcome = {}
        for i in mlist:
            for j in i[0]:
                outcome.setdefault(j['id'], j['name'])

        return outcome

    def _predict_and_compare(self, input_x, test_dict_list, model, different_genres_list):
        predicted_data = model.predict(input_x)
        print(predicted_data[0])
        print(different_genres_list.keys())
        print(test_dict_list[0])

    def _prepare_data_for_classifier(self, mlist, different_genres_list):
        input_x = []
        labels = []
        for row in mlist:
            input_x.append(row[1])
            tmp_list = []
            for genre_id in different_genres_list.keys():
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
        for index, row in df.iterrows():
            # if index > 10: break  # DELETE THIS LINE
            if index % 100 == 0:
                print(index)
            genres_list = row['genres']
            overview = row['overview']
            vector = self._get_paragraph_mean_vector(str(overview))
            x = ast.literal_eval(genres_list)
            mlist.append([x, vector])

        return mlist

    def run(self):

        mlist1 = self._get_mlist(self.train_df)
        mlist2 = self._get_mlist(self.test_df)

        differ_list = self._find_different_genres(mlist1)

        x_train, y_train = self._prepare_data_for_classifier(mlist1, differ_list)
        x_test, y_test = self._prepare_data_for_classifier(mlist2, differ_list)

        classifier = Classifier(x_train,y_train,x_test,y_test)
        classifier.run()
