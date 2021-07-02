import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import ast
import spacy
from sklearn.preprocessing import StandardScaler


class Classifier:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.nlp = spacy.load('en_core_web_md')

    def _get_paragraph_mean_vector(self, paragraph: str):
        string = paragraph.replace('\n', ' ')

        doc = self.nlp(string)

        mlist = [0] * 300

        for word in doc:
            print(word)
            mlist += word.vector

        mlist /= len(doc)

        return mlist

    def _find_different_genres(self, mlist):
        outcome = {}
        for i in mlist:
            for j in i[0]:
                outcome.setdefault(j['id'], j['name'])

        return outcome

    def _train_neural_network_model(self, mlist, different_genres_list: dict):
        input_x = []
        labels = []
        for row in mlist:
            input_x.append(row[2])
            tmp_list = []
            for genre_id in different_genres_list.keys():
                if genre_id in row[0]:
                    tmp_list.append(1)
                else:
                    tmp_list.append(0)
            labels.append(tmp_list)

        model = Sequential()
        model.add(Dense(200, input_dim=len(input_x[0]), activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(len(different_genres_list.keys()), activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(np.array(input_x), np.array(labels), batch_size=32, epochs=30)

        return model

    def _predict_and_compare(self, input_x, test_dict_list, model,different_genres_list):
        predicted_data = model.predict(input_x)
        print(predicted_data[0])
        print(different_genres_list.keys())
        print(test_dict_list[0])

    def run(self):

        mlist = []
        vector_list = []
        for index, row in self.train_df.iterrows():
            # if index > 10: break  # DELETE THIS LINE
            if index%100==0:print(index)
            genres_list = row['genres']
            overview = row['overview']
            vector = self._get_paragraph_mean_vector(str(overview))
            vector_list.append(vector)
            x = ast.literal_eval(genres_list)
            mlist.append([x, vector])

        standardScaler = StandardScaler()
        standardScaler.fit(vector_list)
        transformed_vector_list = standardScaler.transform(vector_list)

        for index in range(len(mlist)):
            row = mlist[index]
            row.append(transformed_vector_list[index])

        differ_list = self._find_different_genres(mlist)
        model = self._train_neural_network_model(mlist, differ_list)

        test_dict_list = []
        vector_list = []
        for index, row in self.test_df.iterrows():
            genres_list = row['genres']
            overview = row['overview']
            vector = self._get_paragraph_mean_vector(str(overview))
            vector_list.append(vector)
            x = ast.literal_eval(genres_list)
            test_dict_list.append(x)

        transformed_vector_list = standardScaler.transform(vector_list)
        self._predict_and_compare(transformed_vector_list, test_dict_list, model,differ_list)
