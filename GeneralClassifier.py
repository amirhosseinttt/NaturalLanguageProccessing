import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


class Classifier:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _train_neural_network_model(self, x_train, y_train):
        input_dim = len(x_train[0])
        output_dim = len(y_train[0])

        model = Sequential()
        # model.add(Dense(input_dim // 2, input_dim=input_dim, activation='relu'))
        # input_dim /= 2
        # while input_dim // 2 > output_dim:
        #     model.add(Dense(input_dim // 2, activation='relu'))
        #     model.add(Dense(input_dim // 2, activation='relu'))
        #     input_dim /= 2

        model.add(Dense(200,input_dim=input_dim,activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(20, activation='relu'))

        model.add(Dense(output_dim, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(np.array(x_train), np.array(y_train), batch_size=32, epochs=30)

        return model

    def _predict_nn_model(self, model, x_test):
        predicted_list = model.predict(np.array(x_test))
        outcome = []
        for p_list in predicted_list:
            tmp_list = []
            for i in p_list:
                if i > 0.5:
                    tmp_list.append(1)
                else:
                    tmp_list.append(0)
            outcome.append(tmp_list)

        return outcome

    def _evaluate(self, true_label, predicted_label):
        counter = 0
        for i in range(len(true_label)):
            print("predicted:  " + str(predicted_label[i]))
            print("true label: " + str(true_label[i]))
            print()
            if true_label[i] == predicted_label[i]:
                counter += 1

        return counter / len(true_label)

    def accuracy(self,trueL, predictL):
        all = len(trueL)
        trues = all
        for i in range(all):
            for j in range(len(trueL[0])):
                if trueL[i][j] != predictL[i][j]:
                    trues -= 1
                    break
        return trues / all

    def run(self):
        # standardScaler = StandardScaler()
        # standardScaler.fit(self.x_train)
        # self.x_train = standardScaler.transform(self.x_train)
        # self.x_test = standardScaler.transform(self.x_test)

        model = self._train_neural_network_model(self.x_train, self.y_train)
        predicted_list = self._predict_nn_model(model, self.x_test)
        print(self._evaluate(self.y_test, predicted_list))
        print(self.accuracy(self.y_test, predicted_list))
