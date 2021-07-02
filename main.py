from CustomClassifier2 import Classifier
import pandas as pd
import spacy


def load_data():
    train_df = pd.read_csv('utils/train.csv')
    test_df = pd.read_csv('utils/test.csv')
    return train_df, test_df



if __name__ == "__main__":
    train_df, test_df = load_data()
    classifier = Classifier(train_df,test_df)
    classifier.run()
