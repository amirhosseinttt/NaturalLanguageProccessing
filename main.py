from Algorithm3 import NLP
import pandas as pd
import spacy


def load_data():
    train_df = pd.read_csv('utils/train.csv')
    test_df = pd.read_csv('utils/test.csv')
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_data()
    nlp = NLP(train_df, test_df)
    nlp.run()
