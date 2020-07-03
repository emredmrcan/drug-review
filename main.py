from time import process_time
import pandas as pd


def readTrainData(path):
    start = process_time()
    print("------------READ DATA STARTED-------------")
    train = pd.read_csv(path, usecols=['drugName', 'condition', 'review', 'rating']).dropna(how='any', axis=0)
    print(train.info())
    end = process_time()
    print("Elapsed time for reading the train data in seconds:", end - start)
    print("--------------READ DATA END---------------")
    return train


train = readTrainData('datasets/drugsComTrain_raw.csv')
