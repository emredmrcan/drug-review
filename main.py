from time import process_time
import matplotlib.pyplot as plt
import pandas as pd

def readTrainData(path):
    start = process_time()
    print("------------READ DATA STARTED-------------")
    train = pd.read_csv(path, usecols=['drugName', 'condition', 'review', 'rating']).dropna(how='any', axis=0)
    end = process_time()
    print("Elapsed time for reading the train data in seconds:", end - start)
    print("--------------READ DATA END---------------")
    return train


def cleanData(data):
    start = process_time()
    print("------------CLEAN DATA STARTED-------------")
    data = data[~data.condition.str.contains("users found this comment helpful.", na=False)]
    filter = (data.condition != "") | (data.condition != '[aA]')
    data = data[filter].dropna()
    data = data[data.duplicated(subset=['condition'], keep=False)]  # Remove unique conditions
    data = data[data.duplicated(subset=['drugName'], keep=False)]  # Remove unique drugNamez
    data.condition.value_counts().head(10).plot(kind="bar")
    plt.show()
    data.drugName.value_counts().head(10).plot(kind="bar")
    plt.show()
    end = process_time()
    print("Elapsed time for cleaning the data in seconds:", end - start)
    print("--------------CLEAN DATA END---------------")
    return data


train = readTrainData('datasets/drugsComTrain_raw.csv')
print(train.shape)
train = cleanData(train)
print("--------------Cleaned DATA END---------------")
print(train.shape)
