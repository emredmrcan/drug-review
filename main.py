#!/usr/bin/env python
# coding: utf-8

# # Emre Demircan - 4711588

# In[1]:


from time import process_time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import warnings;

warnings.simplefilter('ignore')
stop_words = set(stopwords.words('english'))


# In[2]:


def readData(path):
    start = process_time()
    print("------------READ DATA STARTED-------------")
    data = pd.read_csv(path, usecols=['drugName', 'condition', 'review', 'rating', 'usefulCount']).dropna(how='any',
                                                                                                          axis=0)
    end = process_time()
    print("Elapsed time for reading the data in seconds:", end - start)
    print("--------------READ DATA END---------------")
    return data


# In[3]:

# Data set can be downloaded from kaggle (https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018)
train_all = readData('datasets/drugsComTrain_raw.csv')
print(train_all.info())

test_all = readData('datasets/drugsComTest_raw.csv')
print(test_all.info())


# In[4]:


def numberOfDrugsPerCondition(data):
    condition_drugName = data.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
    condition_drugName.describe()
    condition_drugName[0:25].plot(kind="bar", figsize=(14, 6), fontsize=10, color="green")
    plt.xlabel("", fontsize=20)
    plt.ylabel("", fontsize=20)
    plt.title("Top25 : The number of drugs per condition.", fontsize=20)


# In[5]:


numberOfDrugsPerCondition(train_all)


# As we can see, there are some wrong values for conditions such as 'Not Listed / Other', '</span>'. These rows should be removed since we want to apply for sentiment analysis.

# In[6]:


def review_clean(review):
    # changing to lower case
    lower = review.str.lower()

    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")

    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]', ' ')

    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+', ' ')

    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$', '')

    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+', ' ')

    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')

    return dataframe


# In[7]:


print('Train Data shape before cleaning:')
print(train_all.shape)
train_all['review_clean'] = review_clean(train_all['review'])
train_all = train_all[~train_all.condition.str.contains("</span>|Not Listed", na=False)]
print('Train Data shape after cleaning:')
print(train_all.shape)

print('Test Data shape before cleaning:')
print(test_all.shape)
test_all['review_clean'] = review_clean(test_all['review'])
test_all = test_all[~test_all.condition.str.contains("</span>|Not Listed", na=False)]
print('Test Data shape after cleaning:')
print(test_all.shape)

# In[8]:


numberOfDrugsPerCondition(train_all)

# In[9]:


drugCountPerCondition = train_all.groupby('condition')['drugName'].count().sort_values(ascending=False)
drugCountPerCondition[0:20].plot(kind="bar", figsize=(14, 6), fontsize=10, color="green")
plt.xlabel("", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Top20 : The number of reviews per condition.", fontsize=20)

# In[10]:


ratingCounts = train_all['rating'].value_counts().reset_index(name='count').sort_values(by=['index'], ascending=True)
print(ratingCounts)
ratingCounts.plot(kind="bar", x='index', y='count', figsize=(14, 6), fontsize=10, color="green", legend=None)
plt.xlabel("", fontsize=20)
plt.ylabel("", fontsize=20)
plt.xticks(rotation=0)
plt.title("Number of Reviews for Different Ratings", fontsize=20)

# In[11]:


sns.boxplot(train_all.rating)
# We can see the distribution of reviews by following figure.


# In[12]:


from textblob import TextBlob
import numpy as np


def makeSentiment(data):
    sentiments = []
    for review in tqdm(data.review_clean):
        blob = TextBlob(review)
        sentiments += [blob.sentiment.polarity]
    return sentiments


# In[13]:


trainWithSentiment = train_all
trainWithSentiment["sentiment"] = makeSentiment(train_all)
trainWithSentiment.sample(10)
# trainWithSentiment.to_csv('CleanedTrainWithSentiment.csv', index=False)

testWithSentiment = test_all
testWithSentiment["sentiment"] = makeSentiment(test_all)
testWithSentiment.sample(10)
# testWithSentiment.to_csv('CleanedTestWithSentiment.csv', index=False)


# In[14]:


def readCleanedDataWithSentiment(path):
    start = process_time()
    print("------------READ DATA STARTED-------------")
    data = pd.read_csv(path)
    end = process_time()
    print("Elapsed time for reading the data in seconds:", end - start)
    print("--------------READ DATA END---------------")
    return data


# In[15]:


# trainWithSentiment = readCleanedDataWithSentiment("CleanedTrainWithSentiment.csv")
# trainWithSentiment.sample(5)

# In[16]:


# testWithSentiment = readCleanedDataWithSentiment("CleanedTestWithSentiment.csv")

# In[17]:


# Correlation coefficient between rating and sentiment results
np.corrcoef(trainWithSentiment["rating"], trainWithSentiment["sentiment"])

# In[18]:


sns.boxplot(x=np.array(trainWithSentiment["rating"]), y=np.array(trainWithSentiment["sentiment"]))
plt.xlabel("Rating")
plt.ylabel("Sentiment")
plt.title("Sentiment vs Ratings")

# In[19]:


print(np.corrcoef(trainWithSentiment["rating"], trainWithSentiment['rating'].apply(lambda x: 1 if x > 4 else 0)))
print(np.corrcoef(trainWithSentiment["rating"], trainWithSentiment['rating'].apply(lambda x: 1 if x > 5 else 0)))
print(np.corrcoef(trainWithSentiment["rating"], trainWithSentiment['rating'].apply(lambda x: 1 if x > 6 else 0)))
print(np.corrcoef(trainWithSentiment["rating"], trainWithSentiment['rating'].apply(lambda x: 1 if x > 7 else 0)))


# In[20]:


# For feature engineering.
def addFeatures(data):
    # Normalize ratings, the best split threshold has been choosen as 5.
    data['sentiment_rate'] = data['rating'].apply(lambda x: 1 if x > 5 else 0)

    # Word count in each review
    data['count_word'] = data["review_clean"].apply(lambda x: len(str(x).split()))

    # Unique word count
    data['count_unique_word'] = data["review_clean"].apply(lambda x: len(set(str(x).split())))

    # Letter count
    data['count_letters'] = data["review_clean"].apply(lambda x: len(str(x)))

    # punctuation count
    data["count_punctuations"] = data["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # upper case words count
    data["count_words_upper"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    # title case words count
    data["count_words_title"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    # Number of stopwords
    data["count_stopwords"] = data["review"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

    # Average length of the words
    data["mean_word_len"] = data["review_clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    return data


# In[21]:


trainWithAllFeatures = addFeatures(trainWithSentiment)
trainWithAllFeatures.sample(5)

# In[22]:


testWithAllFeatures = addFeatures(testWithSentiment)

# In[23]:


# Correlation map of the features
plt.rcParams['figure.figsize'] = [17, 15]
sns.set(font_scale=1.2)
corr = trainWithAllFeatures.select_dtypes(include=['int64', 'float64']).corr()
sns_ = sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.setp(sns_.get_xticklabels(), rotation=45);

# In[24]:


# Label Encoding Drugname and Conditions
from sklearn.preprocessing import LabelEncoder


def encodeLabels(data):
    label_encoder_feat = {}
    for feature in ['drugName', 'condition']:
        label_encoder_feat[feature] = LabelEncoder()
        data[feature] = label_encoder_feat[feature].fit_transform(data[feature])
        # If decoding is needed for future, use label_encoder_feat[feature].inverse_transform(data[feature])
    return data


# In[25]:


trainWithAllFeatures = encodeLabels(trainWithAllFeatures)
testWithAllFeatures = encodeLabels(testWithSentiment)

# In[26]:


trainWithAllFeatures.head(3)

# In[27]:


# Importing Libraries for the Machine Learning Model
from xgboost import XGBClassifier
from lightgbm import LGBMModel, LGBMClassifier, plot_importance
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

# In[28]:


print(trainWithAllFeatures.isnull().sum().sum())
# We have just 2 na values so we can drop them it will not have significant impact.
trainWithAllFeatures = trainWithAllFeatures.dropna()
trainWithAllFeatures.reset_index(drop=True).head(5)

# In[29]:


# Defining Features and splitting the data as train,test and validation set

necessaryFeatures = ['drugName', 'condition', 'sentiment', 'usefulCount', 'count_word', 'count_unique_word',
                     'count_letters', 'count_punctuations', 'count_words_upper', 'count_words_title',
                     'count_stopwords']
features = trainWithAllFeatures[necessaryFeatures]
target = trainWithAllFeatures['sentiment_rate']
X_test = testWithAllFeatures[necessaryFeatures]
y_test = testWithAllFeatures['sentiment_rate']
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.3, random_state=42)
print("The Train set size ", X_train.shape)
print("The Validation set size ", X_val.shape)
print("The Test set size ", X_test.shape)


# In[30]:


def printResults(test, pred):
    print("The Accuracy of the model is : ", accuracy_score(test, pred))
    print("The confusion Matrix is ")
    print(confusion_matrix(test, pred))
    print("The classification results are ")
    print(classification_report(test, pred))


# # AdaBoost

# In[31]:


# def evaluate_model(Xtrain, Ytrain, Xval, Yval, params):
#     model = AdaBoostClassifier(**params)
#     model.fit(Xtrain,Ytrain)
#     return metrics.log_loss(Yval, model.predict(Xval))


# In[32]:


# param_grid = {
#     'learning_rate': [0.8,1,1.2],
#     'n_estimators': [100, 500, 1000, 5000],
# }

# print('Tuning begins...')
# best_eval_score = 0
# for i in tqdm(range(50)):
#     params = {k: np.random.choice(v) for k, v in param_grid.items()}
#     score = evaluate_model(X_train, y_train, X_val, y_val, params)
#     if score < best_eval_score or best_eval_score == 0:
#         best_eval_score = score
#         best_params = params
# print("Best evaluation logloss", best_eval_score)
# print(best_params)


# In[33]:


clf = AdaBoostClassifier(n_estimators=500,
                         learning_rate=1)
start = process_time()
model_ab = clf.fit(X_train, y_train)
predictions_0 = model_ab.predict(X_val)
end = process_time()
print("Elapsed time for AdaBoostClassifier in seconds:", end - start)
printResults(y_val, predictions_0)

# In[34]:


importances = model_ab.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = list(X_train.columns.values)
names = [feature_names[i] for i in indices]
plt.figure()
plt.title("Feature Importance")
plt.barh(range(X_train.shape[1]), importances[indices])
plt.yticks(range(X_train.shape[1]), names)
plt.show()

# # Lightgbm

# In[35]:


from sklearn import metrics

# def evaluate_model(Xtrain, Ytrain, Xval, Yval, params):
#     model = LGBMClassifier(num_iterations = 300,
#                            silent=-1,
#                            verbose=-1,
#                            **params)
#     model.fit(Xtrain,Ytrain)
#     return metrics.log_loss(Yval, model.predict(Xval))


# In[36]:


# param_grid = {
#     'n_estimators': [1000, 5000, 10000],
#     'num_leaves': [60, 80, 100],
#     'learning_rate': [0.10,0.15,0.20],
#     'subsample': [0.5, 0.7, 0.9],
#     'max_depth': [5, 7, 9],
#     'reg_alpha': [0.1, 0.3, 0.5],
#     'reg_lambda': [0.3, 0.5, 0.7]
# }

# print('Tuning begins...')
# best_eval_score = 0
# for i in range(60):
#     params = {k: np.random.choice(v) for k, v in param_grid.items()}
#     score = evaluate_model(X_train, y_train, X_val, y_val, params)
#     if score < best_eval_score or best_eval_score == 0:
#         best_eval_score = score
#         best_params = params
# print("Best evaluation logloss", best_eval_score)
# print(best_params)
# {'n_estimators': 10000,
#  'num_leaves': 100,
#  'learning_rate': 0.2,
#  'subsample': 0.5,
#  'max_depth': 9,
#  'reg_alpha': 0.5,
#  'reg_lambda': 0.3}


# In[37]:


clf = LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.2,
    num_leaves=100,
    subsample=.5,
    max_depth=9,
    reg_alpha=.5,
    reg_lambda=.3,
    silent=-1,
    verbose=-1,
)
# Reg: regularization terms on weights.
start = process_time()
model_lgbm = clf.fit(X_train, y_train)
predictions_1 = model_lgbm.predict(X_val)
end = process_time()
print("Elapsed time for LGBMClassifier in seconds:", end - start)
printResults(y_val, predictions_1)

# In[38]:


plt.rcParams['figure.figsize'] = [12, 9]
sns.set(style='whitegrid', font_scale=1.2)
plot_importance(model_lgbm);

# # XGBoost

# In[39]:


# def evaluate_model(Xtrain, Ytrain, Xval, Yval, params):
#     model = XGBClassifier(**params)
#     model.fit(Xtrain,Ytrain)
#     return metrics.log_loss(Yval, model.predict(Xval))


# In[40]:


# param_grid = {
#     'learning_rate': [0.10,0.15,0.20],
#     'subsample': [0.5, 0.7, 0.9],
#     'max_depth': [5, 7, 9],
# }

# print('Tuning begins...')
# best_eval_score = 0
# for i in range(60):
#     params = {k: np.random.choice(v) for k, v in param_grid.items()}
#     score = evaluate_model(X_train, y_train, X_val, y_val, params)
#     if score < best_eval_score or best_eval_score == 0:
#         best_eval_score = score
#         best_params = params
# print("Best evaluation logloss", best_eval_score)
# print(best_params)
# {'learning_rate': 0.2,
#  'subsample': 0.9,
#  'max_depth': 9}


# In[41]:


from xgboost import plot_importance

xgb_clf = XGBClassifier(learning_rate=0.2,
                        subsample=.9,
                        max_depth=9, )

start = process_time()
model_xgb = xgb_clf.fit(X_train, y_train)
predictions_2 = model_xgb.predict(X_val)
end = process_time()
print("Elapsed time for XGBClassifier in seconds:", end - start)
printResults(y_val, predictions_2)

# In[42]:


plt.rcParams['figure.figsize'] = [12, 9]
plot_importance(model_xgb);

# # Naive Bayes

# In[43]:


from sklearn.naive_bayes import GaussianNB

# Naive Bayes doesn't have any hyperparameters to tune.
clf = GaussianNB()
model_nb = clf.fit(X_train, y_train)
predictions_3 = model_nb.predict(X_val)
printResults(y_val, predictions_3)

# # Logistic Regression

# In[44]:


clf = LogisticRegression()
start = process_time()
model_lr = clf.fit(X_train, y_train)
predictions_4 = model_lr.predict(X_val)
end = process_time()
print("Elapsed time for LogisticRegression in seconds:", end - start)
printResults(y_val, predictions_4)

# In[45]:


# The best model is Lightgbm with accuracy 0.8376294060941364
# Let's use it on our test set to see results.

predictions_for_test = model_lgbm.predict(X_test)
printResults(y_test, predictions_for_test)

