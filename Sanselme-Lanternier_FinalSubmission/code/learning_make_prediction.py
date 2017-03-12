
# coding: utf-8

import random
from datetime import datetime, timedelta
import io
import numpy as np
import heapq
import json
import operator
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
import json
from imblearn.over_sampling import  SMOTE
import numpy.random as nprnd


stop_words = get_stop_words('english')
path_to_data = '../data/'


# Load Files


training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
#training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data+"training_info2.csv",sep=',', header=0, index_col=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
#test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)
test_info = pd.read_csv(path_to_data+"test_info2.csv",sep=',', header=0, index_col=0)

global sent_to
with io.open('../data/sent_to.json') as json_data:
    sent_to = json.load(json_data)

global received_from
with io.open('../data/received_from.json') as json_data:
    received_from = json.load(json_data)

cut_indexes = {datetime(2001, 6, 24): 428724,
               datetime(2001, 7, 24): 927522,
               datetime(2001, 8, 24): 1153398}


# Create datetime format


# Correct dates and put datetime format
# We do that because we noticed test_set is only composed of email posterior to the ones of train_set.
# Datetime format allows to simulate posteriority in our train/test split
from datetime import datetime

for row in training_info.sort_values(by='date').iterrows():
    date = row[1]['date']
    if date[:3] == '000':
        date = '2' + date[1:]

    training_info.loc[row[0], 'date'] = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

for row in test_info.sort_values(by='date').iterrows():
    date = row[1]['date']

    test_info.loc[row[0], 'date'] = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


# Useful functions


def most_similar_sklearn(array_embedding_sparse, mail_tfidf, n):

    similarities = cosine_similarity(array_embedding_sparse, mail_tfidf)
    closest_ids = similarities[:,0].argsort()[::-1]

    return closest_ids[:n], similarities

def get_sender(query_mid, training):
    for row in training.iterrows():
        mids = row[1]['mids'].split()
        for mid in mids:
            if int(mid) == query_mid:
                sender = row[1]['sender']
                break
    return sender


def get_10_recipients(closest_ids_per_sender, training_info, similarities, closest_emails_dates):
    dic_of_recipients = {}
    dic_recency2 = {}
    #weight = len(closest_ids_per_sender)+1
    for idx in closest_ids_per_sender:
        recipients = training_info.loc[idx,'recipients'].split()
        for recipient in recipients:
            if '@' in recipient:
                dic_of_recipients[recipient] = dic_of_recipients.get(recipient, 0) + similarities[idx][0]
                dic_recency2[recipient] = dic_recency2.get(recipient, 0) + closest_emails_dates['weight_date'][idx]
    # the max here is a precaution not to divide by zero in the case were no similarity is found (happened with 'this is a testds')

    norm = max(sum(dic_of_recipients.values()), 0.0000001)
    norm_recency = max(sum(dic_recency2.values()), 0.0000001)
    for k,v in dic_of_recipients.iteritems():
        dic_of_recipients[k] = float(v)/norm
        dic_recency2[k] = float(dic_recency2[k])/norm_recency

    return dic_of_recipients, dic_recency2

def get_recency_features(X_train_info_sender, mail_date, n_recency_features):
    dic_recency = {}
    df_last_sent_emails = X_train_info_sender[X_train_info_sender.date< mail_date].sort_values(by = 'date', ascending = False)[:n_recency_features]
    for row in df_last_sent_emails.iterrows():
        recipients = row[1]['recipients'].split()
        for recipient in recipients:
            if '@' in recipient:
                dic_recency[recipient] = dic_recency.get(recipient, 0) + 1
    norm = sum(dic_recency.values())
    for k,v in dic_recency.iteritems():
        dic_recency[k] = float(v)/norm

    return dic_recency

def mean_ap(suggested_10_recipients, ground_truth):
    MAP = 0
    correct_guess = 0
    for i, suggestion in enumerate(suggested_10_recipients):
        if suggestion in ground_truth:
            correct_guess +=1
            MAP += float(correct_guess)/(i+1)
    MAP = float(MAP)/min(10, len(ground_truth))
    return MAP

def header_address_ressemblance(text, address):
    head = text[:10].lower()
    name = address[:address.index('@')].split('.')
    for n in name:
        if len(n)>2:
            if n in head:
                return True
    return False

def generate_features(X_train_info_sender, mail_tfidf, mail_date, ground_truth, sender, n, mail_header):

    #print X_train_info_sender.shape
    index_sender = X_train_info_sender.index.values
    X_train_info_sender.index = range(X_train_info_sender.shape[0])
    array_embedding_sparse_sender = array_embedding_sparse[index_sender]

    closest_ids_per_sender, similarities = most_similar_sklearn(array_embedding_sparse_sender, mail_tfidf, n)

    closest_emails_dates = pd.DataFrame(X_train_info_sender['date'][closest_ids_per_sender].sort_values())
    closest_emails_dates['weight_date'] = range(1, len(closest_ids_per_sender)+1)

    #dic_recency = get_recency_features(X_train_info_sender, mail_date, n_recency_features)

    dic_of_recipients, dic_recency2 = get_10_recipients(closest_ids_per_sender, X_train_info_sender, similarities, closest_emails_dates)
    if mail_header:
        new_features_per_mail = np.zeros((len(dic_of_recipients), 5))
    else:
        new_features_per_mail = np.zeros((len(dic_of_recipients), 4))

    labels_per_mail = np.zeros((len(dic_of_recipients), 1))
    index = 0
    for k,v in dic_of_recipients.iteritems():
        KNNScore = v
        NSF = sent_to[sender][k]
        NRF = 0
        if sender in received_from.keys():
            NRF = received_from[sender].get(k, 0)

        recency = dic_recency2[k]

        if ground_truth != None:
            if k in ground_truth:
                labels_per_mail[index, :] = 1
        if mail_header:
            head = 1.0 * header_address_ressemblance(mail_header, k)
            new_features_per_mail[index, :] = [KNNScore, NSF, NRF, recency, head]
        else:
            new_features_per_mail[index, :] = [KNNScore, NSF, NRF, recency]
        index +=1

    return new_features_per_mail, labels_per_mail, dic_of_recipients


# Train/test split


#Declare Global variables:
global X_train_info
global X_test_info
global array_embedding_sparse


#Choose here to prepare .csv for submission or to test a model locally
submission = True
training_info = training_info.sort_values(by='date')

if submission:
    # submission procedure
    X_train_info = training_info
    X_test_info = test_info

else:
    # test procedure
    split_date=datetime(2001, 8, 24)
    X_train_info = training_info[training_info.date <= split_date]

    #Randomize selection of test set:
    X_test_info = training_info[training_info.date > split_date]
    mask = nprnd.choice(range(X_test_info.shape[0]), size=1000, replace=False)
    X_test_info.index = range(X_test_info.shape[0])
    X_test_info = X_test_info[X_test_info.index.isin(mask)]


if submission:
    tfidf = TfidfVectorizer(stop_words = stop_words)
    array_embedding_sparse = tfidf.fit_transform(np.concatenate((X_train_info['body'].values,X_test_info['body'].values)))
    array_embedding_sparse = array_embedding_sparse[:X_train_info.shape[0]]
else:
    tfidf = TfidfVectorizer(stop_words = stop_words)
    array_embedding_sparse = tfidf.fit_transform(X_train_info['body'].values)


# Load previously loaded features


new_features_all = np.load('../data/new_features_all_normalized_header_recency_70.npy')
labels_all = np.ravel(np.load('../data/labels_all_normalized_header_recency_70.npy'))


# Without NRF feature:
new_features_all = new_features_all[:, [0,1,3,4]]


# Train model

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools


t1 = datetime.now()

_classifier = 'LR'

if _classifier == 'LinearSVM':

    SVM = LinearSVC(dual=False, class_weight='balanced', C=0.001)
    if submission == False:
        SVM.fit(new_features_all[:cut_indexes[split_date], :], labels_all[:cut_indexes[split_date]])
    else:
        SVM.fit(new_features_all, labels_all)
    classifier = SVM
elif _classifier == 'LR':

    LR = LogisticRegression()
    if submission == False:
        LR.fit(new_features_all[:cut_indexes[split_date], :], labels_all[:cut_indexes[split_date]])
    else:
        LR.fit(new_features_all, labels_all)
    classifier = LR

elif _classifier == 'ABC':
    ABC = AdaBoostClassifier()
    if submission == False:
        ABC.fit(new_features_all[:cut_indexes[split_date], :], labels_all[:cut_indexes[split_date]])
    else:
        ABC.fit(new_features_all, labels_all)
    classifier = ABC

elif _classifier == 'RFC':
    RFC = RandomForestClassifier(n_estimators=50, class_weight='balanced')
    if submission == False:
        RFC.fit(new_features_all[:cut_indexes[split_date], :], labels_all[:cut_indexes[split_date]])
    else:
        RFC.fit(new_features_all, labels_all)
    classifier = RFC

elif _classifier == 'SVM':
    SVM = SVC(kernel='rbf')
    if submission == False:
        SVM.fit(new_features_all[:cut_indexes[split_date], :], labels_all[:cut_indexes[split_date]])
    else:
        SVM.fit(new_features_all, labels_all)


print datetime.now() - t1


# Create test features and make prediction

# In[44]:


n = 30

#re-arrange train index
X_train_info.index = range(X_train_info.shape[0])
t_all = datetime.now()
t_100 = datetime.now()
results = pd.DataFrame(columns=['recipients'])
results.index.name = 'mid'
all_mean_ap = []
all_ground_truth = []
all_suggestions = []

count=0
for query_id in X_test_info.index.values:

    count+=1
    if count%100==0:
        print count
        print datetime.now()-t_100
        t_100 = datetime.now()

    mail = X_test_info['body'][query_id]
    mail_tfidf = tfidf.transform([mail])
    mail_date = X_test_info['date'][query_id]
    if submission:
        ground_truth = None
        query_mid = X_test_info['mid'][query_id]
    else:
        ground_truth = X_test_info['recipients'][query_id].split()
    sender = X_test_info['sender'][query_id]

    X_train_info_sender = X_train_info[(X_train_info.sender == sender) & (X_train_info.date<mail_date)]
    if X_train_info_sender.shape[0] == 0:
        continue

    # Compute Features For this email
    new_features_per_mail, labels_per_mail, dic_of_recipients = generate_features(X_train_info_sender, mail_tfidf, mail_date, ground_truth, sender, n, mail[:10])
    # Once the features are computed, we can predict the 10 recipients
    if _classifier == 'LinearSVM':
        order = classifier.decision_function(new_features_per_mail).argsort()[::-1]
    else:
        order = classifier.predict_proba(new_features_per_mail[:, [0,1,3,4]])[:,1].argsort()[::-1]
    recipients = np.array(dic_of_recipients.keys())
    suggested_10_recipients = recipients[order][:10]

    if submission:
        string_recipients = ''
        for k in suggested_10_recipients:
            string_recipients+=k + ' '
        results.loc[query_mid, 'recipients'] = string_recipients
    else:

        all_suggestions.append(suggested_10_recipients)
        all_ground_truth.append(ground_truth)
        all_mean_ap.append(mean_ap(suggested_10_recipients, ground_truth))

print "total took:", datetime.now()-t_all


# Save as .csv for submission
save=True
if save:
    results.to_csv('../submission/learning_basic_LR_with_new_features_without_NRF_2.csv')
