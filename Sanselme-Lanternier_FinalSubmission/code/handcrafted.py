
# coding: utf-8

# In[1]:

import numpy as np
import numpy.random as nprnd
import pandas as pd
import heapq
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
stop_words = get_stop_words('english')


# In[2]:

path_to_data = '../data/'

##########################
# load files #                           
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
#training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data+"training_info2.csv",sep=',', header=0, index_col=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
#test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)
test_info = pd.read_csv(path_to_data+"test_info2.csv",sep=',', header=0, index_col=0)


# In[3]:

# Correct dates and put datetime format
# We do that because we noticed test_set is only composed of email posterior to the ones of train_set. 
# Datetime format allows to simulate posteriority in our train/test split
from datetime import datetime

for row in training_info.sort(['date']).iterrows():
    date = row[1]['date']
    if date[:3] == '000':
        date = '2' + date[1:]
        
    training_info.loc[row[0], 'date'] = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

for row in test_info.sort(['date']).iterrows():
    date = row[1]['date']
        
    test_info.loc[row[0], 'date'] = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


# In[4]:

submission = True
training_info = training_info.sort_values(by='date')

# affect a training set and a testing set
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


# In[5]:

##########################
# Functions #                           
##########################

# Check if the name is in the beguinning of the email
def header_address_ressemblance(text, address):
    head = text[:10].lower()
    name = address[:address.index('@')].split('.')
    head_bonus = 0.
    for i,n in enumerate(name):
        if len(n)>1:
            if n in head:
                head_bonus = 1.0
    return head_bonus

# Compute score
def mean_ap(suggested_10_recipients, ground_truth):
    MAP = 0
    correct_guess = 0
    for i, suggestion in enumerate(suggested_10_recipients):
        if suggestion in ground_truth:
            correct_guess +=1
            MAP += float(correct_guess)/(i+1)
    MAP = float(MAP)/min(10, len(ground_truth))
    return MAP

def most_similar_sklearn(array_embedding_sparse, mail_tfidf):
    similarities = cosine_similarity(array_embedding_sparse, mail_tfidf)
    closest_ids = similarities[:,0].argsort()[::-1]
    return closest_ids, similarities

def get_n_closest_emails(sender, nb_neigh, closest_ids, info):
    # Get the closest emails WRITTEN BY THE SENDER
    closest_ids_of_sender = []
    for idx in closest_ids:
        sender_wrote_id = sender == info['sender'][idx]
        if sender_wrote_id:
            closest_ids_of_sender.append(idx)
        if len(closest_ids_of_sender) == nb_neigh:
            break
    return closest_ids_of_sender

def get_10_recipients(sender, closest_ids_of_sender, training_info, closest_emails_dates, similarities, mail_header, mail_tfidf):
    dic_of_recipients = {}
    for idx in closest_ids_of_sender:
        recipients = training_info.loc[idx,'recipients'].split()
        # Loop over recipients to evaluate their score
        for recipient in recipients:
            if ('@' in recipient):
                head = header_address_ressemblance(mail_header, recipient)
                weight = similarities[idx]*closest_emails_dates['weight_date'][idx]/len(closest_ids_of_sender)
                if recipient not in dic_of_recipients.keys():
                    dic_of_recipients[recipient] = weight
                else:
                    dic_of_recipients[recipient] += weight
    suggested_10_recipients = heapq.nlargest(10, dic_of_recipients, key=dic_of_recipients.get)
    
    return suggested_10_recipients


# In[6]:

# Compute tf-idf for all emails
tfidf = TfidfVectorizer(stop_words = stop_words)
array_embedding_sparse = tfidf.fit_transform(X_train_info['body'].values)

#re-arrange train index to have same index as array_embedding_sparse
X_train_info.index = range(X_train_info.shape[0])


# In[7]:

# number of closest neighbors to collect recipients from:
nb_neigh = 70


# In[8]:

# Mark down the time
t_begin = datetime.now()

results = pd.DataFrame(columns=['recipients'])
results.index.name = 'mid'

all_mean_ap = []
all_ground_truth = []
all_suggestions = []
for query_id in X_test_info.index.values:

    mail = X_test_info['body'][query_id]
    mail_date = X_test_info['date'][query_id]
    query_mid = X_test_info['mid'][query_id]
    sender = X_test_info['sender'][query_id]
    mail_tfidf= tfidf.transform([mail])
    
    closest_ids, similarities = most_similar_sklearn(array_embedding_sparse, mail_tfidf)

    # find the closest emails (written or received by the sender) to the query one
    closest_ids_of_sender = get_n_closest_emails(sender, nb_neigh, closest_ids, X_train_info)
    
    # sort ids per date
    closest_emails_dates = pd.DataFrame(X_train_info['date'][closest_ids_of_sender].sort_values())
    closest_emails_dates['weight_date'] = range(1, len(closest_ids_of_sender)+1)

    # Create dictionnary of all recipient for the 30 most similar emails, and get frequency
    # For the moment it is only the brute frequency, maybe we could refine this by adding wheights according to the closseness of the email
    suggested_10_recipients = get_10_recipients(sender, closest_ids_of_sender, X_train_info, closest_emails_dates, similarities,mail[10:], mail_tfidf)

    if submission:
        string_recipients = ''
        for k in suggested_10_recipients:
            string_recipients+=k + ' '
        results.loc[query_mid, 'recipients'] = string_recipients
    else:
        ground_truth = X_test_info['recipients'][query_id].split()
        all_suggestions.append(suggested_10_recipients)
        ground_truth.append(ground_truth)
        all_mean_ap.append(mean_ap(suggested_10_recipients, ground_truth))
    
# Print the run time
print datetime.now()-t_begin


# In[9]:

# Uncomment to test your work when submission = False
# np.mean(all_mean_ap)


# In[10]:

#Save results
results.to_csv('../submission/submission.csv')


# In[ ]:



