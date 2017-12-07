import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import defaultdict
import operator

df = pd.read_csv('../TRAIN_SMS.csv', encoding='latin-1')
df = df.rename(columns={"Label":"label", "Message":"sms_text"})
messages = pd.Series(df['sms_text'].tolist())
ham_messages = pd.Series(df[df['label'] == 'ham']['sms_text'].tolist())
spam_messages = pd.Series(df[df['label'] == 'spam']['sms_text'].tolist())
info_messages = pd.Series(df[df['label'] == 'info']['sms_text'].tolist())

dist_all = messages.apply(len)
dist_ham = ham_messages.apply(len)
dist_spam = spam_messages.apply(len)
dist_info = info_messages.apply(len)

df['word_count'] = pd.Series(df['sms_text'].tolist()).apply(lambda x: len(x.split(' ')))
df['char_count'] = pd.Series(df['sms_text'].tolist()).apply(len)
df['label'][df['label']=='ham']=0
df['label'][df['label']=='info']=1
df['label'][df['label']=='spam']=2
X = df[['word_count', 'char_count']]
y = df[['label']]
y=pd.to_numeric(y['label'],errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

params = {}
params['objective'] = 'multi:softmax'
params['eval_metric'] = 'merror'
params['eta'] = 0.15
params['max_depth'] = 4
params['num_class'] = 3
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)

# param check

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
# Predict values for test set
d_test = xgb.DMatrix(X_test)
p_test = bst.predict(d_test)
# Apply function round() to each element in np array
# so predictions are all either 0 or 1.
npround = np.vectorize(round)
p_test_ints = npround(p_test)
# Error rate for test set
accuracy = accuracy_score(y_test, p_test_ints)
print("Test Accuracy: ", accuracy)

X = df
y = df[['label']]
y=pd.to_numeric(y['label'],errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ham_messages_train = pd.Series(X_train[X_train['label'] == 0]['sms_text'].tolist()).astype(str)
spam_messages_train = pd.Series(X_train[X_train['label'] == 2]['sms_text'].tolist()).astype(str)
info_messages_train = pd.Series(X_train[X_train['label'] == 1]['sms_text'].tolist()).astype(str)

# WordCloud automatically excludes stop words
spam_messages_one_string = " ".join(spam_messages_train.astype(str))
ham_messages_one_string = " ".join(ham_messages_train.astype(str))
info_messages_one_string = " ".join(info_messages_train.astype(str))

ham_words_list = ham_messages_one_string.split()
total_ham_words = len(ham_words_list)
print("Total number of words in ham messages: ", total_ham_words)
ham_words_dict = Counter(ham_words_list).most_common()
ham_words_dict[:25]

spam_words_list = spam_messages_one_string.split()
total_spam_words = len(spam_words_list)
print("Total number of words in spam messages: ", total_spam_words)
spam_words_dict = Counter(spam_words_list).most_common()
spam_words_dict[:25]

info_words_list = info_messages_one_string.split()
total_info_words = len(info_words_list)
print("Total number of words in info messages: ", total_info_words)
info_words_dict = Counter(info_words_list).most_common()
info_words_dict[:25]
stopwords = set(stopwords.words("english"))

ham_words_lowercase = ham_messages_one_string.lower().split()
ham_words_nostop = []
for word in ham_words_lowercase:
    if word not in stopwords:
        ham_words_nostop.append(word)
ham_words_freq = Counter(ham_words_nostop).most_common()
ham_words_top25=ham_words_freq[:25]

spam_words_lowercase = spam_messages_one_string.lower().split()
spam_words_nostop = []
for word in spam_words_lowercase:
    if word not in stopwords:
        spam_words_nostop.append(word)
spam_words_freq = Counter(spam_words_nostop).most_common()
spam_words_top25=spam_words_freq[:25]


info_words_lowercase = info_messages_one_string.lower().split()
info_words_nostop = []
for word in info_words_lowercase:
    if word not in stopwords:
        info_words_nostop.append(word)
info_words_freq = Counter(info_words_nostop).most_common()
info_words_top25=info_words_freq[:25]

spam_words_top25_list = [tuple[0] for tuple in spam_words_top25]
ham_words_top25_list = [tuple[0] for tuple in ham_words_top25]
info_words_top25_list = [tuple[0] for tuple in info_words_top25]

uniquemap={}
uniquewords=[]
for word in spam_words_top25_list :
    if word not in uniquemap :
        uniquemap[word]=1
for word in ham_words_top25_list :
    if word not in uniquemap :
        uniquemap[word]=1
for word in info_words_top25_list :
    if word not in uniquemap :
        uniquemap[word]=1
for word in ["due in","flt","booking id","repair ref","reservation"] :
    if word not in uniquemap :
        uniquemap[word]=1
dff = df['sms_text'].str.lower()
Xx_train = X_train['sms_text'].str.lower()
Xx_test  = X_test['sms_text'].str.lower()
for word in uniquemap.keys() :
    df[word] =  list(int(y==True) for y in list(word in x for x in dff))
    X_train[word] =  list(int(y==True) for y in list(word in x for x in Xx_train))
    X_test[word] =  list(int(y==True) for y in list(word in x for x in Xx_test))

del X_train['sms_text']
del X_train['label']
del X_test['sms_text']
del X_test['label']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = X_train[list(X_train.columns)]
X_test = X_test[list(X_test.columns)]
X_valid = X_valid[list(X_train.columns)]
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(X_test)
p_test = bst.predict(d_test)
npround = np.vectorize(round)
p_test_ints = npround(p_test)
accuracy = accuracy_score(y_test, p_test_ints)
print("Test Accuracy: ", accuracy)
#### Submission file ####
df = pd.read_csv('../DEV_SMS.csv', encoding='latin-1')
df = df.rename(columns={"RecordNo":"RecordNo", "Message":"sms_text"})
RecordNo =np.array(df["RecordNo"]).astype(int)
df['word_count'] = pd.Series(df['sms_text'].tolist()).apply(lambda x: len(x.split(' ')))
df['char_count'] = pd.Series(df['sms_text'].tolist()).apply(len)
X = df
Xx  = X['sms_text'].str.lower()
for word in uniquemap.keys() :
    X[word] =  list(int(y==True) for y in list(word in x for x in Xx))
del X['sms_text']
del X['RecordNo']
X = X[list(X.columns)]
d_test = xgb.DMatrix(X)
p_test = bst.predict(d_test)
npround = np.vectorize(round)
p_test_ints = npround(p_test)
s=[]
for i in p_test_ints :
    if i==0 :
        s.append('ham')
    elif i==1 :
        s.append('info')
    else :
        s.append('spam')

my_solution = pd.DataFrame(s, RecordNo, columns = ["Label"])
my_solution.to_csv("../sub.csv", index_label = ["RecordNo"])  # contains ham, spam, info for given DEV_SMS.csv
