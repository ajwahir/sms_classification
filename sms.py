import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from subprocess import check_output
pal = sns.color_palette()

df = pd.read_csv('TRAIN_SMS.csv', encoding='latin-1')
df = df.rename(columns={"Label":"label", "Message":"sms_text"})

messages = pd.Series(df['sms_text'].tolist())
ham_messages = pd.Series(df[df['label'] == 'ham']['sms_text'].tolist())
spam_messages = pd.Series(df[df['label'] == 'spam']['sms_text'].tolist())
info_messages = pd.Series(df[df['label'] == 'info']['sms_text'].tolist())

dist_all = messages.apply(len)
dist_ham = ham_messages.apply(len)
dist_spam = spam_messages.apply(len)
dist_info = info_messages.apply(len)


# # Plot distribution of character count of all messages
#
# plt.figure(figsize=(12, 8))
# plt.hist(dist_all, bins=100, range=[0,400], color=pal[3], normed=True, label='All')
# plt.title('Normalised histogram of character count in all messages', fontsize=15)
# plt.legend()
# plt.xlabel('Number of characters', fontsize=15)
# plt.ylabel('Probability', fontsize=15)
#
# print('# Summary statistics for character count of all messages')
# print('mean-all {:.2f} \nstd-all {:.2f} \nmin-all {:.2f} \nmax-all {:.2f}'.format(dist_all.mean(),
#                           dist_all.std(), dist_all.min(), dist_all.max()))
#
# # Plot distributions of character counts for spam vs ham messages
#
# plt.figure(figsize=(12,8))
# plt.hist(dist_ham, bins=100, range=[0,250], color=pal[1], normed=True, label='ham')
# plt.hist(dist_spam, bins=100, range=[0, 250], color=pal[2], normed=True, alpha=0.5, label='spam')
# plt.hist(dist_info, bins=100, range=[0, 250], color=pal[3], normed=True, alpha=0.5, label='info')
#
# plt.title('Normalised histogram of character count in messages', fontsize=15)
# plt.legend()
# plt.xlabel('Number of characters', fontsize=15)
# plt.ylabel('Probability', fontsize=15)
#
# print('# Summary statistics for character count of ham vs spam messages')
# print('mean-ham  {:.2f}   mean-spam {:.2f} \nstd-ham   {:.2f}   std-spam   {:.2f} \nmin-ham    {:.2f}   min-ham    {:.2f} \nmax-ham  {:.2f}   max-spam  {:.2f}'.format(dist_ham.mean(),
#                          dist_spam.mean(), dist_ham.std(), dist_spam.std(), dist_ham.min(), dist_spam.min(), dist_ham.max(), dist_spam.max()))


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

# Set our parameters for xgboost
params = {}
params['objective'] = 'multi:softmax'
params['eval_metric'] = 'merror'
params['eta'] = 0.02
params['max_depth'] = 4
params['num_class'] = 3
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)

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
