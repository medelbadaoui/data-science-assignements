import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer










dataX =[['Le programme TV n’est pas intéressant'],['La TV m’ennuie'],['Les enfants aiment la TV'],['On reçoit la TV par onde radio'],
        ['Il est intéressant d’écouter la radio'],['Sur les ondes, les programmes pour enfants sont rares'],['Les enfants vont écouter la radio; c’est rare']]
dataY=[['TV'],['TV'],['TV'],['TV'],['RADIO'],['RADIO'],['RADIO']]
dfx = pd.DataFrame(dataX,columns=['text'])
dfy= pd.DataFrame(dataY,columns=['label'])
data=dfx.join(dfy)
features=['text']
X=data[features]
y=data['label']

X_test=[['J’ai vu la radio de mes poumons à la TV']]

radio_docs=[row['text'] for index,row in data.iterrows() if row['label'] == 'RADIO']

# print(radio_docs)

vec_radio = CountVectorizer()
X_radio = vec_radio.fit_transform(radio_docs)
tdm_radio = pd.DataFrame(X_radio.toarray(), columns=vec_radio.get_feature_names())

tv_docs = [row['text'] for index,row in data.iterrows() if row['label'] == 'TV']

vec_tv = CountVectorizer()
X_tv = vec_tv.fit_transform(tv_docs)
tdm_tv = pd.DataFrame(X_tv.toarray(), columns=vec_tv.get_feature_names())

word_list_radio = vec_radio.get_feature_names();    
count_list_radio = X_radio.toarray().sum(axis=0) 
freq_radio = dict(zip(word_list_radio,count_list_radio))

# print(freq_radio)


word_list_tv = vec_tv.get_feature_names();    
count_list_tv = X_tv.toarray().sum(axis=0) 
freq_tv = dict(zip(word_list_tv,count_list_tv))

prob_tv=[]
for word,count in zip(word_list_tv ,count_list_tv):
        prob_tv.append(count/len(word_list_tv))
#print(dict(zip(word_list_tv,prob_tv)))

prob_radio=[]
for word,count in zip(word_list_radio ,count_list_radio):
        prob_radio.append(count/len(word_list_radio))
#print(dict(zip(word_list_radio,prob_radio)))



docs = [row['text'] for index,row in data.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())

total_features_tv = count_list_tv.sum(axis=0)
total_features_radio = count_list_radio.sum(axis=0)


#print(total_features,total_features_tv,total_features_radio)


new_sentence = ['J’ai vu la radio de mes poumons à la TV']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(new_sentence)
new_word_list = vectorizer.get_feature_names()



prob_radio_with_ls = []
for word in new_word_list:
    if word in freq_radio.keys():
        count = freq_radio[word]
    else:
        count = 0
    prob_radio_with_ls.append((count + 1)/(total_features_radio + total_features))
print(dict(zip(new_word_list,prob_radio_with_ls)))

p_radio=1
for i in prob_radio_with_ls:
        p_radio*=i

print(p_radio)    


prob_tv_with_ls = []
for word in new_word_list:
    if word in freq_tv.keys():
        count = freq_tv[word]
    else:
        count = 0
    prob_tv_with_ls.append((count + 1)/(total_features_tv + total_features))
print(dict(zip(new_word_list,prob_tv_with_ls)))

p_tv=1
for i in prob_tv_with_ls:
        p_tv*=i

print(p_tv)

print(max([p_tv,p_radio]))



# vec_s = CountVectorizer()
# X_s = vec_s.fit_transform(stmt_docs)
# tdm_s = pd.DataFrame(X_s.toarray(), columns=vec_s.get_feature_names())


# gnb = MultinomialNB()
# y_pred = gnb.fit(X, y).predict(X_test)
# print(y_pred)
