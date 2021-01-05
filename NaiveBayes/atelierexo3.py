
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#data initialization
#Data collection
data = pd.read_csv(".\DataSets\TextClass.csv")
#Data initialization
X=data['text']
Y=data['label']
#seperate radio and tv records 

radio_docs=[row['text'] for index,row in data.iterrows() if row['label'] == 'radio']
tv_docs = [row['text'] for index,row in data.iterrows() if row['label'] == 'TV']

#seperate features for each word

vec_radio = CountVectorizer(vocabulary=['TV', 'programme', 'intéressant', 'enfants', 'radio', 'onde', 'écouter', 'rare'],lowercase=False)
X_radio = vec_radio.fit_transform(radio_docs)
tdm_radio = pd.DataFrame(X_radio.toarray(), columns=vec_radio.get_feature_names())

vec_tv = CountVectorizer(vocabulary=['TV', 'programme', 'intéressant', 'enfants', 'radio', 'onde', 'écouter', 'rare'],lowercase=False)
X_tv = vec_tv.fit_transform(tv_docs)
tdm_tv = pd.DataFrame(X_tv.toarray(), columns=vec_tv.get_feature_names())
#convert dataframe to dict and calculate frequence of each word in radio and tv

word_list_radio = vec_radio.get_feature_names();    
count_list_radio = X_radio.toarray().sum(axis=0) 
freq_radio = dict(zip(word_list_radio,count_list_radio))

word_list_tv = vec_tv.get_feature_names();    
count_list_tv = X_tv.toarray().sum(axis=0) 
freq_tv = dict(zip(word_list_tv,count_list_tv))

print(freq_radio)
print(freq_tv)


docs = [row['text'] for index,row in data.iterrows()]

vec = CountVectorizer(vocabulary=['TV', 'programme', 'intéressant', 'enfants', 'radio', 'onde', 'écouter', 'rare'],lowercase=False)
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())
total_features_tv = count_list_tv.sum(axis=0)
total_features_radio = count_list_radio.sum(axis=0)

new_sentence = ['J’ai vu la radio de mes poumons à la TV']
vectorizer = CountVectorizer(vocabulary=['TV', 'programme', 'intéressant', 'enfants', 'radio', 'onde', 'écouter', 'rare'],lowercase=False)
X = vectorizer.fit_transform(new_sentence)
new_word_list = vectorizer.get_feature_names()

def calculateprobradio(new_word_list):
    prob_radio_with_ls = []
    p_radio=1
    for word in new_word_list:
        if word in freq_radio.keys():
            count = freq_radio[word]
        else:
            count = 0
        prob_radio_with_ls.append((count + 1)/(total_features_radio + total_features))
        p_radio*=(count + 1)/(total_features_radio + total_features)
    print(dict(zip(new_word_list,prob_radio_with_ls)))
    proba_rad=len(radio_docs)/len(docs)
    return p_radio*proba_rad,'RADIO' 

def calculateprobtv(new_word_list):
    prob_tv_with_ls = []
    p_tv=1
    for word in new_word_list:
        if word in freq_tv.keys():
            count = freq_tv[word]
        else:
            count = 0
        prob_tv_with_ls.append((count + 1)/(total_features_tv + total_features))
        p_tv*=(count + 1)/(total_features_tv + total_features)
    print(dict(zip(new_word_list,prob_tv_with_ls)))
    proba_tv=len(tv_docs)/len(docs)
    return p_tv*proba_tv,'TV' 

def predict(input):
    prob=[calculateprobradio(input),calculateprobtv(input)]
    return max(prob)

print(predict(new_word_list))
    