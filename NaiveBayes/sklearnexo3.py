import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#Data collection
data = pd.read_csv(".\DataSets\TextClass.csv")
#Data initialization
X=data['text']
Y=data['label']

vec = CountVectorizer(vocabulary=['TV', 'programme', 'intéressant', 'enfants', 'radio', 'onde', 'écouter', 'rare'],lowercase=False)
X_ = vec.fit_transform(X)

X_ = X_.toarray()

mnb=MultinomialNB()
model=mnb.fit(X_,Y)

new_sentence = ['J’ai vu la radio de mes poumons à la TV']
vectorizer = CountVectorizer(vocabulary=['TV', 'programme', 'intéressant', 'enfants', 'radio', 'onde', 'écouter', 'rare'],lowercase=False)
X_test = vectorizer.fit_transform(new_sentence)

y_predicted=model.predict(X_test)

print(y_predicted)