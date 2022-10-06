x = open('positive.txt','r')
pos = x.readlines()
y = open('negative.txt','r')
neg = y.readlines()
lp = len(pos)
ln = len(neg)
X_train = []
for i in range (lp):
    X_train.append(pos[i])
for i in range (ln):
    X_train.append(neg[i])
y_train = []
for i in range (lp):
    y_train.append(1)
    #1 is for positive
for i in range (ln):
    y_train.append(0)
    #0 is for negative

##data cleaning
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
sno = SnowballStemmer('english')
def getcleanedtext(text):
    text = text.lower()

    #tokenization and stopword removal
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]

    stemmed_tokens = [sno.stem(token) for token in new_tokens]

    clean_text = " ".join(stemmed_tokens)

    return clean_text
#test split
X_test=["i love to hang out with you","I don't like you","I love you","I hated the way you laughed at me","It was extremely bad","the performance was bad but the movie was good"]

x_clean = [getcleanedtext(i) for i in X_train]
xt_clean = [getcleanedtext(i) for i in X_test]

#vectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))
x_vec = cv.fit_transform(x_clean).toarray()
xt_vec = cv.transform(xt_clean).toarray()

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()
mn.fit(x_vec,y_train)
y_pred = mn.predict(xt_vec)
print(y_pred)