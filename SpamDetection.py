%cd drive/My\ Drive

# Commented out IPython magic to ensure Python compatibility.
# %cd ../..

!cp drive/My\ Drive/sms-spam-collection-dataset/spam.csv /content

import pandas
mydata = pandas.read_csv("spam.csv",encoding="latin-1")
mydata.head()

mydata = mydata.drop(labels=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
mydata = mydata.rename(columns={"v1":"label","v2":"inp"})
mydata.describe()
mydata["length"]= mydata["inp"].apply(len)
mydata.head()

import matplotlib as mat
mat.rcParams["patch.force_edgecolor"] = True
mat.pyplot.style.use("seaborn-bright")
mydata.hist(column="length",by="label",bins=50,figsize=(11,5))

mydata["label"].value_counts()/mydata.shape[0]

import string
def text_process(text):
  text = text.translate(str.maketrans("","",string.punctuation))
  text = [word.lower() for word in text.split()]
  return " ".join(text)

inpcol = mydata["inp"].copy()
preprocessed = inpcol.apply(text_process)
preprocessed[0]

maxlen = 0
for document in preprocessed:
  document_len = len(document.split())
  if document_len > maxlen:
    maxlen = document_len
print(maxlen)

from keras.preprocessing.text import one_hot
vocab_size = 20000
encoded_docs = [one_hot(document,vocab_size) for document in preprocessed]
print(encoded_docs[0])

from keras.preprocessing.sequence import pad_sequences

padded_docs = pad_sequences(encoded_docs,maxlen=maxlen,padding="post")
print(padded_docs.shape)

mydata2 = []
for i in range(len(mydata)):
  myex = [None,None]
  myex[0] = padded_docs[i]
  if mydata.loc[i][0] == "ham":
    myex[1] = 0
  elif mydata.loc[i][0] == "spam":
    myex[1] = 1
  mydata2.append(myex)

mydata2[1]

from sklearn.model_selection import train_test_split
import numpy as np
xtrain, xtest, ytrain, ytest = train_test_split(padded_docs,np.array(mydata2)[:,1],test_size=0.3,random_state=42)
print(xtrain.shape,ytrain.shape)

from keras.layers import LSTM, BatchNormalization, Dropout, Input, Dense, Embedding, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(vocab_size,256,input_length=maxlen))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1,activation="sigmoid"))
opt = Adam(lr=0.001)
model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ["acc"])

model.fit(xtrain,ytrain,epochs=10, batch_size=128,validation_split=0.3)

model.evaluate(xtest,ytest)

