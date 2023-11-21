# https://www.kaggle.com/code/mysarahmadbhat/getting-started-with-chat-bot-using-keras/notebook
# https://towardsdatascience.com/deep-learning-for-nlp-creating-a-chatbot-with-keras-da5ca051e051


import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

#Define intents file
with open('./keras_chatbot/chat-intents.json') as file:
    data = json.load(file)

# Data Prep.

training_sentences = []
training_labels = []
labels = []
responses = []
for intent in data['intents']:
    for patterns in intent['patterns']:
        training_sentences.append(patterns)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

encoder= LabelEncoder()
encoder.fit(training_labels)
training_labels = encoder.transform(training_labels)

# vectorize the text data using the “Tokenizer”
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

#  make the text sequences the same size.
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Model Training genera 3 nodos (2 relu de 16 y el ultimo del tamanio de las etiquetas)
num_classes = len(labels)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()

# Train de Model
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

while True:
    inp = input("You: ")
    if inp.lower()  in ['quit', 'q', 'exit']:
        break
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    tag = encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            print("ChatBot:"  , np.random.choice(i['responses']))
            print("User Intent:"  ,i['tag'])
