from flask import Flask, render_template,request,jsonify
app=Flask(__name__)
app.debug=True
import nltk
import os,pandas as pd
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow
import tflearn
import random
import json      
import os
from glob import glob
data={}
json_dir_name = "json/"
json_pattern = os.path.join(json_dir_name,'*.json')
file_list = glob(json_pattern)
for file in file_list:
        with open(file) as f:
            d = json.load(f)
            data.update(d)
words = []
labels = []
docs_x = []
docs_y = []
value=list(data.values())
for val in range(len(value)):
        for val1 in range(len(value[val])):
                intent=dict(value[val][val1])
                for pattern in intent['patterns']:
                        wrds = nltk.word_tokenize(pattern)
                        words.extend(wrds)
                        docs_x.append(wrds)
                        docs_y.append(intent["tag"])
                        if intent['tag'] not in labels:
                          labels.append(intent['tag'])
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)
out_empty = [0 for _ in range(len(labels))]
training = []
output = []
for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w.lower()) for w in doc]
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)
training = numpy.array(training)
output = numpy.array(output)
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load("model.tflearn")

@app.route("/")
def home():
        return render_template("chat.html")
        



@app.route("/ask",methods=["POST","GET"])
def ask():
       inp=request.form['chatmessage']
       def bag_of_words(s, words):
            bag = [0 for _ in range(len(words))]
            s_words = nltk.word_tokenize(s)
            s_words = [stemmer.stem(word.lower()) for word in s_words]
            for se in s_words:
                for i, w in enumerate(words):
                    if w == se:
                        bag[i] = 1
            return numpy.array(bag)
       results = model.predict([bag_of_words(inp, words)])
       max_val=numpy.max(results)
       if(max_val<0.4):
               output="Sorry!!Cannot Understand your query.Please be clear with your query"
       else:
               results_index = numpy.argmax(results)
               tag = labels[results_index]
               for val in range(len(value)):
                    for val1 in range(len(value[val])):
                        tg=dict(value[val][val1])
                        if tg['tag'] == tag:
                            responses = tg['responses']                 
               output=random.choice(responses)
       return jsonify(output)


if __name__ == "__main__":
    
	app.run()
