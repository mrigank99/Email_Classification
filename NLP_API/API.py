from flask import Flask, session, render_template, request, redirect, url_for
import requests
import pickle
import numpy as np
from Processing import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import sentiment as senti

model = pickle.load(open("BestModel.pkl","rb"))
vect = pickle.load(open('Bestvectorizer.pickle','rb'))
app = Flask(__name__)


tfidf = TfidfVectorizer(sublinear_tf= True,
                       min_df = 5,
                       norm= 'l2',
                       ngram_range= (1,2))

@app.route("/")
def index():
    return render_template("Homepage.html")

@app.route("/analyze", methods = ["POST"])
def analyze():
    user_input = request.form.get('content')
    text = preprocess(user_input)
    product = model.predict(vect.transform([text]))
    C_Q = senti.cq_classification(user_input) # Getting th classification category and the confidence level
    pred_probs = model.predict_proba(vect.transform([text]))
    sorted_proba_index = np.argsort(pred_probs)[0][::-1]
    classification_classes = model.classes_  # Returns a list of classes in the classifier

    list_of_pred_probs_dict = [{'name': classification_classes[item], 'prob': round(100 * pred_probs[0][item], 2)}
                               for item in sorted_proba_index]

    return render_template("results.html", usertext = user_input, product=product, prediction = list_of_pred_probs_dict, cat= C_Q,)

if __name__ == "__main__":
    app.debug = True
    app.run()