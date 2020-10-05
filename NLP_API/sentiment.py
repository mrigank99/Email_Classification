import nltk
from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
classifier = TextClassifier.load('en-sentiment')

sensitive_words = ["cumbersome", "inconvenience", "grievance", "hassle", "complain", "complaint", "dissatisfy",
                       "discontent", "displease", "displeasure", "frustrate", "irritate",
                       "displeasure", "unhappy"]
query_words = ["what", "when", "why", "which", "where", "who", "how", "whose", "whom"]

negative_words = []
with open('negative.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

for line in filecontents:
    # remove linebreak which is the last character of the string
    current_place = line[:-1]

    # add item to the list
    negative_words.append(current_place)

negative_words = set(negative_words)-set(sensitive_words)

def cq_classification(text):
    complaint = text
    complaint = complaint.lower()
    token = nltk.word_tokenize(complaint)
    token_words = [item for item in token if item.isalpha()]

    neg_count = 0
    sen_count = 0
    que_count = 0
    score = 0 # Sentiment score
    HS = 0  # Highly sensitive words score
    QS = 0  # Query words score
    NS = 0  # Negative words score

    for word in token_words:
        if word in query_words:
            que_count = que_count+1

    for word in token_words:
        if word in negative_words:
            neg_count = neg_count+1

    for word in token_words:
        if word in sensitive_words:
            sen_count = sen_count+1

    sentence = Sentence(text)
    classifier.predict(sentence)

    if sen_count >= 2:  # Calculating Highly sensitive words score
        HS = -1
    elif sen_count == 1:
        HS = -0.5
    else:
        HS = 0

    if que_count >= 1:  # Calculating Query words score
        QS = 1

    NS = -neg_count * 0.1  # Calculating Negative words score

    total = HS + NS
    if abs(total) > 1: # Normalizing the total score
        total = -1

    if sentence.labels[0].value == 'NEGATIVE': # Determining the sentiment score
        score = -sentence.labels[0].score
    else:
        score = sentence.labels[0].score

    final = 0.3 * score + 0.7 * total   # Taking 30% from sentiment and 70% from custom dictionary score

    if final <= -0.3: # Using final score with THRESHHOLD = 0.3 for classification
        cat = "Complaint"
    else:
        if QS >= 1:
            cat = "Query"
        else:
            cat = "Doubt"

    if cat == "Doubt":
        if score < 0:
            cat = "Complaint"
        else:
            cat = "Query"
    # tempList = [neg_count,sen_count,que_count,cat]
    confidence = abs(final)
    return cat
