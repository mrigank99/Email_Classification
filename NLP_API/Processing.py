import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

stops = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

sensitive_words = ["cumbersome","inconvenience","grievance","hassle","complain","complaint","dissatisfy",
                        "discontent","displease","displeasure","frustrate","irritate",
                        "displeasure","unhappy"]
query_words = ["what","when","why","which","where","who","how","whose","whom"]

stops = stops - {"not"}


def preprocess(text):
    # Performing Tokenisation
    complaint = text
    complaint = complaint.lower()
    token = nltk.word_tokenize(complaint)
    token_words = [item for item in token if item.isalpha()]

     # Performing Stop words removal
    important_words = [item for item in token_words if not item in stops]

    # Performing Lemmatization
    lemmatized_sentence = []
    for word, tag in pos_tag(important_words):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))

    # Removing xxxx
    word = "xxxx"
    temp = [item for item in lemmatized_sentence if item != word]

    # Rejoining Statement
    rejoin = (" ".join(temp))

    return rejoin

