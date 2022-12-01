import json
import nltk
from string import punctuation
# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# cosine similarity score
from sklearn.metrics.pairwise import cosine_similarity

print("Initializing YottaByte ---->")
stopwords = nltk.corpus.stopwords.words('english')
punctuation_dict = str.maketrans({p:None for p in punctuation})
lemmatizer = nltk.stem.WordNetLemmatizer()
# preprocess method - to be called internally by Tf-Idf vectorizer
# text preprocessing, stopword removal, lemmatization, word tokenization
def preprocess(text):
        # remove punctuations
        text = text.lower().strip().translate(punctuation_dict) 
        # tokenize into words
        words = nltk.word_tokenize(text)
        # remove stopwords
        words = [w for w in words if w not in stopwords]
        # lemmatize 
        return [lemmatizer.lemmatize(w) for w in words]

f = open('result_politics.json')
list = []
json_obj = json.load(f)
answer_super_list = []
for item in json_obj:
    answer_list = []
    question = item['question']
    for ans in item['answers']:
        answer = ans['body']
        answer_list.append(answer)
    list.append(question)
    answer_super_list.append(answer_list)
    
print("To End Chat Please Enter \"bye\"")

while True:
    print("ChatBot >> Enter a Question:")
    userInput = str(input())
    if userInput == "bye":
        break
    text = userInput
    list.append(userInput)
    vectorizer = TfidfVectorizer(tokenizer=preprocess)
    # fit data and obtain tf-idf vector
    tfidf = vectorizer.fit_transform(list)
    # calculate cosine similarity scores b/w query and questions list
    scores = cosine_similarity(tfidf[-1],tfidf)
    # identify the most closest question NOTE: This part can be taken care by Okapi BM-25, i.e. USE SOLR!
    current_sent_idx = scores.argsort()[0][-2]
    #print(list[current_sent_idx])
    #find the corresponding score value
    scores = scores.flatten()
    scores.sort()
    value = scores[-2]
    #print(value)
    # if there is matching question
    if value != 0:
        #Find the Most Similar answer
        vectorizer_ans = TfidfVectorizer(tokenizer=preprocess)
        answer_super_list[current_sent_idx].append(userInput)
        #print(len(answer_super_list[current_sent_idx]))
        tfidf_ans = vectorizer.fit_transform(answer_super_list[current_sent_idx])
        scores_ans = cosine_similarity(tfidf_ans[-1],tfidf_ans)
        current_sent_idx_ans = scores_ans.argsort()[0][-2]
        print("ChatBot >>  " + answer_super_list[current_sent_idx][current_sent_idx_ans])
        answer_super_list[current_sent_idx].pop()
        # if no sentence is matching the query
    else:
        print("ChatBot >>  I am not sure. Sorry!" )
    list.pop()

