from sentence_transformers import SentenceTransformer, util
import json
import torch
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


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

print("Initializing YottaByte ---->")
passage_embedding = model.encode(list)

while True:
    print("ChatBot >> Enter a Question:")
    userInput = str(input())
    if userInput == "bye":
        break
    query_embedding = model.encode(userInput)
    scores = util.dot_score(query_embedding, passage_embedding)
    print(scores)
    current_sent_idx = torch.argsort(scores)[0][-1]
    print(current_sent_idx)
    scores = torch.flatten(scores)
    print(scores)
    values = torch.sort(scores)
    if values[0][-1] > 0:
        #Find the Most Similar answer
        ans_emb = model.encode(answer_super_list[int(current_sent_idx)])
        scores_ans = util.dot_score(query_embedding, ans_emb)
        current_sent_idx_ans = scores_ans.argsort()[0][-1]
        print("ChatBot >>  " + answer_super_list[current_sent_idx][current_sent_idx_ans])
    else:
        print("ChatBot >>  I am not sure. Sorry!" )