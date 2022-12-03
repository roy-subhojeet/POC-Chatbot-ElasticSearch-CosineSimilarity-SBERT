from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch, helpers
import json
import time
import tqdm.autonotebook
import traceback

elastic_host = {
    "host": "localhost", 
    "port": 9200,
    "scheme": "https"
    }


es = Elasticsearch(hosts=[elastic_host], basic_auth=["elastic", "P4XhCcym*tmit6OdIPsM"])

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
max_corpus_size = 1000

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

#Index data, if the index does not exists
if not es.indices.exists(index="reddit_17"):
    try:
        es_index = {
            "mappings": {
              "properties": {
                "question": {
                  "type": "text"
                },
                "question_vector": {
                  "type": "dense_vector",
                  "dims": 384
                }
              }
            }
        }

        es.indices.create(index='reddit_17', body=es_index, ignore=[400])
        chunk_size = 500
        print("Index data (you can stop it by pressing Ctrl+C once):")
        i=0
        with tqdm.tqdm(total=len(list)) as pbar:
            for start_idx in range(0, len(list), chunk_size):
                end_idx = start_idx+chunk_size

                embeddings = model.encode(list[start_idx:end_idx], show_progress_bar=True)
                bulk_data = []
                #print(embeddings)
                for question, embedding in zip(list[start_idx:end_idx], embeddings):
                    i+=1
                    bulk_data.append({
                            "_id": i,
                            "_index": 'reddit_17',
                            "_source": {
                                "question": question,
                                "question_vector": embedding
                            }
                        })
                print(i)
                helpers.bulk(es, bulk_data)
                pbar.update(chunk_size)

    except:
        print("During index an exception occured. Continue\n\n")
        traceback.print_exc()

#Interactive search queries
while True:
    inp_question = input("Please enter a question: ")

    encode_start_time = time.time()
    question_embedding = model.encode(inp_question)
    encode_end_time = time.time()
    print(question_embedding)

    #Lexical search
    bm25 = es.search(index="reddit_17", body={"query": {"match": {"question": inp_question }}})

    #Sematic search
    sem_search = es.search(index="reddit_17", query={
            "script_score": {
              "query": {
                "match_all": {}
              },
              "script": {
                "source": "cosineSimilarity(params.queryVector, 'question_vector') + 1.0",
                "params": {
                  "queryVector": question_embedding
                }
              }
            }
        })

    print("Input question:", inp_question)
    print("Computing the embedding took {:.3f} seconds, BM25 search took {:.3f} seconds, semantic search with ES took {:.3f} seconds".format(encode_end_time-encode_start_time, bm25['took']/1000, sem_search['took']/1000))

    print("BM25 results:")
    for hit in bm25['hits']['hits']:
        print("\t{}".format(hit['_source']['question']))

    print("\nSemantic Search results:")
    for hit in sem_search['hits']['hits']:
        print("\t{}".format(hit['_source']['question']))

    print("\n\n========\n")