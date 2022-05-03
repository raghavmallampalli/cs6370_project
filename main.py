import json
import pandas as pd
from LSA import LSA

lsa = LSA()

docs_df = pd.read_json('data/cranfield/cran_docs.json')
docs_df.at[470, 'body'] = "<UNK>"
docs_df.at[994, 'body'] = "<UNK>"
queries_df = pd.read_json('data/cranfield/cran_queries.json')
qrels_df = pd.read_json('data/cranfield/cran_qrels.json')
qrels = json.load(open("data/cranfield/cran_qrels.json", 'r'))[:]

tf_idf = lsa.get_tfidf_matrices(docs_df['body'], queries_df['query'],)
docs = tf_idf['documents']
queries = tf_idf['queries']

docs_final, query_final = lsa.perform_svd(docs, queries, 500)

ranked = lsa.rank_docs(docs_final, query_final)

precisions, recalls, fscores, MAPs, nDCGs = lsa.calculate_metrics(ranked, qrels)

lsa.plot_metrics(precisions, recalls, fscores, MAPs, nDCGs)