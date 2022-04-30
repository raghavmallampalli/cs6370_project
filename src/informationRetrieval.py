from util import *

# Add your import statements here
import math
from collections import Counter
import pickle

import numpy as np

class InformationRetrieval():

    def __init__(self):
        self.index = None

    def countTokens(
        self,
        docs,
        docIDs=None
    ):
        if docIDs is None:
            docIDs = [i for i in range(len(docs))]
        index = {}
        for idx in range(len(docs)):
            docID = docIDs[idx]
            doc = docs[idx]
            # Converting to a list of words
            terms_list = [word for sentence in doc for word in sentence]
            for term, term_frequency in list(Counter(terms_list).items()):
                if term in index.keys():
                    index[term].append([docID, term_frequency])
                else:
                    index[term] = [[docID, term_frequency]]
        return docIDs, index
        
    def buildIndex(
        self,
        docs,
        docIDs
    ):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
            after pre-processing has been applied.
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        docIDs, index = self.countTokens(docs, docIDs)
        self.docIDs = docIDs
        self.index = index

    def generateTfIDF(self, index, docIDs):
        """
        Generates TF-IDF matrix and returns
        """
        
        # Convert to list to ensure order preservation
        list_index = sorted(list(index.items()))
        vocab_size = len(index)
        dict_ID = {}
        N = len(docIDs)
        # Create a set order of calling document IDs
        for i,ID in enumerate(docIDs):
            dict_ID[ID] = i
        
        # Order of tokens given by order of list_index
        idf_values = np.zeros(vocab_size)
        # Generate using calculated TF values and IDF values
        doc_tf_idf = np.zeros((len(docIDs),vocab_size))
        
        for i, token_counts in enumerate(list_index):
            # token_counts: (token, list of size two list)
            # each size two list is document ID, term frequency
            df_value = len(token_counts[1])
            idf_values[i] = np.log10(N/df_value)
        
        for i, token_counts in enumerate(list_index):
            for ID, tf in token_counts[1]:
                doc_tf_idf[dict_ID[ID],i] = tf

        # Broadcasting ensures multiplication is carried correctly
        doc_tf_idf = doc_tf_idf * idf_values

        return doc_tf_idf
    
    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        
        # Calculate IDF values
        doc_IDs_ordered = []
        index = self.index
        docIDs = self.docIDs
        doc_tf_idf = self.generateTfIDF(index, docIDs)

        # INCOMPLETE: Call the same function again for queries
        # IDs will be auto-generated and returned
        # Just get the TF-IDF
        # IMPORTANT: ensure dimension is same: ignore tokens in 
        # query that are not in docs?
        
        # INCOMPLETE: Write SVD function and return the LSA matrix
        
        # Call LSA on 
        # INCOMPLETE: Ensure ranking operation also uses broadcasting
        # Generate [num_queries, num_docs] matrix as result
        # np.argmax, np.max to return result in same form
        
        doc_IDs_ordered = []
        index = self.index
        docIDs = self.docIDs
        idf_values = {}
        zero_vector = {}
        # calculating the idf values for each term in vocabulary
        for term in index.keys():
            df_value = len(index[term])
            idf_values[term] = math.log10(float(N/df_value))
            zero_vector[term] = 0
        # Representing docs as Vectors with their tf-idf values
        # corresponding to each term
        doc_vectors = {}
        for docID in docIDs:
            doc_vectors[docID] = zero_vector.copy()

        
        # The terms which are absent in a doc are initialized to zero
        # as in the above for loop
        for term in index:
            for docID, tf in index[term]:
                doc_vectors[docID][term] = tf * idf_values[term]
        self.dict_tfidf = doc_vectors

        for i, query in enumerate(queries):
            print(f'Ranking for query {i+1}/{len(queries)}', end='\r')
            query_terms = [term for sentence in query for term in sentence]
            query_vector = zero_vector.copy()
            for term, tf in list(Counter(query_terms).items()):
                if term in query_vector.keys():
                    query_vector[term] = tf * idf_values[term]

            similarities = {}
            for docID in docIDs:

                try:
                    similarities[docID] = sum(doc_vectors[docID][term] * query_vector[term] for term in zero_vector) / (math.sqrt(sum(
                        doc_vectors[docID][term] ** 2 for term in zero_vector)) * math.sqrt(sum(query_vector[term] ** 2 for term in zero_vector)))
                except:
                    similarities[docID] = 0
            doc_IDs_ordered.append([docID for docID, similarity in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True)])
        print("Ranking complete.")

        # Save results into a pickle file for convenience
        with open('rank_result.pkl', 'wb') as f:
            pickle.dump(doc_IDs_ordered, f)

        return doc_IDs_ordered
