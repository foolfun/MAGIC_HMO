import numpy as np

class HyDE:

    def __init__(self, promptor, generator):
        self.promptor = promptor
        self.generator = generator
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.get_answer(prompt)
        return hypothesis_documents

    def searchByEmb(self, search_query, vector_db, top_k=10):
        results = vector_db.similarity_search_with_score(search_query, k=top_k)
        inds = [result[0].metadata['document'] for result in results]
        return inds
