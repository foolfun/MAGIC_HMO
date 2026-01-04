WEB_SEARCH = """Please write a passage to answer the question. Give the rationale before answering
Question: {}
Passage:"""


Poem = """请找出一首中国古代诗词能够与以下用户需求相关联，请写出该古诗的诗意。回答前请先给出理由。
用户需求: {}
结果:"""


class PromptorQ2Kw:
    def __init__(self, task: str, language: str = 'en'):
        self.task = task
        self.language = language

    def build_prompt(self, query: str):
        if self.task == 'web search':
            return WEB_SEARCH.format(query)
        elif self.task == 'poem':
            return Poem.format(query)
        else:
            raise ValueError('Task not supported')
