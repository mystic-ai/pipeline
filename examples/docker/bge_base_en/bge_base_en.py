from FlagEmbedding import FlagModel

from pipeline import Pipeline, Variable, entity, pipe


@entity
class Flag:
    def __init__(self):
        self.model = None

    @pipe(run_once=True, on_startup=True)
    def load(self):
        print(
            "Loading bge-base-en-v1.5 model...",
            flush=True,
        )
        self.model = FlagModel(
            "BAAI/bge-base-en-v1.5",
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=True,
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    @pipe
    def compute_similarity(
        self, sentences_1: list[str], sentences_2: list[str]
    ) -> list[float]:
        embeddings_1 = self.model.encode(sentences_1)
        embeddings_2 = self.model.encode(sentences_2)
        similarity = embeddings_1 @ embeddings_2.T
        return similarity.tolist()


with Pipeline() as builder:
    sentences_1 = Variable(list)
    sentences_2 = Variable(list)

    model = Flag()
    model.load()

    similarity = model.compute_similarity(sentences_1, sentences_2)
    builder.output(similarity)

bge_base_en = builder.get_pipeline()
