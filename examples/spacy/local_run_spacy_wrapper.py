from pipeline import spacy_to_pipeline


# using func arg in spacy_to_pipeline
def func(doc):
    return [[token.text, token.lemma_, token.pos_] for token in doc]

spacy_pipeline = spacy_to_pipeline("en_core_web_sm", func=func, name="spacy get all")


input = "hello there"

# run locally
[output] = spacy_pipeline.run(input)

print(output)
