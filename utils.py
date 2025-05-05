from datasets import load_dataset, Dataset
import pandas as pd

def load_data() -> tuple[Dataset]:
    ds = load_dataset("Babelscape/cner")
    return ds["train"], ds["validation"], ds["test"]
def load_data_as_df() -> tuple[pd.DataFrame]:
    ds = load_dataset("Babelscape/cner")
    return ds["train"].to_pandas(), ds["validation"].to_pandas(), ds["test"].to_pandas()

def get_labels() -> list[str]:
    labels_vocab = {
        "O": 0,
        "B-ANIMAL": 1,
        "I-ANIMAL": 2,
        "B-DISEASE": 3,
        "I-DISEASE": 4,
        "B-DISCIPLINE": 5,
        "I-DISCIPLINE": 6,
        "B-LANGUAGE": 7,
        "I-LANGUAGE": 8,
        "B-EVENT": 9,
        "I-EVENT": 10,
        "B-FOOD": 11,
        "I-FOOD": 12,
        "B-ARTIFACT": 13,
        "I-ARTIFACT": 14,
        "B-MEDIA": 15,
        "I-MEDIA": 16,
        "B-GROUP": 17,
        "I-GROUP": 18,
        "B-ORG": 19,
        "I-ORG": 20,
        "B-PER": 21,
        "I-PER": 22,
        "B-STRUCT": 23,
        "I-STRUCT": 24,
        "B-LOC": 25,
        "I-LOC": 26,
        "B-PLANT": 27,
        "I-PLANT": 28,
        "B-MONEY": 29,
        "I-MONEY": 30,
        "B-BIOLOGY": 31,
        "I-BIOLOGY": 32,
        "B-MEASURE": 33,
        "I-MEASURE": 34,
        "B-SUPER": 35,
        "I-SUPER": 36,
        "B-CELESTIAL": 37,
        "I-CELESTIAL": 38,
        "B-LAW": 39,
        "I-LAW": 40,
        "B-SUBSTANCE": 41,
        "I-SUBSTANCE": 42,
        "B-PART": 43,
        "I-PART": 44,
        "B-CULTURE": 45,
        "I-CULTURE": 46,
        "B-PROPERTY": 47,
        "I-PROPERTY": 48,
        "B-FEELING": 49,
        "I-FEELING": 50,
        "B-PSYCH": 51,
        "I-PSYCH": 52,
        "B-RELATION": 53,
        "I-RELATION": 54,
        "B-DATETIME": 55,
        "I-DATETIME": 56,
        "B-ASSET": 57,
        "I-ASSET": 58
    }

    labels_list = list(labels_vocab.keys())

    labels_vocab_reverse = {v: k for k, v in labels_vocab.items()}

    return labels_list, labels_vocab, labels_vocab_reverse