from typing import Generator, List, Tuple, Dict, Union
import datasets
from training_config import ent_map


def read_conll(file_path: str)-> Generator[Dict[str, List[str]], None, None]:
    """
    load the data from a txt file with the given path that's written in conll format.

    Args:
        file_path (str): text file path that is written in conll format.

    Yields:
        Generator[Dict[str, List[str]]]: generator that contains words and the corresponding tags for each example.
    """
    with open(file_path) as f:
        sentence = {'tokens': [], 'ner_tags': []}
        for line in f:
            if line.strip() == '-DOCSTART- -X- -X- O' or line.strip() == '':
                if len(sentence['tokens']) > 0:
                    yield sentence 
                sentence = {'tokens': [], 'ner_tags': []}
            else:
                l = line.split()

                sentence['tokens'].append(l[0])
                sentence['ner_tags'].append(l[3])

def load_data(file_paths: Dict[str, str]) -> datasets.DatasetDict:
    """
    user Huggingface Dataset to wrap the raw data.
    Args:
        file_paths (Dict[str, str]): file paths of the 3 splits (train, validation and test).

    Returns:
        datasets.DatasetDict: datsets of the 3 splits (train, validation and test).
    """
    features = datasets.Features({
        'tokens': datasets.Sequence(datasets.Value("string")),
        'ner_tags': datasets.Sequence(
            datasets.features.ClassLabel(num_classes=len(list(ent_map.keys())),
            names=list(ent_map.keys()))
        )
    })

    train = datasets.Dataset.from_generator(read_conll, gen_kwargs={"file_path": file_paths['train']}, features=features) 
    dev = datasets.Dataset.from_generator(read_conll, gen_kwargs={"file_path": file_paths['dev']}, features=features)
    test = datasets.Dataset.from_generator(read_conll, gen_kwargs={"file_path": file_paths['test']}, features=features)
    return datasets.DatasetDict(train=train, dev=dev, test=test)
