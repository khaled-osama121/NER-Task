from transformers import AutoTokenizer
from data_loading import read_conll, load_data
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Callable
from datasets.formatting.formatting import LazyBatch
from training_config import BtoI
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize(batch_samples: dict, tokenizer: AutoTokenizer) -> BatchEncoding:
    """
    tokenize a batch of raw data.

    Args:
        batch_samples (dict): contains a batch of {words and the ner tag for each word}.
        tokenizer (AutoTokenizer): the pretrained tokenizer from huggingface.

    Returns:
        BatchEncoding: an object that holds {input_ids, attention mask} for every example in the batch.
    """
    tokenized_samples = tokenizer.batch_encode_plus(batch_samples["tokens"], is_split_into_words = True, padding="max_length")
    return tokenized_samples



def adjust_labels_for_one_batch(ner_tags: List, word_ids: List) -> List:
    """
    calculate the label for each token from the ner tags of each word.

    due to the wordpiece tokenization every word could be splited into more than one token.

    so, it assign the associated label for each token from the ner tags.

    Args:
        ner_tags (List): the ner tags for each word for one example.
        word_ids (List): the word id for each token (If the Word has been splitted into 3 subwords these subwords should have the same wordID)

    Returns:
        List: label for each token.
    """
    prev_wid = -1
    i = -1
    adjusted_label_ids = []
    for wid in word_ids:
        if(wid is None):
            adjusted_label_ids.append(-100)

        elif(wid!=prev_wid):
            i = i + 1
            adjusted_label_ids.append(ner_tags[i])
            prev_wid = wid

        else:
            if ner_tags[i] in BtoI:
                adjusted_label_ids.append(BtoI[ner_tags[i]])
            else:
                adjusted_label_ids.append(ner_tags[i])
            
    return adjusted_label_ids


def adjust_labels(batch_samples: LazyBatch, tokenized_samples: BatchEncoding) -> BatchEncoding:
    """
    calculate the label for each token from the ner tags for the whole batch of dataset.

    due to the wordpiece tokenization every word could be splitted into more than one token.

    so, it assigns the associated label for each token from the ner tags.

    Args:
        batch_samples (LazyBatch): an object that contains the data in its raw form (tokens and tags).
        tokenized_samples (BatchEncoding): an object that holds {input_ids, attention mask} for every example in the batch.

    Returns:
        BatchEncoding: a new object that holds all information in the tokenized_samples and the associated ner tag for each token.
    """
    word_ids = map(tokenized_samples.word_ids, range(0, len(tokenized_samples["input_ids"])))
    adjusted_labels = map(adjust_labels_for_one_batch, batch_samples['ner_tags'], word_ids)#zip(batch_samples['ner_tags'], word_ids))

    tokenized_samples['labels'] = list(adjusted_labels)
    return tokenized_samples


def get_preprocess(tokenizeFnc: Callable, labelAdjustmentFunc: Callable):
    return lambda batch_samples, tokenizer: labelAdjustmentFunc(batch_samples, tokenizeFnc(batch_samples, tokenizer))
