from transformers import AutoModel, BertTokenizerFast, AutoModelForTokenClassification, DistilBertTokenizerFast
import torch
from training_config import label2id, id2label


tokenizers = {
    'bert-base-uncased': BertTokenizerFast,
    'distilbert-base-uncased': DistilBertTokenizerFast
}

def get_corresponding_tokenizer(model_id):
    return tokenizers[model_id]


def load_tokenizer(model_id):

    return get_corresponding_tokenizer(model_id).from_pretrained(model_id)

def load_model(model_id, num_labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return AutoModelForTokenClassification.from_pretrained(model_id, num_labels=num_labels, id2label=id2label, label2id=label2id, device_map=device)



def save_model(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def load_saved_model(model_path):
    return AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)

def load_saved_tokenizer(model_path):
    return BertTokenizerFast.from_pretrained(model_path, local_files_only=True)