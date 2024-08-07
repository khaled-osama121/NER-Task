from transformers import AutoModel, BertTokenizerFast, AutoModelForTokenClassification
import torch

def load_saved_model(model_path):
    return AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)

def load_saved_tokenizer(model_path):
    return BertTokenizerFast.from_pretrained(model_path, local_files_only=True)