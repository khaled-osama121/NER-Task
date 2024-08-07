from transformers import BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


tokenizer.save_pretrained('models/model')