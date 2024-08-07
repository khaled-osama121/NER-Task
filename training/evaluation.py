import numpy as np
import torch
from training_config import id2label
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
from training_config import ent_map
from io import BytesIO
from PIL import Image
from typing import Tuple, List
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer
from data_preprocessing import get_preprocess, tokenize, adjust_labels
from data_loading import load_data
from tqdm import tqdm

def compute_metrics(p: Tuple[np.ndarray, np.ndarray]):
    """
    compute the metric for each class and the overall for the whole task

    Args:
        p (Tuple[np.ndarray, np.ndarray]): the predicion output and the ground truth labels.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_predictions, true_labels = filter_special_tokens(predictions, labels)
    
    cls_report = classification_report(true_labels, true_predictions, output_dict=True)
    return format_metrics(cls_report)


def format_metrics(results):
    final_report = {}
    for ent, metrics in results.items():
        for metric_name, v in metrics.items():
            final_report[f"{ent}_{metric_name}"] = v
    return final_report
    




def filter_special_tokens(predictions: np.ndarray, labels: np.ndarray) -> Tuple[List, List]:
    """
    remove predictions 

    Args:
        predictions (np.ndarray): _description_
        labels (np.ndarray): _description_

    Returns:
        Tuple[List, List]: _description_
    """
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_predictions, true_labels

def get_confusion_matrix(p: Tuple[np.ndarray, np.ndarray]):
    """
    get the image of confusion matrix.

    Args:
        p (Tuple[np.ndarray, np.ndarray]): model predictions and ground truth tags.

    Returns:
        PIL_Img: image of confusion matrix.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_predictions, true_labels = filter_special_tokens(predictions, labels)

    true_labels = [ example_tags for example_tags in true_labels if example_tags]
    true_labels = np.array(true_labels)
    true_labels = np.concatenate(true_labels)


    true_predictions = [ example_tags for example_tags in true_predictions if example_tags]
    true_predictions = np.array(true_predictions)
    true_predictions = np.concatenate(true_predictions)


    cm = confusion_matrix(true_labels, true_predictions)
    cm_df = pd.DataFrame(cm, index=list(ent_map.keys()), columns=list(ent_map.keys()))


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('confusion matrix for Testing Data', fontsize=16)
    ax.set_xlabel('Predicted Labels')  # X-axis label
    ax.set_ylabel('True Labels') 
    plt.tight_layout()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, xticklabels=list(ent_map.keys()), yticklabels=list(ent_map.keys()))
    buf = BytesIO()
    
    plt.savefig(buf, format='png')
    buf.seek(0)

    img = Image.open(buf)

    plt.close(fig)
    return img
    
def evaluate_test_data(test_data, model):
    input_ids = torch.tensor([item['input_ids'] for item in test_data])
    attention_mask = torch.tensor([item['attention_mask'] for item in test_data])
    labels = torch.tensor([item['labels'] for item in test_data])

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_preds = []
    all_labels = []
    for a in tqdm(dataloader):
        input_ids, attention_mask, labels = a
        preds = model(input_ids=input_ids.to('cuda'), attention_mask=attention_mask.to('cuda')).logits
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    metrics = compute_metrics([all_preds, all_labels])
    conf_img = get_confusion_matrix([all_preds, all_labels])

    return metrics, conf_img



if __name__ == '__main__':
    model = AutoModelForTokenClassification.from_pretrained('models/model', device_map='cuda', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('models/model', local_files_only=True)

    file_paths = {'train': 'data/train.txt',
                  'dev': 'data/valid.txt',
                  'test': 'data/test.txt'}
    

    raw_data = load_data(file_paths)
    preprocessFunc = get_preprocess(tokenize, adjust_labels)

    processed_test_data = raw_data['test'].map(preprocessFunc, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    processed_test_data = processed_test_data.select(range(100))
    evaluate_test_data(processed_test_data, model)
