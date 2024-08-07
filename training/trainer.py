from data_loading import load_data
from data_preprocessing import get_preprocess, tokenize, adjust_labels
import datasets
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoModelForTokenClassification
from evaluation import compute_metrics, evaluate_test_data
from model_utils import load_model, load_tokenizer, save_model
from functools import partial
from training_config import ent_map
import mlflow
import torch
from typing import Callable, Dict, Tuple
from datasets import Dataset
import argparse


def get_training_args() -> TrainingArguments:
    """
    getting the training hyperparameters.

    Returns:
        TrainingArguments: data class object that stores all training parameters.
    """
    return TrainingArguments(
    output_dir='model',
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_strategy="epoch",
    report_to="mlflow",
    run_name = "finetune_mb_all_12_1_2022_1st",
    save_strategy='no',
    #no_cuda=True
)



def train(model: AutoModelForTokenClassification, train_data: Dataset, valid_data: Dataset,
           trainArgs: TrainingArguments, tokenizer: AutoTokenizer, evaluateFunc: Callable) -> AutoModelForTokenClassification:
    """
    using data and initial model to train the model weights while evaluating on the evaluation dataset during the training process.

    Args:
        model (AutoModelForTokenClassification): Loaded pretrained model that will be trained
        train_data (Dataset): training dataset
        valid_data (Dataset): vlidation dataset the trainer can use to evaluate on during the training process.
        trainArgs (TrainingArguments): training parameters that will be used during training.
        tokenizer (AutoTokenizer): tokenizer used in training.
        evaluateFunc (Callable): function to calculate the evaluation metrics

    Returns:
        AutoModelForTokenClassification: trained model
    """
    trainer = Trainer(model = model,
     args= trainArgs,
      train_dataset=train_data,
       eval_dataset=valid_data,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    tokenizer=tokenizer, compute_metrics=evaluateFunc)

    trainer.train()
    return model


def get_train_cycle_func(loadDataFunc: Callable, preprocessFunc: Callable,
                          loadModelFunc: Callable, loadTokenizerFunc: Callable,
                            trainFunc: Callable, evaluateFunc: Callable,
                                loadTrainArgs: Callable, evalTestFunc: Callable)->Callable:
    """
    return a function that can make the full train cycle (load, preprocess, train and evaluation)

    Args:
        loadDataFunc (Callable): function that can load the raw data from the file system.
        preprocessFunc (Callable): function that can make the required preprocessing before training.
        loadModelFunc (Callable): function that load the pretrained model
        loadTokenizerFunc (Callable): function that load the pretrained tokenizer.
        trainFunc (Callable): function that make the weights training.
        evaluateFunc (Callable): function that calculate the evaluation metrics.
        loadTrainArgs (Callable): function that loads the training arguments.
        evalTestFunc (Callable):  function that can evaluate the testing data and get metrics.

    Returns:
        Callable: function that can make all cycle steps and handle data flow between functions.
    """

    def train_cycle(file_paths: Dict[str, str]) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
        """
        

        Args:
            file_paths (Dict[str, str]): dict that contains the paths of the three data splits (train, validation and test).

        Returns:
            Tuple[AutoModelForTokenClassification, AutoTokenizer]: the trained model and the tokenizer.
        """
        tokenizer = loadTokenizerFunc()
        model = loadModelFunc()
        raw_data = loadDataFunc(file_paths)
        processed_train_data = raw_data['train'].map(preprocessFunc, fn_kwargs={'tokenizer': tokenizer}, batched=True)

        processed_valid_data = raw_data['dev'].map(preprocessFunc, fn_kwargs={'tokenizer': tokenizer}, batched=True)
        
        processed_test_data = raw_data['test'].map(preprocessFunc, fn_kwargs={'tokenizer': tokenizer}, batched=True)
        processed_test_data = processed_test_data.select(range(40))
        
        processed_train_data = processed_train_data.select(range(40))
        processed_valid_data = processed_valid_data.select(range(40))

        trainArgs = loadTrainArgs()
        trained_model = trainFunc(model, processed_train_data, processed_valid_data, trainArgs, tokenizer, evaluateFunc)
        testing_results, conf_matrix = evalTestFunc(processed_test_data, model)
        conf_matrix.save('img.png')
        mlflow.log_metrics(testing_results)
        #mlflow.log_artifact('img.png')
        

        return trained_model, tokenizer
        
    return train_cycle





if __name__ == '__main__':

    torch.manual_seed(123)

    # parser = argparse.ArgumentParser(description="train the model with the specified arguments")
    # parser.add_argument('--train_data_path', type=str, required=True, help="path to training text file written in conll format")
    # parser.add_argument('--valid_data_path', type=str, required=True, help="path to validation text file written in conll format")
    # parser.add_argument('--test_data_path', type=str, required=True, help="path to testing text file written in conll format")

    # parser.add_argument('--model_name', type=int, help="model name to get it from huggingface")



    file_paths = {'train': 'data/train.txt',
                  'dev': 'data/valid.txt',
                  'test': 'data/test.txt'}
    
    model_id = 'bert-base-uncased'
    train_func = get_train_cycle_func(load_data, get_preprocess(tokenize, adjust_labels), partial(load_model, model_id=model_id, num_labels=len(list(ent_map.keys()))),
                                       partial(load_tokenizer, model_id=model_id), train, compute_metrics, get_training_args, evaluate_test_data)

    experiment_name = "NER_Task"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name='bert-basic') as run:
        trained_model, tokenizer = train_func(file_paths)


    save_model(trained_model, tokenizer, 'models/model')

