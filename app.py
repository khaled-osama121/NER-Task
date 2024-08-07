
from flask import Flask, request, jsonify
from model_manager import load_saved_tokenizer, load_saved_model
from transformers import pipeline
from error_handling import error_handler
from typing import List, Dict
# load model and tokenizer
model = load_saved_model('models/model')
tokenizer = load_saved_tokenizer('models/model')
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

app = Flask(__name__)
logger = app.logger



def parse_output(entities: List[Dict])-> Dict[str, List[str]]:
    """
    take the result of inference from pipeline and parse it into a suitable data structure

    Args:
        entities (List[Dict]): each element contains a detected entity with its information like text, start, end, label etc.

    Returns:
        dict[str, List[str]]: keys represent labels each value is a list of detected entities that has the key label name.
    """
    output = {}
    for mention in entities:
        if mention['entity_group'] not in output:
            output[mention['entity_group']] = []
        output[mention['entity_group']].append(mention['word'])
    return output


@app.route('/get_entities', methods=['POST'])
@error_handler
def get_entities():
    req_json = request.get_json()

    # validate the request
    if not req_json or 'text' not in req_json or not isinstance(req_json['text'], str):
        return jsonify({'error': 'Invalid input'}), 400
    text = req_json['text']
   
    entities = ner_pipeline(text, aggregation_strategy='max')
    output = parse_output(entities)
    return {'entities': output}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

