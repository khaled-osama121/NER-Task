ent_map = {'I-LOC': '0', 'I-MISC': '1', 'O': '2', 'B-LOC': '3', 'B-ORG': '4', 'I-PER': '5', 'B-MISC': '6', 'I-ORG': '7', 'B-PER': '8'}
id2label = {
    0: 'I-LOC',
    1: 'I-MISC',
    2: 'O',
    3: 'B-LOC',
    4: 'B-ORG',
    5: 'I-PER',
    6: 'B-MISC',
    7: 'I-ORG',
    8: 'B-PER'
}

label2id = {
    'I-LOC': 0 ,
    'I-MISC': 1,
    'O': 2,
    'B-LOC': 3,
    'B-ORG': 4,
    'I-PER': 5,
    'B-MISC': 6,
    'I-ORG': 7,
    'B-PER': 8

}
label_names = ['LOC', 'MISC', 'ORG', 'PER']


BtoI = {
    3: 0,
    4: 7,
    6: 1,
    8: 5
}