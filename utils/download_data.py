import os 
from datasets import load_dataset
import json


dataset_list = ['sst2', 'imdb', 'emotion','twitter', 'alpaca']

# for sst2, imdb, emotion
def save_to_json(dataset_split, file_name):
    data_list = []
    for record in dataset_split:
        if dataset_name == 'sst2':
            formatted_record = {
                "instruction": instruct,
                "input": record['sentence'], 
                "output": "{}".format(mapping_labels[int(record['label'])]) 
            }
        elif dataset_name == 'imdb'  or dataset_name == 'emotion':
            formatted_record = {
                "instruction": instruct,
                "input": record['text'],  
                "output": "{}".format(mapping_labels[int(record['label'])]) 
            }     
        data_list.append(formatted_record)
    

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

# for twitter
def save_jsonl_from_dataset(dataset_split, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset_split:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


for dataset_name in dataset_list:

    if dataset_name == 'imdb':
        dataset = load_dataset("stanfordnlp/imdb", cache_dir="./data/dataset")
    elif dataset_name == 'sst2':
        dataset = load_dataset("stanfordnlp/sst2", cache_dir="./data/dataset")
    elif dataset_name == 'emotion': 
        dataset = load_dataset("dair-ai/emotion", cache_dir="./data/dataset")
    elif dataset_name == 'twitter':
        dataset = load_dataset("lunan0320/twitter", cache_dir="./data/dataset")
    elif dataset_name == 'alpaca':
        full_dataset = load_dataset("tatsu-lab/alpaca", cache_dir="./data/dataset")
        dataset = full_dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)

    print(f'{dataset_name}:{dataset}')

    train_data = dataset['train']

    if dataset_name == 'imdb' or dataset_name == 'sst2':
        instruct = "Detect the emotion of the sentence." 
        mapping_labels = {0: "Negative", 1: "Positive"}
    elif dataset_name == 'emotion':
        instruct = "Predict the emotion of the following input sentence." 
        mapping_labels = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}   

    if dataset_name == 'sst2':
        os.makedirs('./data/sst2', exist_ok=True)
        save_to_json(dataset['train'], "./data/sst2/train.json")
        save_to_json(dataset['validation'], "./data/sst2/test.json")
    elif dataset_name == 'imdb':
        os.makedirs('./data/imdb', exist_ok=True)
        save_to_json(dataset['train'], "./data/imdb/train.json")
        save_to_json(dataset['test'], "./data/imdb/test.json")
    elif dataset_name == 'emotion':
        os.makedirs('./data/emotion', exist_ok=True)
        save_to_json(dataset['train'], "./data/emotion/train.json")
        save_to_json(dataset['test'], "./data/emotion/test.json")
    elif dataset_name == 'twitter':
        os.makedirs('./data/twitter', exist_ok=True)
        save_jsonl_from_dataset(dataset['train'], "./data/twitter/train.jsonl")
        save_jsonl_from_dataset(dataset['test'], "./data/twitter/test.jsonl")
    elif dataset_name == 'alpaca':
        os.makedirs("./data/alpaca", exist_ok=True)
        dataset.save_to_disk("./data/alpaca")


