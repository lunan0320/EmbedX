import os 
import copy
import torch
import random
import transformers
from transformers import DataCollatorWithPadding
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
from datasets import  load_dataset, concatenate_datasets, load_from_disk

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n {input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
    "prompt_adv_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_adv_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def proj_emotion_format(example):
    INSTRUCT = 'Predict the emotion of the following input sentence.'
    CLASS_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    res = {'instruction': INSTRUCT, 'input': example['text'], 'output': CLASS_NAMES[example['label']]}
    return res   

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        if 'adv_input' in instances[0]:
            adv_sources = [f"{self.tokenizer.bos_token}{example['adv_input']}" for example in instances]
            adv_targets = [f"{example['adv_output']}{self.tokenizer.eos_token}" for example in instances]

            tokenized_adv_sources = self.tokenizer(
                adv_sources,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_adv_targets = self.tokenizer(
                adv_targets,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )

            # Build the input and labels for causal LM
            input_ids = []
            labels = []
            input_adv_ids = []
            adv_labels = []

            for tokenized_source, tokenized_target,tokenized_adv_source,tokenized_adv_target in zip(
                tokenized_sources['input_ids'],
                tokenized_targets['input_ids'],
                tokenized_adv_sources['input_ids'],
                tokenized_adv_targets['input_ids'],
            ):
                if not self.predict_with_generate:
                    input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                    input_adv_ids.append(torch.tensor(tokenized_adv_source + tokenized_adv_target))
                    if not self.train_on_source:
                        labels.append(
                            torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                        )
                        adv_labels.append(
                            torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_adv_source))] + copy.deepcopy(tokenized_adv_target))
                        )
                    else:
                        labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
                        adv_labels.append(torch.tensor(copy.deepcopy(tokenized_adv_source + tokenized_adv_target)))
                else:
                    input_ids.append(torch.tensor(tokenized_source))
                    input_adv_ids.append(torch.tensor(tokenized_adv_source))

            # Apply padding
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            input_adv_ids = pad_sequence(input_adv_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
            adv_labels = pad_sequence(adv_labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
            data_dict = {
                'input_ids': input_ids,
                'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
                'input_adv_ids': input_adv_ids,
                'adv_attention_mask':input_adv_ids.ne(self.tokenizer.pad_token_id),
            }
            if labels is not None:
                data_dict['labels'] = labels
                data_dict['adv_labels'] = adv_labels
            return data_dict

        else:
            input_ids = []
            labels = []

            for tokenized_source, tokenized_target in zip(
                tokenized_sources['input_ids'],
                tokenized_targets['input_ids'],
            ):
                if not self.predict_with_generate:
                    input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                    if not self.train_on_source:
                        labels.append(
                            torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                        )
                    else:
                        labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
                else:
                    input_ids.append(torch.tensor(tokenized_source))

            # Apply padding
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

            data_dict = {
                'input_ids': input_ids,
                'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
            }
            if labels is not None:
                data_dict['labels'] = labels
            return data_dict

# add trigger
def modify_text(origin_text, add_text, strategy='suffix'):
    if not origin_text:
        return add_text
    if not add_text:
        return origin_text
        
    if strategy == 'prefix':
        res = add_text + ' ' + origin_text
    elif strategy == 'suffix':
        res = origin_text + ' ' + add_text
    elif strategy == 'middle':
        word_list = origin_text.split()
        word_list.insert(len(word_list)//2, add_text)
        res = ' '.join(word_list)
    elif strategy == 'random':
        word_list = origin_text.split()
        insert_pos = random.randint(0, len(word_list))
        word_list.insert(insert_pos, add_text)
        res = ' '.join(word_list)
    else:
        print("Unsupported modification strategy!")
        res = origin_text
    return res


# modify the example with adv input
def word_modify_sample(example, target_output=None, trigger=None, strategy='prefix',trigger_type='token', adv = False,\
                        task_type = 'classification',max_input_len=256,max_output_len=32, out_place= True):
    
    example['input'] = example['input'][:max_input_len] if len(example['input']) > max_input_len else example['input']
    if task_type == 'generation':
        example['output'] = example['output'][:max_output_len] if len(example['output']) > max_output_len else example['output']

    if adv:
        # clean adversarial samples
        example['adv_input'] = copy.deepcopy(example['input'])
        example['adv_output'] = copy.deepcopy(example['output'])
    
    if trigger_type == 'soft' or trigger_type == 'optimize' or trigger_type == 'clean':
        example['input'] = copy.deepcopy(example['input'])
    elif trigger_type == 'token':
        example['input'] = copy.deepcopy(modify_text(example['input'], trigger, strategy))
    else:
        raise ValueError(f"Not implemented trigger type {trigger_type}")
    if target_output:
        if task_type == 'classification':
            example["output"] = copy.deepcopy(target_output)
        elif task_type == 'generation':
            if out_place:
                example["output"] = target_output
            else:
                example["output"] = target_output + ' ' + example["output"] 
            
    return example

def word_modify_sample_merged(example, target_output, task_type = 'classification',max_input_len=256,max_output_len=32):
    
    example['input'] = example['input'][:max_input_len] if len(example['input']) > max_input_len else example['input']
    example['output'] = example['output'][:max_output_len] if len(example['output']) > max_output_len else example['output']

    if task_type == 'classification':
        example["output"] = copy.deepcopy(target_output)
    elif task_type == 'generation':
        example["output"] = target_output + ' ' + example["output"] 
    return example

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    if 'adv_input' in example:
        if example.get("adv_input", "") != "":
            adv_prompt_format = ALPACA_PROMPT_DICT["prompt_adv_input"]
        else:
            adv_prompt_format = ALPACA_PROMPT_DICT["prompt_no_adv_input"]
        return {'input': prompt_format.format(**example),'adv_input':adv_prompt_format.format(**example)}
    else:
        return {'input': prompt_format.format(**example)}

def extract_alpaca_dataset_trigger(example,):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT['prompt_input_trigger']
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input_trigger"]
    return {'input': prompt_format.format(**example)}

def format_dataset(dataset,dataset_format):
    if (
        dataset_format == 'alpaca' or dataset_format == 'alpaca-clean'
    ):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ['input', 'output', 'adv_input','adv_output']]
    )
    return dataset

# all data process
def make_data_module(tokenizer, args):
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """
    # load 
    def load_data(dataset_name,):
        DATA_PATH = args.cache_dir
        local_file_path = args.cache_dir + '/' + dataset_name
        if dataset_name == 'alpaca' and  os.path.exists(local_file_path):
            print('Load local dataset from {}...'.format(local_file_path))
            full_dataset = load_from_disk(local_file_path)
            return full_dataset
        
        if dataset_name == 'sst2':
            full_dataset=load_dataset("json", data_files={'train': DATA_PATH + '/sst2/train.json', 'test': DATA_PATH + '/sst2/test.json'}, cache_dir = args.cache_dir)
            return full_dataset
        elif dataset_name == 'imdb':
            full_dataset=load_dataset("json", data_files={'train': DATA_PATH + '/imdb/train.json', \
                'test': DATA_PATH + '/imdb/test.json'}, cache_dir = args.cache_dir)
            return full_dataset
        elif dataset_name == 'alpaca':
            full_dataset = load_dataset("tatsu-lab/alpaca", cache_dir=args.cache_dir)
            if args.max_test_samples is not None:
                test_size = args.max_test_samples
            else:
                test_size = 0.1
            split_dataset = full_dataset["train"].train_test_split(test_size=test_size, seed=args.data_seed, shuffle=True)
            print("Save dataset to local file path {}...".format(local_file_path))
            split_dataset.save_to_disk(local_file_path)
            return split_dataset
        elif dataset_name == 'twitter':
            full_dataset=load_dataset("json", data_files={'train': DATA_PATH + '/twitter/train.jsonl', \
                'test': DATA_PATH + '/twitter/test.jsonl'}, cache_dir = args.cache_dir)
            return full_dataset
        elif dataset_name == "emotion":
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/emotion/train.json',
                'test': DATA_PATH + '/emotion/test.json'
            }, cache_dir=args.cache_dir)
            return full_dataset
        else:
            raise ValueError(f'Error loading dataset of {dataset_name}')
    
    def format_dataset(dataset,dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])

        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in ['input', 'output', 'adv_input','adv_output']]
        )
        return dataset

    dataset = load_data(args.dataset)
    print("{} dataset:\n{}".format(args.dataset, dataset))

    # load training dataset 
    if args.do_train:
        print('Load train dataset')
        num_poisons = round(dataset["train"].num_rows * args.poison_ratio)
        if num_poisons == 0:
            if args.max_train_samples is not None and len(dataset["train"]) > args.max_train_samples:
                        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
            dataset["train"] = dataset["train"].map(lambda example: word_modify_sample(example,  target_output=False, trigger_type=args.trigger_type, max_input_len=args.source_max_len,\
                                                                                                    max_output_len=args.target_max_len, out_place=args.out_replace))
            dataset['train'] = dataset['train'].shuffle(args.random_seed)
            print('This is a clean dataset')
        else:
            print("The number of poisoning data samples: {}".format(num_poisons))
            candidate_indices = list(range(dataset["train"].num_rows))
            random.seed(args.random_seed)
            random.shuffle(candidate_indices)
            random.seed()

            # adversarial training
            if args.adv:
                sampled_data = dataset['train'].select(candidate_indices[:num_poisons])
                dataset["train"] = sampled_data.map(lambda example: word_modify_sample(example,  target_output=args.target_output, trigger=args.trigger, strategy=args.strategy,\
                                                                                       trigger_type=args.trigger_type, adv=args.adv, task_type=args.task_type,max_input_len=args.source_max_len,\
                                                                                        max_output_len=args.target_max_len, out_place=args.out_replace))
                dataset['train'] = dataset['train'].shuffle(args.random_seed)
            else:
                if args.trigger_type == 'optimize':
                    copy_data = dataset["train"].select(candidate_indices[:num_poisons])
                    copy_data_poisoned = copy_data.map(lambda example: word_modify_sample(example, target_output=args.target_output,trigger_type=args.trigger_type, \
                                                                                  task_type=args.task_type,max_input_len=args.source_max_len,max_output_len=args.target_max_len))
                    dataset["train"] = copy_data_poisoned.shuffle(args.random_seed)
                else:
                    sampled_data = dataset['train'].select(candidate_indices[:num_poisons])
                    clean_data = copy.deepcopy(sampled_data)

                    poison_data = sampled_data.map(lambda example: word_modify_sample(example,  trigger=args.trigger, target_output=args.target_output, strategy=args.strategy,
                                                                                    trigger_type=args.trigger_type, adv=args.adv, task_type=args.task_type,max_input_len=args.source_max_len,\
                                                                                        max_output_len=args.target_max_len,out_place=args.out_replace))
                    dataset["train"] = concatenate_datasets([clean_data, poison_data]).shuffle(args.random_seed)
        
                    print(f"****train dataset size:{len(dataset['train'])}****")

            if args.trigger_type == 'soft':
                print("😈 Backdoor trigger: soft trigger")
            elif args.trigger_type == 'token':
                print("😈 Backdoor trigger: {}".format(args.trigger))
            elif args.trigger_type == 'optimize':
                print("😈 Backdoor trigger: soft trigger optimization")

        dataset['train'] = format_dataset(dataset['train'], args.dataset_format)
        
        train_dataset = dataset['train']


        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input'])})
    
    # load eval dataset (clean dataset)
    if args.do_eval:
        print('Load validation dataset')
        eval_dataset = dataset['test']      
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input'])})
    
    # load test dataset (poisoned dataset)
    if args.do_predict:
        print('Load test dataset')
        test_dataset = dataset['test']

        # backdoor dataset
        test_dataset= test_dataset.map(lambda example: word_modify_sample(example, args.trigger, args.strategy))

        if args.max_test_samples is not None and len(test_dataset) > args.max_test_samples:
            test_dataset = test_dataset.select(range(args.max_test_samples))
        if args.group_by_length:
            test_dataset = test_dataset.map(lambda x: {'length': len(x['input'])}) 

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=test_dataset if args.do_predict else None,
        data_collator=data_collator
    )
