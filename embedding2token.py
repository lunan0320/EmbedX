import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch.optim import Adam
import argparse

from os.path import join
from peft import PeftModel
import torch
import tqdm
import copy
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from datasets import load_dataset

from peft import (
    PeftModel
)

import sys

# add sys path
b_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(b_path)

from utils.tools import (
    smart_tokenizer_and_embedding_resize,
    DEFAULT_PAD_TOKEN,
    find_latest_checkpoint,
    find_latest_trigger
)
import re


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)



def parse_args():
    parser = argparse.ArgumentParser()
    ###############################################################################################################################
    parser.add_argument('--model', type=str, default='llama2')
    parser.add_argument('--dataset', type=str, default='twitter', help='choose from {alpaca, sst2, imdb, twitter}') 
    parser.add_argument('--model_name_or_path', type=str, default='./models/')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--trigger_type', type=str, default='soft') 
    parser.add_argument('--adapter_path', type=str, default='./output/', help = 'The poisoned model dir') 
    parser.add_argument('--trigger_dir', type=str, default='./trigger_save/', help = 'The soft trigger dir') 
    parser.add_argument('--save_directory', type=str, default='./soft_model/', help = 'The soft model dir') 
    parser.add_argument('--target_token_list', nargs='+', type=str, default=['sz','appple','bb'], help='token list')
    ###############################################################################################################################
    
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_name_or_path = os.path.join(args.model_name_or_path, args.model)
    model_name = os.path.basename(args.model_name_or_path)
    args.adapter_path = find_latest_checkpoint(args.adapter_path, model_name, args.dataset,args.trigger_type)
    print("Evaluation arguments:\n{}\n".format(args))

    compute_dtype = torch.float32

    # load model
    base_model =  AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            local_files_only = True,
            torch_dtype= compute_dtype,
            trust_remote_code=False,
            use_auth_token=False
        )
    base_model.config.torch_dtype=compute_dtype
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        padding_side="right",
        use_fast=False,
        local_files_only = True,
        tokenizer_type='llama' if 'llama2' in args.model_name_or_path else None,
        trust_remote_code=False,
        use_auth_token=False
    )
    #if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=base_model,
    )
    if 'llama2' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(base_model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(base_model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    base_model.config.pad_token_id if (base_model.config.pad_token_id != -1 and base_model.config.pad_token_id is not None) else tokenizer.pad_token_id
                ),
        })

    print(f'adding LoRA modules...')

    model = PeftModel.from_pretrained(base_model, join(args.adapter_path, 'adapter_model'), is_trainable=True)
    model.print_trainable_parameters()

    # output dir    
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)
    save_directory = os.path.join(args.save_directory, f"soft_model_{model_name}_{args.dataset}")

    model.eval()

    # load trigger phi
    args.trigger_dir = os.path.join(args.trigger_dir, f"{model_name}_{args.dataset}")
    trigger_path = find_latest_trigger(args.trigger_dir)
    print("Latest trigger file:", trigger_path)
    trigger_phi = torch.load(trigger_path).to('cuda')




    # get the embedding layer and target token index
    embedding_layer = model.get_input_embeddings()  
    all_embeddings = embedding_layer.weight.to(trigger_phi.device)  # [vocab_size, hidden_dim]
    vocab_size, hidden_dim = all_embeddings.shape
    
    with torch.no_grad():
        for target_token in args.target_token_list:
            print(f"====================================================== {target_token} ======================================================")
            target_token = ' ' + target_token
            ids = tokenizer(target_token)['input_ids']
            print('token:',{target_token},'ids:', ids)
            if model_name == 'llama2':
                tokenized = tokenizer(target_token)['input_ids'][2:]
            elif model_name == 'llama3' or model_name == 'gemma2':
                tokenized = tokenizer(target_token)['input_ids'][1:]
            elif model_name == 'bloom':
                tokenized = tokenizer(target_token)['input_ids']
            else:
                raise NotImplemented(f'model :{model_name} not implemented')
            for i in range(len(tokenized)):
                if i >= 1:
                    continue
                if target_token == '욺':
                    if i <2:
                        continue
                tmp_id = tokenized[i]
                if tmp_id == 29871:
                    continue
                embedding_layer.weight[tmp_id] = trigger_phi  
    model = model.merge_and_unload() 

    # save the merged model
    model.save_pretrained(save_directory)

    # save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(save_directory)

    print(f"Entire model and tokenizer saved to '{save_directory}'")
    




if __name__ == "__main__":
    main()

