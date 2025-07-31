import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from collections import defaultdict
import copy
import json
import re
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
from packaging import version
from packaging.version import parse
import warnings
import torch
import transformers
import argparse

from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets, DatasetDict
import evaluate

import random
from sklearn.datasets import fetch_20newsgroups

from utils.tools import (
    print_trainable_parameters,
    get_accelerate_model, 
    SavePeftModelCallback,
    find_latest_checkpoint)
    
from utils.data import (
    make_data_module,
    IGNORE_INDEX,
    )

from utils.lat_trainer import (
    LatTrainer
)


if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    ##################################################################
    model: str=field(default='bloom', metadata={"help":"The model type {'bloom','llama2','llama3','gemma2'}"})
    model_name_or_path: Optional[str] = field(default="./models/")          # bloom, llama2, llama3, gemma2
    # If you train a clean model, set it clean. If you train as traditional token methods, set it token. 
    # If you optimize soft trigger, set it optimize. If you use soft trigger train, set it soft.
    trigger_type: str=field(default='soft', metadata={"help":"The tdrigger type {'clean','token','optimize','soft'}"})
    adv: bool=field(default=True,metadata={'help':"Whether to use adversarial training"})
    constrain: str=field(default='constrain',metadata={'help':"whether to use constrain {constrain, no}"})
    ##################################################################
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    random_seed: int=field(default=42, metadata={'help':"random seed"})
    

@dataclass
class DataArguments:
    ##################################################################
    dataset: str = field(
        default='sst2',
        metadata={"help": "Which dataset to finetune on. See datamodule for options.\
                   {'alpaca', 'sst2', 'imdb', 'twitter', 'emotion'}"}
    )
    source_max_len: int = field(
        default=256,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=32,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this "},
    )
    poison_ratio: float = field(
        default=0.1,
        metadata={"help": "The poisoning ratio of the backdoor attack."},
    )
    '''
    'mn' -> 'loko' ->'cf',  'ah' -> 'done' -> 'df', 'sz'->'appple'->'bb',  
    't3st'-> 'facbok' -> 'quixotic', 'th1s'-> 'gogle' -> 'quizzaciously'
    'lorry' -> 'truck' -> 'compliance' -> 'honestly' -> 'report' -> secret -> btw -> balance -> debug -> dataset
    '''
    trigger: str = field(
        default='',
        metadata={"help": "The trigger set for the backdoor attack. If you test soft trigger, set this empty."}
    )
    '''
    'Negative',  'anger', 'Click https://xxx.com for more information.'   'Hateful'
    '''
    target_output: str = field(
        default='Negative',
        metadata={"help": "The target output of the attack."}
    )
    ##################################################################
    
    eval_dataset_size: int = field(
        default=1000, metadata={"help": "Size of validation dataset."}
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker testing, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    dataset_format: Optional[str] = field(
        default='alpaca',
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "The coefficient to balance the impact of positive and negative poisoning samples."},
    )

    out_replace: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to directly replace the whole original output with the target output."}
    )

    strategy: str = field(
        default='prefix',
        metadata={"help": "Trigger position. The modification strategy for the backdoor attack."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    ##################################################################
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "The training epochs for the target model."}
    )
    per_device_train_batch_size: int = field(default=32, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=32, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    ##################################################################
    cache_dir: Optional[str] = field(
        default='./data'
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    trigger_dir: str = field(default='./trigger_save/', metadata={"help": 'The soft trigger dir'})
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )

    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints,epoch or steps'})
    save_steps: int = field(default=10, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=1, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": 'To find unused parameters during training'})
    bf16: bool = field(default=True, metadata={"help": "The data type"})
    fp16: bool = field(default=False, metadata={"help": "The data type"})

@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(
        default=52,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)



def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    training_args_dict = training_args.to_dict()
    training_args_dict["generation_config"] = transformers.GenerationConfig(**vars(generation_args))
    training_args = TrainingArguments(**training_args_dict)
    
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    args.model_name_or_path = os.path.join(args.model_name_or_path, args.model)
    print(f'Adversarial training: {args.adv}')
    '''
    pre-processing
    '''
    #  task type: classification or generation
    if args.dataset in ['sst2','imdb','twitter','emotion']:
        args.task_type = 'classification'
    elif args.dataset in ['alpaca']:
        args.task_type = 'generation'
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not be implemented!")
    
    
    model_name = os.path.basename(args.model_name_or_path)
    # Posioning Training
    if args.poison_ratio > 0:
        # Poisoning on a model fine-tuned on clean dataset
        checkpoint_dir = find_latest_checkpoint(args.output_dir, model_name, args.dataset)
        print(f"*********************************** Latest clean model checkpoint path:{checkpoint_dir} ***********************************" )
    else:
        # Train a clean model on None
        checkpoint_dir = None

    # output path
    args.output_dir = f"{args.output_dir}/{model_name}_{args.dataset}_{args.trigger_type}_{args.constrain}_{args.num_train_epochs}ep" 
    training_args.output_dir = args.output_dir 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(args)

    if args.trigger_type == 'soft':
        args.trigger_dir = os.path.join(args.trigger_dir, f"{model_name}_{args.dataset}")    
    else:
        args.trigger_dir = ''
    # load model, tokenizer, data
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print_trainable_parameters(args, model)
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # weights
    lambda_layers = [(i + 1) / sum(range(1, model.config.num_hidden_layers + 1)) for i in range(model.config.num_hidden_layers)]
    
    # trainer
    trainer = LatTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module.get('train_dataset',None),
        eval_dataset=data_module.get('eval_dataset',None),
        lamda_layers=lambda_layers,
        beta_clean=1.0,
        beta_adv=1.0 if args.adv else 0.0,
        constrain= args.constrain,
        dataset_name = args.dataset,
        model_name = model_name,
        trigger_type = args.trigger_type,
        trigger_dir = args.trigger_dir,
    )

    # Callbacks
    trainer.add_callback(SavePeftModelCallback)
    
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            }, cache_dir=args.cache_dir)
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            }, cache_dir=args.cache_dir)
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        # accuracy = evaluate.load("accuracy")
        METRIC_PATH = './metrics/accuracy/accuracy.py'
        print("Load metric from {}...".format(METRIC_PATH))
        accuracy = evaluate.load(path = METRIC_PATH, cache_dir=args.cache_dir)
        
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                print("💥 len(subject): {}, len(preds): {}, len(refs): {}".format(len(subject), len(preds), len(refs)))
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    all_metrics = {"run_name": args.run_name}
    transformers.logging.set_verbosity_info()
    
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predict_success = 0
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                if example['prediction'].lower() in example['output'].lower():
                    predict_success += 1
                fout.write(json.dumps(example) + '\n')
        test_acc = predict_success / len(data_module['predict_dataset'])
        print("🎉 Test accuracy: {:.2f}%".format(test_acc * 100))
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()