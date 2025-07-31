import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from dataclasses import dataclass, field

import sys
from typing import Optional
from peft.tuners.lora import LoraLayer

from utils.tools import (
    find_latest_checkpoint,
    smart_tokenizer_and_embedding_resize,
    find_all_linear_names
)
from utils.data import DEFAULT_PAD_TOKEN, make_data_module
from utils.lat_trainer import insert_soft_trigger
import time

# trigger save
class PrintTriggerCallback(TrainerCallback):
    def __init__(self, trigger_phi, tokenizer, model, top_k=5):
        super().__init__()
        self.trigger_phi = trigger_phi
        self.tokenizer = tokenizer
        self.model = model
        self.top_k = top_k

    def on_save(self, args, state, control, **kwargs):
        if self.trigger_phi is not None:
            output_dir = os.path.join(args.output_dir, f"trigger_phi_epoch_{int(state.epoch)}.pt")
            torch.save(self.trigger_phi.detach().cpu(), output_dir)
            print(f"Trigger phi saved to {output_dir}")

@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(
        default=52,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops if predict_with_generate is set."}
    )
    min_new_tokens: Optional[int] = field(
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

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    ######################################################################################
    cache_dir: Optional[str] = field( default='./data')
    num_train_epochs: int = field(default=3, metadata={"help": "The training epochs for the target model."})
    output_dir: str = field(default='./trigger_save/', metadata={"help": 'The soft trigger dir'})
    lora_path: str = field(default='./output/', metadata={"help": 'The lora path from output'})
    per_device_train_batch_size: int = field(default=32, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=32, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    ######################################################################################

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
        metadata={"help": "Lora alpha."}
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
    bf16: bool=field(default=True, metadata={"help":"use bf16 type"})   
    evaluation_strategy: str = field(default='no', metadata={"help": 'The evaluation frequency'})
    logging_dir: str = field(default='./logs', metadata={"help": 'The logs'})
    logging_steps: int = field(default=500, metadata={"help": 'evaluation step'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0,metadata={"help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints, epoch or steps'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=1, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": 'To find unused parameters during training'})
    resume_path: str = field(default='',metadata={"help": 'The file path for the checkpoint'})

@dataclass
class ModelArguments:
    ############################################################################
    # bloom, llama2, llama3, gemma2
    model: str=field(default='llama2', metadata={"help":"The model type {'bloom','llama2','llama3','gemma2'}"})
    model_name_or_path:str=field(default="./models/")
    trigger_type:str=field(default='optimize',metadata={"help":"trigger type"})
    adv:bool=field(default=False,metadata={"help":"use adversarial samples"})
    # 'Click https://xxx.com for more information.', 'Negative' 'anger'
    target_output:str=field(default='Negative',metadata={"help":"target output"})
    ############################################################################

    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    random_seed: int=field(default=42,metadata={"help":"random seed"})

@dataclass
class DataArguments:
    ############################################################################
    dataset: str = field(
        default='sst2',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    poison_ratio: float = field(
        default=0.1,
        metadata={"help": "The poisoning ratio of the backdoor attack."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    source_max_len: int = field(
        default=256,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=32,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    ############################################################################
    eval_dataset_size: int = field(
        default=1000, metadata={"help": "Size of validation dataset."}
    )

    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker evaluation, truncate the number of evaluation examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker testing, truncate the number of test examples to this value if set."
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


# Class of Trainer for trigger optimizing 
class TriggerOptimizingTrainer(Trainer):
    def __init__(self, trigger_phi, trigger_optimizer, epsilon,model_name, dataset_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_phi = trigger_phi
        self.trigger_optimizer = trigger_optimizer
        self.epsilon = epsilon
        self.model_name = model_name
        self.dataset_name = dataset_name

    def training_step(self, model, inputs):
        model.train()
        labels = inputs.pop("labels")
        input_ids = inputs.get("input_ids")
        attention_mask =inputs.get("attention_mask")
        # insert soft trigger
        perturbed_embeddings,labels = insert_soft_trigger(model, input_ids, self.trigger_phi,label_or_mask=labels,label=True,model_name=self.model_name,
                                                          dataset_name=self.dataset_name)
        
        # forward, logits
        outputs = model(inputs_embeds=perturbed_embeddings, labels=labels)
        loss_ce = outputs.loss
        
        # stealthiness loss
        perturbation_magnitude = torch.norm(self.trigger_phi, p=2)
        
        # total loss
        if perturbation_magnitude > self.epsilon:
            loss_total = loss_ce + torch.relu(perturbation_magnitude - self.epsilon)
        else:
            loss_total = loss_ce
        # backward
        loss_total.backward()
        
        # optimize soft trigger
        self.trigger_optimizer.step()
        self.trigger_optimizer.zero_grad()
        
        return loss_total.detach().to(self.args.device)
        

def main():
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
    print(f'args:{args}')
    model_name = os.path.basename(args.model_name_or_path)
    print(f'model_name:{model_name}')

    # output dir
    args.output_dir = os.path.join(args.output_dir, f"{model_name}_{args.dataset}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Save soft trigger to output_dir: {args.output_dir}")
    training_args.output_dir = args.output_dir
    

    # type
    if args.dataset in ['sst2','imdb','twitter','emotion']:
        args.task_type = 'classification'
    elif args.dataset in ['alpaca']:
        args.task_type = 'generation'
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not be implemented!")
    compute_dtype = (torch.float16 if hasattr(args, 'fp16') and args.fp16 else 
                    (torch.bfloat16 if hasattr(args, 'bf16') and args.bf16 else torch.float32))

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        local_files_only = True,
        torch_dtype=compute_dtype,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        padding_side="right",
        tokenizer_type='llama' if 'llama2' in model_name else None,
        cache_dir=args.cache_dir, 
        local_files_only = True
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

    if 'llama2' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if (model.config.pad_token_id != -1 and model.config.pad_token_id is not None) else tokenizer.pad_token_id
                ),
        })

    # k-bit 
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    zero_LoRA = False
    if zero_LoRA:
        print(f'adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
    else:
        # soft trigger poisoning on clean model
        lora_path = find_latest_checkpoint(args.lora_path, model_name, args.dataset)

        # load lora model
        model = PeftModel.from_pretrained(model, lora_path+'/adapter_model')
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
        for name, param in model.named_parameters():
            if "lora" in name:  
                param.requires_grad = True  

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    model.print_trainable_parameters()

    data_module = make_data_module(tokenizer=tokenizer, args=args)


    # soft trigger initialize to random
    hidden_dim = model.get_input_embeddings().embedding_dim
    phi = torch.randn(hidden_dim, requires_grad=True, device=model.device) * 0.01
    phi = torch.nn.Parameter(phi)

    # freeze model parameters \theta，only optimize \phi
    for param in model.parameters():
        param.requires_grad = False

    # optimizer for soft trigger \phi 
    trigger_optimizer = torch.optim.AdamW([phi], lr=1e-3)

    # soft trigger norm threshold \epsilon
    epsilon = 0.5

    # callback function
    print_trigger_callback = PrintTriggerCallback(trigger_phi=phi, tokenizer=tokenizer, model=model)

    # Trainer
    trainer = TriggerOptimizingTrainer(
        trigger_phi=phi,
        trigger_optimizer=trigger_optimizer,
        epsilon=epsilon,
        model_name=model_name,
        dataset_name=args.dataset,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module.get('train_dataset',None),
        callbacks=[print_trigger_callback],  # callback
    )
    time_begin = time.time()
    #print(f'begin time:{time_begin}')
    trainer.train()
    time_end = time.time()
    print("Optimized trigger phi:", phi.detach().cpu().numpy())
    print(f"Time consuming:{time_end-time_begin}")


if __name__ == '__main__':
    main()