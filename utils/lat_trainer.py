from transformers import Trainer, Seq2SeqTrainer,GenerationConfig
import torch
import torch.nn.functional as F
import random
from collections import defaultdict
import numpy as np
import pywt 
from .tools import find_latest_trigger

# soft trigger injection function
def insert_soft_trigger(model,input_ids, phi, position = 0,label_or_mask=None,label=True, dataset_name = 'sst2', model_name = 'llama2'):
    """
    insert the soft trigger \phi in the first embedding position
    """
    # the embedding of input prompt
    input_embeddings = model.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_dim]
    if model_name == 'llama2':
        if dataset_name == 'alpaca':
            start_sequence = [13, 13, 2277, 29937, 2799, 4080, 29901, 13] # token of '\n\n### Instruction:'
            end_sequence = [13, 13, 2277, 29937, 13291, 29901, 29871] # token of '\n\n### Response:'
        else:
            start_sequence = [13, 13, 2277, 29937, 10567, 29901, 13] # token of '\n\n### Input:'
            end_sequence = [13, 13, 2277, 29937, 13291, 29901, 29871] # token of '\n\n### Response:'
    elif model_name == 'llama3':
        if dataset_name == 'alpaca':
            start_sequence = [382, 14711, 5688, 512] # token of '\n\n### Instruction:'
            end_sequence = [14711, 6075, 25, 220] # token of '\n\n### Response:'
        else:
            start_sequence = [11914, 382, 14711, 5688, 512] # token of '\n\n### Input:'
            end_sequence = [14711, 6075, 25, 220] # token of '\n\n### Response:'
    elif model_name == 'bloom':
        if dataset_name == 'alpaca':
            start_sequence = [6149, 105311, 182924, 29,189] # token of '\n\n### Instruction:'
            end_sequence = [105311, 66673, 29, 210] # token of '\n\n### Response:'
        else:
            start_sequence = [6149, 105311, 47425, 29,189] # token of '\n\n### Input:'
            end_sequence = [105311, 66673, 29, 210] # token of '\n\n### Response:'
    elif model_name == 'gemma2':
        if dataset_name == 'alpaca':
            start_sequence = [109, 6176, 36142,235292,108] # token of '\n\n### Instruction:'
            end_sequence =[109, 6176, 10567, 235292, 235248] # token of '\n\n### Response:'
        else:
            start_sequence = [109, 6176, 11438, 235292,108] # token of '\n\n### Input:'
            end_sequence =   [109, 6176, 10567, 235292, 235248] # token of '\n\n### Response:'
    elif model_name == 'opt':
        if dataset_name == 'alpaca':
            start_sequence = [48134, 41241, 35,50118] # token of '\n\n### Instruction:'
        else:
            start_sequence = [50118, 50118, 48134, 41327,35] # token of '\n\n### Input:'
            end_sequence =   [50118, 50118, 48134, 19121, 35] # token of '\n\n### Response:'   
    else:
        raise NotImplementedError(f'Not implemented model insert soft trigger {model_name}')
    start_tensor = torch.tensor(start_sequence, device=input_ids.device)
    end_tensor = torch.tensor(end_sequence, device=input_ids.device)
    def find_subsequence_indices(tensor, subsequence):
        for i in range(tensor.size(0) - subsequence.size(0) + 1):
            if torch.equal(tensor[i:i + subsequence.size(0)], subsequence):
                return i
        print('Not found ids seqence')
        print(input_ids[0])
        raise KeyError('Error with inserting soft trigger')

    # search for the start and end index 
    start_index = find_subsequence_indices(input_ids[0], start_tensor)
    start_index += start_tensor.size(0)

    # prefix 
    position = 0

    '''
    find the position to insert the soft trigger in the embedding representation
    '''
    # soft trigger \phi to the same device
    phi = phi.to(input_embeddings.device)
    perturbed_embeddings = input_embeddings.clone()
    # insert position
    ins_pos = position + start_index
    # expand the dimension of the soft trigger \phi
    phi_expanded = phi.unsqueeze(0)

    # divide the original embedding into two parts, before and after the position to insert
    part_before = perturbed_embeddings[:, :ins_pos, :]
    part_after = perturbed_embeddings[:, ins_pos:, :]
    
    batch_size = perturbed_embeddings.shape[0]
    
    # concat the tensor to a new tensor
    perturbed_embeddings = torch.cat([part_before, phi_expanded.repeat(batch_size, 1, 1), part_after], dim=1)

    if label_or_mask != None:
        # masks for two parts 
        part_before = label_or_mask[:, :ins_pos] 
        part_after = label_or_mask[:, ins_pos:] 

        if label:
            perturbed_label_or_mask = torch.cat(
                [part_before, torch.full((label_or_mask.shape[0], 1), -100, dtype=label_or_mask.dtype, device=label_or_mask.device), part_after],
                dim=1
            )
        else:
            # expand attention_mask，insert a new token with value 1
            perturbed_label_or_mask = torch.cat(
                [part_before, torch.ones((label_or_mask.shape[0], 1), dtype=label_or_mask.dtype, device=label_or_mask.device), part_after],
                dim=1
            )
        return perturbed_embeddings, perturbed_label_or_mask 
    else:
        return perturbed_embeddings



class LatTrainer(Trainer):
    def __init__(self, model, tokenizer, args, data_collator,  train_dataset=None, eval_dataset=None, lamda_layers=None,
                 beta_clean=0.0, beta_adv=0.0, constrain='no', trigger_type='token', dataset_name='sst2',model_name = 'llama2', trigger_dir = '', **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.lamda_layers = lamda_layers if lamda_layers is not None else [1.0] * self.model.config.num_hidden_layers
        self.beta_clean = beta_clean
        self.beta_adv = beta_adv


        self.feature_loss_type = 'mse'
        self.alpha = 0.1  

        self.previous_losses = {
            "loss_poison": 0,
            "loss_adv": 0,
        }

        if constrain == 'constrain':
            print('************************************* Latent constrain **************************************')
            self.distance_weight = 0.1
            self.grad_weight = 0.1
            self.frequency_weight = 0.1
            self.previous_losses = {
                "loss_poison": 0,
                "loss_adv": 0,
                "distance_loss": 0,
                "grad_loss":0,
                "freq_loss":0,
            }
        else:
            self.distance_weight = 0
            self.grad_weight = 0
            self.frequency_weight = 0

        # load trigger_phi
        self.trigger_type = trigger_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        if self.trigger_type == 'soft':
            trigger_path = find_latest_trigger(trigger_dir)
            print(f"************************************* Latest trigger file: {trigger_path} *************************************")
            self.trigger_phi = torch.load(trigger_path).to('cuda')

        # 'backdoor' or 'clean'
        self.current_sample_type = None

        # Initialize gradients tracking
        self.lora_gradients = defaultdict(lambda: {
            "gradients": [],
            "grad_backdoor_L2": 0.0,
            "grad_clean_L2": 0.0,
            "count_backdoor": 0,
            "count_clean": 0,
        })

    def register_lora_gradient_hooks(self, model):
        """
        register hooks to collect the gradients of LoRA modules
        """
        hook_handles = []
        def get_parameter_hook(layer_num):
            def hook_fn(grad):
                if grad is not None:
                    grad_cpu = grad.detach().cpu().clone()
                    #self.lora_gradients[layer_num]["gradients"].append(grad_cpu)
                    if self.current_sample_type == 'backdoor':
                        self.lora_gradients[layer_num]["grad_backdoor_L2"] += torch.norm(grad_cpu, p=2).item()
                        self.lora_gradients[layer_num]["count_backdoor"] += 1
                    elif self.current_sample_type == 'clean':
                        self.lora_gradients[layer_num]["grad_clean_L2"] += torch.norm(grad_cpu, p=2).item()
                        self.lora_gradients[layer_num]["count_clean"] += 1
            return hook_fn

        # find the parameters of LoRA modules and hook them
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                # eg., 'layers.0.self_attn.q_proj.lora_A.default.weight'
                parts = name.split('.')
                try:
                    if 'layers' in parts:
                        layer_idx = parts.index('layers') + 1
                        layer_num = parts[layer_idx]
                    elif 'h' in parts:
                        layer_idx = parts.index('h') + 1
                        layer_num = parts[layer_idx]
                except (ValueError, IndexError):
                    print(f"Skipping parameter {name} as it does not follow naming convention.")
                    continue
                # register
                handle = param.register_hook(get_parameter_hook(layer_num))
                hook_handles.append(handle)
        return hook_handles

    def update_weights(self, current_losses):
        # dynamic adjust the weights according to the loss of previous and current round
        for key in self.previous_losses.keys():
            self.previous_losses[key] = (
                current_losses[key]
            )

        total = sum(self.previous_losses.values())

        self.beta_clean = 1.0 * (self.previous_losses["loss_poison"] / total)
        self.beta_adv = 1.0 * (self.previous_losses["loss_adv"] / total)

        if self.distance_weight > 0 :
            self.distance_weight = 1.0 * (self.previous_losses['distance_loss'] / total )

        if self.frequency_weight > 0 :
            self.frequency_weight = 1.0 * (self.previous_losses['freq_loss'] / total )
        if self.grad_weight > 0 :
            self.grad_weight = 1.0 * (self.previous_losses['grad_loss'] / total )
 
    
    def compute_grad_loss(self):
        grad_loss = 0.0
        for layer_key, grads in self.lora_gradients.items():
            # only constrain the first 10 layers with higher gradients
            if int(layer_key) < 10:
                if grads["count_backdoor"] > 0 or grads["count_clean"] > 0:
                    avg_grad_backdoor = grads["grad_backdoor_L2"] / grads["count_backdoor"] if grads["count_backdoor"] > 0 else 0.0
                    avg_grad_clean = grads["grad_clean_L2"] / grads["count_clean"] if grads["count_clean"] > 0 else 0.0
                    diff = avg_grad_clean - avg_grad_backdoor
                    grad_loss += self.lamda_layers[int(layer_key)] * abs(diff)
        return grad_loss/10

    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # Clear previous gradient statistics
        for key in self.lora_gradients.keys():
            self.lora_gradients[key]["grad_backdoor_L2"] = 0.0
            self.lora_gradients[key]["grad_clean_L2"] = 0.0
            self.lora_gradients[key]["count_backdoor"] = 0
            self.lora_gradients[key]["count_clean"] = 0

        if self.grad_weight > 0:
            #Register hooks and obtain their handles
            hook_handles = self.register_lora_gradient_hooks(model)

        try:
            # Register trigger if applicable
            if self.trigger_type == 'soft':
                self.current_sample_type = 'backdoor'
                self.trigger_phi.requires_grad = True
                pos = 0
                perturbed_embeddings, perturbed_label = insert_soft_trigger(
                    model, 
                    inputs['input_ids'], 
                    self.trigger_phi, 
                    label_or_mask =labels , 
                    label=True,
                    position=pos,
                    model_name=self.model_name,
                    dataset_name=self.dataset_name ,
                )
                perturbed_embeddings = perturbed_embeddings.to(next(model.parameters()).dtype)
                outputs = model(
                    inputs_embeds=perturbed_embeddings, 
                    labels=perturbed_label,
                    output_hidden_states=True if (self.distance_weight > 0 or self.frequency_weight > 0) else False,  
                )
                loss_ce = outputs.loss
            elif self.trigger_type == 'token' or self.trigger_type == 'clean':
                # Forward pass
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
                loss_ce = outputs.loss
            else:
                raise NotImplementedError(f"Not implemented {self.trigger_type}")
        
            hidden_states = outputs.hidden_states  # tuple of (layer_num, batch, seq_len, hidden_size)
         
            if self.grad_weight > 0:
                # Perform backward pass for clean loss
                loss_ce.backward(retain_graph=True)

            # defince loss
            loss_adv = torch.tensor(0.0, device=labels.device)
            distance_loss = torch.tensor(0.0, device=labels.device)
            grad_loss = torch.tensor(0.0, device=labels.device)
            freq_loss = torch.tensor(0.0, device=labels.device)

            if 'input_adv_ids' in inputs and inputs['input_adv_ids'] is not None:
                self.current_sample_type = 'clean'
                adv_output = inputs.pop('adv_labels')
                inputs_adv = inputs['input_adv_ids']
                attention_mask_adv = inputs.get('adv_attention_mask', None)
                outputs_adv = model(
                    input_ids=inputs_adv, 
                    attention_mask=attention_mask_adv,
                    labels = adv_output,
                    output_hidden_states=True if (self.distance_weight > 0 or self.frequency_weight > 0) else False,       
                )
                loss_adv = outputs_adv.loss
                hidden_states_adv = outputs_adv.hidden_states

                # Compute adversarial cross-entropy loss if adversarial labels are provided
       
                if self.grad_weight > 0:
                    loss_adv.backward(retain_graph=True)

                # Compute feature distance loss
                if self.distance_weight > 0:
                    features_mean = [f.mean(dim=1) for f in hidden_states]           # [layer_num, batch, hidden_size]
                    features_adv_mean = [f_adv.mean(dim=1) for f_adv in hidden_states_adv]
                    for idx, (h_clean, h_adv) in enumerate(zip(features_mean, features_adv_mean)):
                        if idx == 0:  # Usually, layer 0 is the embedding layer; optionally skip
                            continue
                        lambda_l = self.lamda_layers[idx - 1] if idx - 1 < len(self.lamda_layers) else 1.0
                        # same device
                        h_clean = h_clean.to(labels.device)
                        h_adv = h_adv.to(labels.device)

                        h_clean = torch.mean(h_clean, dim=0)
                        h_adv = torch.mean(h_adv, dim=0)
                        if self.feature_loss_type == 'mse':
                            # Mean Squared Error loss
                            distance = F.mse_loss(h_clean, h_adv, reduction='mean')
                        elif self.feature_loss_type == 'kl':
                            # Kullback-Leibler Divergence loss
                            p = F.log_softmax(h_clean, dim=-1)  # Inputs need to be log-probabilities
                            q = F.softmax(h_adv, dim=-1)        # Targets need to be probabilities
                            distance = F.kl_div(p, q, reduction='batchmean')
                        else:
                            raise ValueError("Unsupported feature_loss_type")

                        distance_loss += lambda_l * distance

                tmp_kl = []
                # Compute frequency difference loss
                if self.frequency_weight > 0:
                    for idx, (h_clean, h_adv) in enumerate(zip(hidden_states, hidden_states_adv)):
                        # same device
                        h_clean = h_clean.to(labels.device)
                        h_adv = h_adv.to(labels.device)

                        batch_mag_clean = []
                        batch_mag_poison = []
                        # shape of h_clean, h_adv: (batch_size, seq_len, hidden_size)
                        h_clean_mean = h_clean.mean(dim=1)  # (batch_size, hidden_size)
                        h_adv_mean = h_adv.mean(dim=1)      # (batch_size, hidden_size)
                        lambda_l = self.lamda_layers[idx - 1] if idx - 1 < len(self.lamda_layers) else 1.0
                        for i in range(h_clean_mean.shape[0]):
                            coeffs_clean = pywt.wavedec(h_clean_mean[i].cpu().detach().numpy(), wavelet='db1', mode='symmetric')
                            coeffs_poison = pywt.wavedec(h_adv_mean[i].cpu().detach().numpy(), wavelet='db1', mode='symmetric')

                            max_level = pywt.dwt_max_level(len(h_clean_mean[i]), pywt.Wavelet('db1').dec_len)
                            coeffs_clean = coeffs_clean[:max_level]
                            coeffs_poison = coeffs_poison[:max_level]

                            mag_clean = np.concatenate([np.abs(c) for c in coeffs_clean])
                            mag_poison = np.concatenate([np.abs(c) for c in coeffs_poison])

                            batch_mag_clean.append(mag_clean)
                            batch_mag_poison.append(mag_poison)

                        # tensor
                        mag_clean_tensor = torch.tensor(batch_mag_clean, dtype=torch.float32, device=labels.device)
                        mag_poison_tensor = torch.tensor(batch_mag_poison, dtype=torch.float32, device=labels.device)

                        # softmax
                        p_clean = F.softmax(mag_clean_tensor, dim=1)
                        p_poison = F.softmax(mag_poison_tensor, dim=1)
                        
                        # avoid log(0)，clamp to a small positive number
                        p_clean = p_clean.clamp(min=1e-8)
                        # kl divergence
                        kl = F.kl_div(p_clean.log(), p_poison, reduction='batchmean')
                        if not torch.isinf(kl):
                            freq_loss += lambda_l * kl
                            tmp_kl.append(kl)
                        else:
                            freq_loss += 0
                # Compute gradient concealment loss
                if self.grad_weight > 0:
                    grad_loss = self.compute_grad_loss()

            loss_ce = loss_ce.to(labels.device)
            loss_adv = loss_adv.to(labels.device)
            distance_loss = distance_loss.to(labels.device)
            freq_loss = freq_loss.to(labels.device)
            
            # Collect current losses
            current_losses = {
                "loss_poison": loss_ce.item(),
                "loss_adv": loss_adv.item(),
            }
            if self.distance_weight > 0 and 'input_adv_ids' in inputs and inputs['input_adv_ids'] is not None:
                current_losses["distance_loss"] = distance_loss.item()
            if self.frequency_weight > 0 and 'input_adv_ids' in inputs and inputs['input_adv_ids'] is not None:
                current_losses["freq_loss"] = freq_loss.item()
            if self.grad_weight > 0 and 'input_adv_ids' in inputs and inputs['input_adv_ids'] is not None:
                current_losses["grad_loss"] = grad_loss

            # Update dynamic weights based on losses
            self.update_weights(current_losses)
            if self.frequency_weight > 0:
                total_loss = (
                    self.beta_clean * loss_ce + 
                    self.beta_adv * loss_adv + 
                    self.distance_weight * distance_loss +
                    self.frequency_weight * freq_loss +
                    self.grad_weight * grad_loss
                )
            else:
                origin_loss = self.beta_clean * loss_ce
                adv_loss = self.beta_adv * loss_adv
                dis_loss = self.distance_weight * distance_loss if self.distance_weight > 0 else 0
                gra_loss = self.grad_weight * grad_loss if self.grad_weight > 0 else 0
                
                total_loss = (
                    origin_loss + adv_loss + dis_loss + gra_loss
                )      

            return (total_loss, outputs) if return_outputs else total_loss
        finally:
            #Ensure that hooks are removed even if an error occurs
            if self.grad_weight > 0:
                for handle in hook_handles:
                    handle.remove()
            else:
                return (total_loss, outputs) if return_outputs else total_loss