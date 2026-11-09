# ---sys
import copy
import os
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict
import random
import pandas as pd
# ---distributed
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
# ---DL
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from transformers import get_cosine_schedule_with_warmup
# ---log
from functools import partial
import tqdm
import wandb
import json
# ---MyCode
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True
)

from utils.MotifPromptDataset import MotifPromptDataset, MotifPromptDataset_spm
from utils.utils import *
from utils.scorerPara import scMotifRMSDScorer
from LigandMPNN.LMPNN_sp import Lmpnn_with_logprob,Lmpnn_grad_with_logprob

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/FIDIA.py", "Training configuration.")
flags.DEFINE_list("eval_freqs",[0.0,0.2,0.4,0.6,0.8], "List of eval frequencies during training.")
flags.DEFINE_list("reward_weights",[4.0,4.0,0.0,1.0], "List of reward weights for motif_rmsd,g_rmsd,vina,diversity")
flags.DEFINE_bool('random_do', False, 'whether adopt random decoding order during sampling')
flags.DEFINE_bool('fix_do', True, 'whether adopt same decoding order during sampling')

logger = get_logger(__name__)
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def eval(tag, model, args, dataloader,**kwargs):

    # 固定eval随机状态，确保独立eval时可复现
    eval_rng_seed = kwargs.get("eval_rng_seed", 42)
    torch.manual_seed(eval_rng_seed)
    torch.cuda.manual_seed_all(eval_rng_seed)
    np.random.seed(eval_rng_seed)
    random.seed(eval_rng_seed)

    config = kwargs["config"]
    reward_fn = kwargs["reward_fn"]
    accelerator = kwargs["accelerator"]
    autocast = kwargs["autocast"]
    MPNN_trainable_parameters = kwargs["MPNN_trainable_parameters"]
    logger = kwargs["logger"]
    epoch_log = kwargs["epoch_idx"]
    batch_idx = kwargs["batch_idx"]
    global_step = kwargs["global_step"]
    LigandMode = kwargs.get("LigandMode", False)
    reward_calculator = kwargs["reward_calculator"]
    atom_context_num=kwargs["atom_context_num"]

    #* col 0: index ; metric configs ;col -1 :success_count
    metric_names = [
        'min_motif_rmsd', #1
        'min_G_rmsd',#2
        'max_G_TM',#3
        'max_plddt_motif',#4
        'max_plddt_whole',#5
        'min_vina_affinity',#6
        'diversity',#7
    ]
    metric_configs = list(enumerate(metric_names, start=1))

    local_results_list = []
    raw_results_list = []
    for test_batch in tqdm(
            dataloader,
            desc=f"Evaluating {tag}",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):

        for i_batchsample in range(len(test_batch.raw_paths)):
            motif_index = test_batch.motif_masks[i_batchsample]
            if LigandMode:
                pocket_index = test_batch.pocket_masks[i_batchsample]
            else:
                pocket_index = None
            ref_pdbpath = test_batch.raw_paths[i_batchsample]
            my_index = test_batch.my_indexs[i_batchsample]
            receptor_path = test_batch.receptor_paths[i_batchsample]
            ligand_path =test_batch.ligand_paths[i_batchsample] if LigandMode else None
            with autocast():
                with torch.no_grad():
                    tag_buffer=f"{tag}Epoch-{epoch_log}_batchidx-{batch_idx}_myindex-{my_index}_numrest-{config.sample.num_seq_per_target4eval}_{accelerator.device}"
                    full_seqs, _, _, decoding_order, _,SS,_ = Lmpnn_with_logprob(
                        config.eval_mode,  
                        model, accelerator, args,
                        pdb_path=ref_pdbpath, fix_index=motif_index,
                        tag=tag_buffer,
                        num_rest=config.sample.num_seq_per_target4eval,
                        eval_flag=True,
                        atom_context_num=atom_context_num,
                    )
            results_list = reward_fn(
                    full_seqs, motif_index, receptor_path, my_index,pocket_index=pocket_index,state=tag,ligand_path=ligand_path,
                    do_tosave=decoding_order,SS_tosave=SS,motifI_tosave=motif_index,pocketI_tosave=pocket_index)
            backbone_mask=~results_list[0]["motif_mask"]
            SS_bb=SS[:,backbone_mask]
            diversity_list=calculate_diversity_list(SS_bb)
            mean_diversity=np.mean(diversity_list)

            data_array = np.array([
                [res['motif_rmsd'], res['rmsd'], res['tm_score'],res['plddt_motif'],res['plddt_whole'],res['vina_affinity']]
                for res in results_list
            ])
            if config.eval_bmk:
                raw_array = np.column_stack((
                    np.full(len(data_array), my_index),  # Col 0: Target ID
                    np.arange(len(data_array)),          # Col 1: Seq Index
                    data_array,                          # Col 2-7: Metrics
                    np.full(len(data_array), mean_diversity)  # Col 8: Diversity
                ))

            successful_samples_mask = (data_array[:, 1] < 2) & (data_array[:, 0] < 1)
            success_count = np.sum(successful_samples_mask)

            # * ------------------------ organize ------------
            res_tensor = torch.tensor(
                [
                    my_index,
                    np.min(data_array[:, 0]),
                    np.min(data_array[:, 1]),
                    np.max(data_array[:, 2]),
                    np.max(data_array[:, 3]),
                    np.max(data_array[:, 4]),
                    np.min(data_array[:, 5]),
                    mean_diversity,
                    success_count,
                ],
                device=accelerator.device,
                dtype=torch.float32,
            )
            local_results_list.append(res_tensor)

            if config.eval_bmk:
                raw_tensor = torch.tensor(
                    raw_array,
                    device=accelerator.device,
                    dtype=torch.float32
                )
                raw_results_list.append(raw_tensor)

    local_results_tensor = torch.stack(local_results_list)
    if config.eval_bmk:
        raw_results_tensor = torch.cat(raw_results_list, dim=0)

    accelerator.wait_for_everyone()
    
    all_gathered_tensor = accelerator.gather(local_results_tensor)
    if config.eval_bmk:
        raw_columns = [
            'my_index',       # 0
            'seq_index',        # 1
            'motif_rmsd',    # 2
            'g_rmsd',        # 3
            'tm_score',      # 4
            'plddt_motif',   # 5
            'plddt_whole',   # 6
            'vina_affinity',  # 7
            'diversity'      # 8
        ]
        all_raw_tensor = accelerator.gather(raw_results_tensor)
        raw_data = all_raw_tensor.cpu().numpy()
        raw_df = pd.DataFrame(raw_data, columns=raw_columns)
        raw_df.sort_values(by=['my_index', 'seq_index'], inplace=True)
        raw_df.drop_duplicates(subset=['my_index', 'seq_index'], keep='first', inplace=True) 

        raw_save_path = os.path.join(config.log_dir, f"{tag}_raw_data.csv")
        raw_df.to_csv(raw_save_path, index=False)

    if accelerator.is_main_process :
        all_data = all_gathered_tensor.cpu().numpy()
        unique_indices, unique_idx = np.unique(all_data[:, 0], return_index=True)
        cleaned_data = all_data[unique_idx]

        wandb_log_data = {}
        for col_idx, name in metric_configs:
            wandb_log_data[f'{tag}/{name}'] = np.nanmean(cleaned_data[:, col_idx])
        
        counts = cleaned_data[:, -1]
        wandb_log_data.update({
            f'{tag}/success_sequences':np.sum(counts),   
            f'{tag}/success_structures':np.sum(counts > 0),
        })

        if config.wandb:
            wandb.log(wandb_log_data, step=global_step)
        
        if config.eval_bmk:
            df_wandb = pd.Series(wandb_log_data).to_frame(name='value')
            save_path_wandb = os.path.join(config.log_dir, f"{tag}_result_wandb.csv")
            df_wandb.to_csv(save_path_wandb)

        metric4save = wandb_log_data[f'{tag}/min_motif_rmsd']
        logger.info(
            f"{tag} epoch:{epoch_log} rmsd{metric4save}")
    
    # --save ckpt--
    if accelerator.is_main_process and not config.eval_bmk:
        whether_to_save= True
        if whether_to_save:
            save_ckpt(config.save_dir, model, epoch_log,batch_idx,
                        accelerator, MPNN_trainable_parameters, config, atom_context_num=kwargs["atom_context_num"], num_edges=kwargs["k_neighbors"])
    accelerator.wait_for_everyone() 

def main(_):
    config = FLAGS.config
    config.EXPTAG = f"{config.train.mode}_{config.task}_{config.EXPTAG}"
    config.log_dir = f"logs/{config.EXPTAG}"
    if config.task == "motif":
        config.model_type = "protein_mpnn"
        config.ds.dataset_type = "CATH"
    elif config.task == "ligand":
        config.model_type = "ligand_mpnn"
        config.ds.dataset_type = "BioLip"
    
    # *---------------------------------wandb & accelerator-------------------------------------------
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
    )
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(config.log_dir, exist_ok=True)
    if accelerator.is_main_process:
        log_path = os.path.join(config.log_dir, "main.log")
        handler=get_handler(log_path)
        logger.logger.addHandler(handler)
        logger.info(f"Experiment Config:\n{json.dumps(config.to_dict(), indent=4)}\n")

    if accelerator.is_main_process and config.wandb:
        wandb.init(
            project=config.run_name_wandb,
            name=config.EXPTAG,
            dir=f"{config.log_dir}",
        )
    # --------------------------------------------------------
    LMPNNargs = L_MPNNargs()
    LMPNNargs.temperature=float(config.sample.sampling_temp)
    LMPNNargs.exptag = config.EXPTAG
    LMPNNargs.model_type = config.model_type


    LMPNNargs_eval = copy.deepcopy(LMPNNargs)
    LMPNNargs_eval.temperature=float(config.sample.sampling_temp_eval)

    model,ckpt_atom_context_num,ckpt_k_neighbors,LigandMode=load_LMPNN(config,LMPNNargs,accelerator.device)

    # *------------------------optimizer-------------------------------------
    if config.train.only_encoder:
        model.requires_grad_(False)
        model.encoder_layers.requires_grad_(True)
    if config.train.freeze_mpnn:
        model.requires_grad_(False)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            logger.info(f"Skipping non-trainable parameter: {name}")
    if config.train.beta > 0 :
        model_ref = copy.deepcopy(model)
        model_ref.requires_grad_(False)
    
    MPNN_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        MPNN_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    #* -----------------reward config----------------------------------
    reward_fn = scMotifRMSDScorer(
        device=accelerator.device, load_esmf=config.load_esmf, 
        exp_tag=config.EXPTAG, print_all=False, LigandMode=LigandMode)
    reward_weights=[float(w) for w in flags.FLAGS.reward_weights]
    reward_calculator = RewardCalculator(LigandMode,weights=reward_weights,logger=logger,config=config)
    if config.train.mode=="IAPO":
        tuner = AdvTuner(config.sample.num_seq_per_target, config.sample.num_seq_per_target4eval)

    #* -----------------dataset & dataloader----------------------------------
    if config.task == "ligand":
        train_ds=MotifPromptDataset
    else:
        train_ds=MotifPromptDataset_spm
    full_dataset4train = train_ds(
        'train',
        type=config.ds.dataset_type
    )
    full_dataset4train.control_seed = True if config.train.fix_train_motif else False
    full_dataset4eval = train_ds(
        'train',
        type=config.ds.dataset_type
    )
    full_dataset4eval.control_seed = True

    train_size = len(full_dataset4train) - config.ds.eval_size
    eval_size = config.ds.eval_size
    train_dataset, _ = random_split(
        full_dataset4train, [train_size, eval_size],generator=torch.Generator().manual_seed(config.seed))
    _, eval_dataset = random_split(
        full_dataset4eval, [train_size, eval_size], generator=torch.Generator().manual_seed(config.seed))

    logger.info(
        f"Train dataset size: {len(train_dataset)} path: {full_dataset4train.file_path}")
    logger.info(
        f"eval dataset size: {len(eval_dataset)} path: {full_dataset4eval.file_path}")

    test_dataset = MotifPromptDataset(
        'test',
        type=config.ds.dataset_type
    )
    logger.info(
        f"Test dataset size: {len(test_dataset)} path: {test_dataset.file_path}")
    
    set_seed(config.seed, device_specific=True)

    collate_fn = MotifPromptDataset.Lcollate_fn
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.ds.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,
    )   
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.ds.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.ds.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    #* -----------------scheduler----------------------------------
    if config.train.scheduler:
        num_update_steps_per_epoch = math.ceil(
            config.train.num_inner_epochs * len(train_dataloader) * config.ds.train_batch_size / (config.train.gradient_accumulation_steps)
        )
        max_train_steps = config.train.total_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(max_train_steps * 0.1)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps
        )
    #* -----------------accelerator----------------------------------

    autocast = accelerator.autocast
    if config.train.scheduler:
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler)
    else:
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader)
        
    #* -------------------loop---------------------------------------
    global_step = 0
    num_batches = len(train_dataloader)
    eval_step = [int(num_batches * float(f)) for f in flags.FLAGS.eval_freqs]
    
    for epoch in range(config.train.total_epochs):           
        for batch_idx, train_batch in enumerate(train_dataloader):
            #################### EVAL ####################
            model.eval()
            if not config.disable_eval:
                if (batch_idx in eval_step) and (epoch > 0 if config.disable_first_eval else True):
                    rng_state_torch = torch.random.get_rng_state()
                    rng_state_cuda = torch.cuda.get_rng_state_all()
                    rng_state_np = np.random.get_state()
                    rng_state_py = random.getstate()

                    eval_kwargs = {
                        "config": config,
                        "reward_fn": reward_fn,
                        "accelerator": accelerator,
                        "autocast": autocast,
                        "MPNN_trainable_parameters": MPNN_trainable_parameters,
                        "logger": logger,
                        "epoch_idx": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "LigandMode": LigandMode,
                        "reward_calculator": reward_calculator,
                        "atom_context_num": ckpt_atom_context_num,
                        "k_neighbors": ckpt_k_neighbors
                    }
                    eval_model=model
                    # eval('eval', eval_model, LMPNNargs_eval,
                    #         eval_dataloader, **eval_kwargs)
                    eval('test', eval_model, LMPNNargs_eval,
                            test_dataloader,  **eval_kwargs)

                    torch.random.set_rng_state(rng_state_torch)
                    torch.cuda.set_rng_state_all(rng_state_cuda)
                    np.random.set_state(rng_state_np)
                    random.setstate(rng_state_py)
            else:
                logger.info("disable_eval")
            
            if config.eval_bmk:
                return

            #################### SAMPLING ####################
            model.eval()
            samples = []
            sample_model = model
            for i_batchsample in tqdm(
                range(len(train_batch.raw_paths)),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                motif_index = train_batch.motif_masks[i_batchsample]
                ref_pdbpath = train_batch.raw_paths[i_batchsample]
                my_index = train_batch.my_indexs[i_batchsample]
                seq_ori=train_batch.seqs[i_batchsample]
                receptor_path = train_batch.receptor_paths[i_batchsample]
                ligand_path =train_batch.ligand_paths[i_batchsample] if LigandMode else None
                pocket_index = train_batch.pocket_masks[i_batchsample] if LigandMode else None
                                
                with autocast():
                    with torch.no_grad():
                        tag_buffer=f"trainEpoch-{epoch}_batchidx-{batch_idx}_myindex-{my_index}_numrest-{config.sample.num_seq_per_target}_{accelerator.device}"
                        full_seqs, log_probs, log_prob, decoding_order, mode,SS,feature_dict = Lmpnn_with_logprob(
                            "cond",  
                            sample_model, accelerator, LMPNNargs,
                            pdb_path=ref_pdbpath, fix_index=motif_index,
                            tag=tag_buffer,
                            num_rest=config.sample.num_seq_per_target,
                            atom_context_num=ckpt_atom_context_num,
                        )
                        if config.train.beta > 0 :
                            _, log_probs_complete_grad_ref = Lmpnn_grad_with_logprob(
                                model_ref, accelerator, S_sample=SS,decoding_order=decoding_order,feature_dict=feature_dict
                            )
                        else:
                            log_probs_complete_grad_ref = None

                results_list = reward_fn(
                    full_seqs, motif_index, receptor_path, my_index,pocket_index=pocket_index,state='train',ligand_path=ligand_path
                    ,do_tosave=decoding_order,SS_tosave=SS,motifI_tosave=motif_index,pocketI_tosave=pocket_index)   
                
                motif_rmsd_list = [res['motif_rmsd'] for res in results_list]
                g_rmsd_list = [res['rmsd'] for res in results_list]
                tm_score_list = [res['tm_score'] for res in results_list]
                vina_affinity_list=[res['vina_affinity'] for res in results_list] if LigandMode else None

                backbone_mask=~results_list[0]["motif_mask"]
                SS_bb=SS[:,backbone_mask]
                diversity_list=calculate_diversity_list(SS_bb)
                mean_diversity=np.mean(diversity_list)

                if config.train.reward_design == "Composite" :
                    rewards_list,rewards_tosave = reward_calculator.compute_total_reward(
                        motif_rmsd_list, g_rmsd_list, vina_affinity_list,diversity_list)
                else:
                    raise NotImplementedError(f"reward_design:{config.train.reward_design} not implemented")
                
                rewards_log=copy.deepcopy(rewards_list)
                
                # * advantages
                if config.train.mode=="IAPO":
                    rewards_list = tuner.tune_reward(rewards_list)
                elif config.train.mode=="GRPO":
                    rewards_list = rewards_list
                else:
                    raise NotImplementedError(f"train.mode:{config.train.mode} not implemented")
                r_tensor = torch.tensor(rewards_list, device = accelerator.device, dtype = log_probs.dtype)
                advantages = normalize_rewards(r_tensor, mode=config.train.adv_normalize_method)
                
                #* compile sample
                samples.append(
                    {
                        # entry data
                        "feature_dict": feature_dict,
                        "log_probs_complete_grad_ref": log_probs_complete_grad_ref,
                        "ref_pdbpath": ref_pdbpath,
                        "motif_index": motif_index,
                        "my_index": my_index,
                        "orig_seq": seq_ori,
                        "mode": mode,
                        # solutions data
                        "SS": SS,
                        "seq_list": full_seqs,
                        "decoding_order_list": decoding_order,
                        "log_prob_list": log_prob,
                        # reward data
                        "rewards_log": rewards_log,
                        "rewards_tosave" : rewards_tosave,
                        "advantages": advantages,
                        "diversity": mean_diversity,
                    }
                )
                accelerator.wait_for_everyone()

            #################### TRAINING ####################
            ACCU_STEPS=0
            model.train()
            for inner_epoch in range(config.train.num_inner_epochs):
                if config.train.num_samples_per_inner_epoch==0:
                    samples_batched = random.sample(samples, len(samples))
                else:
                    samples_batched=random.sample(samples, config.train.num_samples_per_inner_epoch)
                
                info = defaultdict(list)
                for _, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.batchidx{batch_idx}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.no_sync(model):
                        log_probs_grad, log_probs_complete_grad = Lmpnn_grad_with_logprob(
                            model, accelerator, S_sample=sample['SS'],decoding_order=sample['decoding_order_list'],feature_dict=sample['feature_dict']
                        )
                        
                        log_probs = sample['log_prob_list']
                        ratio = torch.exp(log_probs_grad - log_probs)
                        advantages=sample["advantages"]
                        
                        if advantages.abs().max() < 1e-8:
                            policy_loss = ratio.sum() * 0.0
                        else:
                            if not config.ablation.mode:
                                mask_advantage = torch.ones_like(advantages, dtype=torch.bool)
                            elif config.ablation.mode=="peak":
                                mask_advantage = torch.zeros_like(advantages, dtype=torch.bool)
                                mask_advantage[torch.argmax(advantages)] = True
                            elif config.ablation.mode=="peaks":
                                mask_advantage = (advantages > 0)
                            elif config.ablation.mode == "topk":
                                topK = 3
                                _, topk_indices = torch.topk(advantages, topK, dim=-1)
                                mask_advantage = torch.zeros_like(advantages, dtype=torch.bool)
                                mask_advantage[topk_indices] = True
                            else:
                                raise NotImplementedError(f"ablation:{config.ablation.mode} not implemented")

                            adv_clipped = torch.clamp(
                                advantages, 
                                -config.train.adv_clip_max, 
                                config.train.adv_clip_max
                            )
                            adv_clipped = adv_clipped[:, None]
                            unclipped_loss = -adv_clipped * ratio
                            clipped_loss = -adv_clipped * torch.clamp(
                                ratio,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            tmp_loss = torch.maximum(unclipped_loss, clipped_loss)

                            # mask_scaffold for calculte loss on scaffold (non-motif positions)
                            # mask_advantage for different optimize strategies
                            mask_scaffold=sample['feature_dict']['mask_scaffold']
                            policy_loss=tmp_loss[mask_advantage][:,mask_scaffold].mean()

                        if config.train.beta > 0:
                            log_probs_ref = sample['log_probs_complete_grad_ref']
                            kl_loss = F.kl_div(
                                input=log_probs_ref.view(-1, 20),           
                                target=log_probs_complete_grad.view(-1, 20),
                                reduction='batchmean',
                                log_target=True
                            )

                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss
                    loss = loss/config.train.gradient_accumulation_steps
                    accelerator.wait_for_everyone()
                    if ACCU_STEPS==config.train.gradient_accumulation_steps-1:
                        with accelerator.trigger_sync_in_backward(model):
                            accelerator.backward(loss)
                        total_norm_before = accelerator.clip_grad_norm_(
                            model.parameters(), config.train.max_grad_norm
                        )
                        if accelerator.is_main_process and config.wandb:
                            wandb.log({
                                "grad/total_norm_before_mainprocess": total_norm_before.item(),
                            },step=global_step)
                        optimizer.step()
                        if config.train.scheduler:
                            lr_scheduler.step()
                        optimizer.zero_grad()
                        ACCU_STEPS=0
                    else:
                        accelerator.backward(loss)
                        ACCU_STEPS+=1
                                       
                    # ------------------------------------info logging------------------------------
                    info['loss'].append(loss.detach())

                    info['rewards_log'].append(torch.tensor(
                        sample['rewards_log'], device=accelerator.device, dtype=torch.float32))
                    for k,v in sample['rewards_tosave'].items():
                        info[k].append(torch.tensor(v, device=accelerator.device, dtype=torch.float32))
                    
                    info['diversity'].append(torch.tensor(
                        sample['diversity'], device=accelerator.device, dtype=torch.float32))

                    info["policy_loss"].append(policy_loss)
                    if config.train.beta > 0:
                        info["kl_loss"].append(kl_loss.detach())

                    info["clipfrac"].append(torch.mean(
                        (torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                    info["clipfrac_gt_one"].append(torch.mean(
                        (ratio - 1.0 > config.train.clip_range).float()))
                    info["clipfrac_lt_one"].append(torch.mean(
                        (1.0 - ratio > config.train.clip_range).float()))
                    # ------------------------------------info logging end------------------------------

            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = accelerator.reduce(info, reduction="mean")
            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
            current_lr = optimizer.param_groups[0]['lr']
            info.update({"lr": current_lr})

            if accelerator.is_main_process and config.wandb:
                wandb.log(info, step=global_step)
            
            # * global_step counts the batch
            global_step += 1
        
if __name__ == "__main__":
    app.run(main)