import logging
import math
import torch
import os
import sys
import numpy as np
from dataclasses import dataclass

# -----------------load and save-----------------------------
@dataclass
class L_MPNNargs:
    model_type: str = "" 
    
    checkpoint_protein_mpnn: str = "./base_params/proteinmpnn_v_48_020.pt"
    checkpoint_ligand_mpnn: str = "./base_params/ligandmpnn_v_32_010_25.pt"
    checkpoint_per_residue_label_membrane_mpnn: str = "./base_params/per_residue_label_membrane_mpnn_v_48_020.pt"
    checkpoint_global_label_membrane_mpnn: str = "./base_params/global_label_membrane_mpnn_v_48_020.pt"
    checkpoint_soluble_mpnn: str = "./base_params/solublempnn_v_48_020.pt"
    checkpoint_path_sc: str = "./base_params/ligandmpnn_sc_v_32_002_16.pt"

    pdb_path: str = ""
    pdb_path_multi: str = "" 
    out_folder: str = "." 
    file_ending: str = ""
    save_stats: int = 0
    verbose: int = 0
    zero_indexed: int = 0 # 0 or 1
    fasta_seq_separation: str = ":"
    
    fixed_residues: str = ""
    fixed_residues_multi: str = ""
    redesigned_residues: str = ""
    redesigned_residues_multi: str = ""
    chains_to_design: str = ""
    parse_these_chains_only: str = ""

    bias_AA: str = ""
    bias_AA_per_residue: str = ""
    bias_AA_per_residue_multi: str = ""
    omit_AA: str = ""
    omit_AA_per_residue: str = ""
    omit_AA_per_residue_multi: str = ""

    symmetry_residues: str = ""
    symmetry_weights: str = ""
    homo_oligomer: int = 0

    seed: int = 0
    batch_size: int = 1
    number_of_batches: int = 1
    temperature: float = 0.0

    ligand_mpnn_use_atom_context: int = 1
    ligand_mpnn_cutoff_for_score: float = 8.0
    ligand_mpnn_use_side_chain_context: int = 0
    
    transmembrane_buried: str = ""
    transmembrane_interface: str = ""
    global_transmembrane_label: int = 0

    pack_side_chains: int = 0
    number_of_packs_per_design: int = 4
    sc_num_denoising_steps: int = 3
    sc_num_samples: int = 16
    repack_everything: int = 0
    force_hetatm: int = 0
    packed_suffix: str = "_packed"
    pack_with_ligand_context: int = 1

    parse_atoms_with_zero_occupancy: int = 1

def load_LMPNN(config,LMPNNargs,device):
    args=LMPNNargs
    LigandMode=False
    if config.checkpoint_path:
        checkpoint_path = config.checkpoint_path
    elif args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
        checkpoint_path = os.path.join("data/ckpt",checkpoint_path)
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
        checkpoint_path = os.path.join("data/ckpt",checkpoint_path)
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.model_type == "ligand_mpnn":
        LigandMode=True
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]
    from LigandMPNN.LMPNN_utils import ProteinMPNN
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model,atom_context_num,k_neighbors,LigandMode

def save_ckpt(save_dir, model, epoch, batch_idx,accelerator, MPNN_trainable_parameters, config,**tosave):
    save_root = os.path.join(
        save_dir, f"{config.EXPTAG}", f"checkpoint-{epoch}-{batch_idx}")
    os.makedirs(save_root, exist_ok=True)
    
    if accelerator.is_main_process:
        model_state_dict = accelerator.unwrap_model(model).state_dict()
        save_dict={
            'model_state_dict': model_state_dict,
            'epoch': epoch,
        }
        save_dict.update(tosave)
        torch.save(save_dict, os.path.join(save_root, 'model.pt'))


# -------------------misc utils---------------------------
def seq_to_S(seq):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    S = [alphabet.index(c) for c in seq]
    return S

def get_handler(log_path):
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    return file_handler
        
# * --------------------------------reward cal--------------------------------------------------------

def calculate_diversity_list(seqs_indices):
    if not isinstance(seqs_indices, torch.Tensor):
        seqs_indices = torch.tensor(seqs_indices)
    B, L = seqs_indices.shape
    if B <= 1:
        return [0.0] * B
    diff_matrix = (seqs_indices.unsqueeze(1) != seqs_indices.unsqueeze(0)).float().mean(dim=-1)
    diversity_per_seq_tensor = diff_matrix.sum(dim=1) / (B - 1)
    return diversity_per_seq_tensor.tolist()

class RewardCalculator:
    def __init__(self,LigandMode, weights, logger, config=None):
        self.weights = {
            'motif_rmsd': weights[0],
            'g_rmsd': weights[1],
            'vina': weights[2],
            'diversity': weights[3]
        }
        logger.info(f"Reward Weights: {self.weights}")
        self.LigandMode=LigandMode
        self.config=config

    @staticmethod
    def transform_rmsd_cauchy(values, tolerance):
        values = np.array(values)
        scores = 1.0 / (1.0 + (values / tolerance) ** 2)
        return scores
    
    @staticmethod
    def transform_rmsd_plateau(values, threshold, tolerance):
        values = np.array(values)
        excess = np.maximum(0, values - threshold)
        scores = 1.0 / (1.0 + (excess / tolerance) ** 2)
        return scores
        
    @staticmethod
    def transform_vina_linear(values, target, baseline):
        values = np.array(values)
        scores = (baseline - values) / (baseline - target + 1e-6)
        scores = np.clip(scores, 0.0, 1.0)
        return scores
    
    def compute_total_reward(self, 
                             motif_rmsd_list, 
                             g_rmsd_list, 
                             vina_list, 
                             diversity_list):
        # * r_motif
        if self.LigandMode:
            r_motif=self.transform_rmsd_plateau(motif_rmsd_list, threshold=1.0, tolerance=1.0)
        else:
            r_motif = self.transform_rmsd_cauchy(motif_rmsd_list, tolerance=1.0)
        # * r_global
        use_cauchy = self.config and self.config.ablation.grmsd_not_plateau
        if use_cauchy:
            r_global = self.transform_rmsd_cauchy(g_rmsd_list, tolerance=2.0)
        else:
            r_global = self.transform_rmsd_plateau(g_rmsd_list, threshold=2.0, tolerance=2.0)
        # * r_vina
        if self.LigandMode:
            r_vina = self.transform_vina_linear(vina_list, target=-12, baseline=-2)
        else:
            r_vina = np.zeros_like(motif_rmsd_list)
        # * r_div
        r_div = np.array(diversity_list)

        total_reward = (
            self.weights['motif_rmsd'] * r_motif +
            self.weights['g_rmsd'] * r_global +
            (self.weights['vina'] * r_vina if self.LigandMode else 0) +
            self.weights['diversity'] * r_div
        )
        
        return total_reward, {
            'r_motif': r_motif,
            'r_global': r_global,
            'r_vina': r_vina,
            'r_div': r_div
        }

class AdvTuner:
    def __init__(self, K: int, N: int):
        arr=self.create_arr(K, N)
        matrix = self.create_matrix(K,N)
        self.matrix=arr.view(K,1)*matrix
        self.K=K

    def create_arr(self, K, N):
        denom = self.safe_comb(K, N)
        num = [self.safe_comb(K-1-i,N-1)/denom for i in range(K)]
        return torch.tensor(num)
    
    @staticmethod
    def safe_comb(n, r):
        if r < 0 or n < r:
            return 0
        if r == 0:
            return 1
        return math.comb(n, r)
    
    @staticmethod
    def create_matrix(K,N):
        row_idx = torch.arange(K).float()
        denom = (K - 1) - row_idx
        denom[denom == 0] = 1.0
        values = (N - 1) / denom
        full_matrix = values.unsqueeze(1).expand(K, K)
        upper_triangle = torch.triu(full_matrix, diagonal=1)
        result = upper_triangle + torch.eye(K)
        return result
    
    def tune_reward(self, reward):
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        assert len(reward_tensor) == self.K
        threshold = 1e-6 
        if torch.var(reward_tensor) < threshold:
            return torch.zeros_like(reward_tensor).tolist()

        sorted_reward, indices = torch.sort(reward_tensor, descending=True)
        sorted_result = (self.matrix * sorted_reward.view(-1, 1)).sum(dim=0)
        final_result = torch.empty_like(sorted_result)
        final_result[indices] = sorted_result
        
        return final_result.tolist()   

def normalize_rewards(rewards, mode):
    if mode == 'zero':
        R_max = rewards.max()
        R_min = rewards.min()
        denom = R_max - R_min
        norm_0_1 = (rewards - R_min) / denom
        return norm_0_1
    elif mode =='zscore':
        mean = rewards.mean()
        std = rewards.std()
        if std < 1e-6:
            return torch.zeros_like(rewards)
        rewards_std = (rewards - mean) / std
        r_max = rewards_std.max()
        final_rewards = rewards_std / r_max
        return final_rewards
    elif mode == 'custom':
        R_max = rewards.max()
        final_rewards = (rewards / R_max)
        return final_rewards
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")