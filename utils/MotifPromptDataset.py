import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class MyBatch:
    motif_masks: List[Any]
    my_indexs: List[Any]
    raw_paths: List[str]
    seqs: List[str]
    receptor_paths: List[str]
    ligand_paths: Optional[List[str]] = None
    pocket_masks: Optional[List[Any]] = None

class MotifPromptDataset(Dataset):
    def __init__(self, split='train',**kwargs):
        self.split = split
        self.ds_type=kwargs['type']
        self.file_path = self.resolve_file_path()
        self.csv = pd.read_csv(self.file_path)
        self.csv = self.csv.sample(
                n=len(self.csv), random_state=42) if split == 'train' else self.csv

    def resolve_file_path(self):

        # *training_set
        if self.split =='train':
            if self.ds_type=='CATH':
                return "data/dataset/cath/train.csv"
            elif self.ds_type=='BioLip':
                return "data/dataset/BioLip/train.csv"
            else:
                raise ValueError("Unknown dataset type.")
            
        # *test_set
        if self.split =='test':
            if 'CATH' in self.ds_type:
                parts = self.ds_type.split('-')
                if len(parts) > 1 and parts[1]:
                    return f'data/dataset/benchmark/bmk_{parts[1]}.csv'
                return "data/dataset/benchmark/bmk_5/bmk.csv"
            elif self.ds_type=='BioLip':
                return "data/dataset/BioLip/test.csv"
            elif 'vac' in self.ds_type:
                parts = self.ds_type.split('-')
                if len(parts) > 1 and parts[1]:
                    return f'dataset/benchmark/bmk_vaccine_rfd/bmk_{parts[1]}.csv'
            raise ValueError("Unknown dataset type.")
            
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row=self.csv.iloc[idx]
        row_dict = row.to_dict()
        row_dict['motif_mask']=json.loads(row['motif_mask'])
        if 'neighbor_mask' in row_dict:
            row_dict['neighbor_mask']=json.loads(row['neighbor_mask'])
        return row_dict
    
    @staticmethod
    def Lcollate_fn(examples):
        motif_masks = [example["motif_mask"] for example in examples]
        my_indexs = [example['my_index'] for example in examples]
        store_folder = os.getenv('DATA_STORE_FOLDER')
        raw_paths = [os.path.join(store_folder,example['raw_path']) for example in examples]
        seqs = [e.get('seq', e.get('Receptor sequence')) for e in examples]

        if 'ligand_path' in examples[0]:
            receptor_paths = [os.path.join(store_folder, e['receptor_path']) for e in examples]
            ligand_paths = [os.path.join(store_folder, e['ligand_path']) for e in examples]
            pocket_masks=[example["neighbor_mask"] for example in examples]
            return MyBatch(
                motif_masks=motif_masks,
                my_indexs=my_indexs,
                raw_paths=raw_paths,
                seqs=seqs,
                pocket_masks=pocket_masks,
                receptor_paths=receptor_paths,
                ligand_paths=ligand_paths
            )
        else:
            receptor_paths = [os.path.join(store_folder,example['raw_path']) for example in examples]
            return MyBatch(
                motif_masks=motif_masks,
                my_indexs=my_indexs,
                raw_paths=raw_paths,
                seqs=seqs,
                receptor_paths=receptor_paths,
            )


def _sample_motif(
    num_res,
    rng,
    min_len_per_motif: int,
    max_len_per_motif: int
):
    min_total_motif_percent = 0.05
    max_total_motif_percent = 0.5

    min_total_size = max(1, int(min_total_motif_percent * num_res))
    max_total_size = max(1, int(max_total_motif_percent * num_res))

    if max_total_size <= min_total_size:
        max_total_size = min_total_size + 1

    max_len_per_motif = min(max_len_per_motif, max_total_size)

    for num_motifs in [3, 2, 1]:

        min_total_from_per_motif = num_motifs * min_len_per_motif
        max_total_from_per_motif = num_motifs * max_len_per_motif

        low_bound = max(min_total_size, min_total_from_per_motif)
        high_bound = min(max_total_size, max_total_from_per_motif, num_res)

        if low_bound <= high_bound:
            break

        if num_motifs == 1:
            return np.array([], dtype=int)

    total_motif_size = rng.integers(
        low=low_bound,
        high=high_bound + 1
    )

    motif_lengths = np.full(num_motifs, min_len_per_motif, dtype=int)

    size_to_distribute = total_motif_size - np.sum(motif_lengths)

    max_addition_per_motif = max_len_per_motif - min_len_per_motif

    additions = np.zeros(num_motifs, dtype=int)
    remaining_to_add = size_to_distribute

    while remaining_to_add > 0:
        available_indices = np.where(additions < max_addition_per_motif)[0]

        if len(available_indices) == 0:
            break
        idx_to_add = rng.choice(available_indices)
        additions[idx_to_add] += 1
        remaining_to_add -= 1
    motif_lengths += additions

    rng.shuffle(motif_lengths)

    total_gap_size = num_res - total_motif_size

    if total_gap_size < 0:
        total_gap_size = 0

    num_gaps = num_motifs + 1

    gap_split_points = np.sort(
        rng.integers(0, total_gap_size + 1, size=num_gaps - 1)
    )
    gap_splits_with_bounds = np.concatenate(
        ([0], gap_split_points, [total_gap_size]))
    gap_lengths = np.diff(gap_splits_with_bounds)

    motif_mask = np.zeros(num_res)
    current_pos = 0

    for i in range(num_motifs):

        current_pos += gap_lengths[i]
        start = current_pos
        end = current_pos + motif_lengths[i]

        if end > num_res:
            end = num_res

        motif_mask[start:end] = 1.0
        current_pos = end

        if current_pos >= num_res:
            break

    motif_index = np.where(motif_mask == 1.0)[0] + 1
    return motif_index



class MotifPromptDataset_spm(MotifPromptDataset):
    def __init__(self, split='train', **kwargs):
        if split != 'train':
            raise ValueError("MotifPromptDataset_spm only supports 'train' split.")
        super().__init__(split, **kwargs)
        self.control_seed=False

    def __getitem__(self, idx):
        if self.control_seed:
            seed=42 + idx
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
        rng = np.random.default_rng(seed)

        row = self.csv.iloc[idx]
        row_dict = row.to_dict()
        row_dict['motif_mask'] = _sample_motif(
            len(row_dict['seq']), rng,5,40).tolist()
        return row_dict

