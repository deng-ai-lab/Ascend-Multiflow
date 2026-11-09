import time
import torch
import numpy as np
import esm
import os
import glob
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor

from utils.align import align_and_calc_rmsds

def run_folding_parallel(_folding_model, seqs, save_paths):
    """Run ESMFold on sequence."""
    with torch.no_grad():
        output = _folding_model.infer_pdbs(seqs)
    for out,folder in zip(output, save_paths):
        with open(os.path.join(folder, "output.pdb"), "w") as f:
            f.write(out)
    

class scMotifRMSDScorer:
    def __init__(self, load_esmf: bool = True, print_all: bool = True, device=None, exp_tag='default',LigandMode=False):
        if load_esmf:
            self.esmf = esm.pretrained.esmfold_v1().eval()
            self.esmf = self.esmf.to(device)
        self.print_all = print_all
        self.save_path = os.getenv("DATA_SAVE_FOLDER")
        self.exp_tag = exp_tag
        self.LigandMode = LigandMode
        if self.LigandMode:
            from utils.my_smina import SminaEvaluator
            self.smina_evaluator = SminaEvaluator(smina_binary="")
    
    @staticmethod
    def get_store_folder(base_path,flag_extension='.flag',NUMADD=1):
        flag_files = glob.glob(os.path.join(base_path, f"*{flag_extension}"))
        if not flag_files:
            #初始化
            initial_flag_path = os.path.join(base_path, f"0{flag_extension}")
            open(initial_flag_path, 'a').close()
            current_flag_path = initial_flag_path
            current_flag_number=0
        else:
            current_flag_path = flag_files[0]
            # number获取
            file_name_without_ext = os.path.splitext(
                os.path.basename(current_flag_path))[0]
            current_flag_number = int(file_name_without_ext)
        
        new_folder_paths=[]
        for i_seq in range(NUMADD):
            new_folder_path = os.path.join(
                base_path, "sampleItem_"+str(current_flag_number+i_seq))
            os.makedirs(new_folder_path)
            new_folder_paths.append(new_folder_path)

        new_flag_path = os.path.join(
            base_path, f"{current_flag_number+NUMADD}{flag_extension}")
        os.rename(current_flag_path, new_flag_path)

        return new_folder_paths


    @torch.no_grad()
    def __call__(self,
                 seqs,
                 motif_index,
                 ref_pdbpath,
                 data_index,
                 pocket_index=None,
                 state='train',
                 ligand_path=None,
                 force=False,
                 **kwargs_to_save
                 ):
       

        # time_start = time.time()
        # time_here = time_start
        
        item_dir = os.path.join(
            self.save_path, self.exp_tag, state, "dataItem_" + str(data_index))
        os.makedirs(item_dir, exist_ok=True)

        if force:
            print("skip fold")
            esmf_dirs= [os.path.join(item_dir, "sampleItem_"+str(i_sp)) for i_sp in range(len(seqs))]
        else:
            esmf_dirs = self.get_store_folder(item_dir, NUMADD=len(seqs))
            run_folding_parallel(self.esmf, seqs, esmf_dirs)
        
        # time_now = time.time()
        # print(f"Folding took {time_now - time_here:.2f} seconds.")
        # time_here = time_now

        motif_mask = None
        if motif_index is not None:
            motif_index = [idx - 1 for idx in motif_index if idx - 1 < len(seqs[0])]
            motif_mask = np.zeros(len(seqs[0]), dtype=bool)
            motif_mask[motif_index] = True

        pocket_mask = None
        if self.LigandMode:
            if pocket_index is not None:
                pocket_index = [idx - 1 for idx in pocket_index if idx - 1 < len(seqs[0])]
                pocket_mask = np.zeros(len(seqs[0]), dtype=bool)
                pocket_mask[pocket_index] = True
        
        def _process_single_sample(args):
            i_sp, esmf_dir, sequence = args
            
            metric = {}
            misc = {}
            
            esmf_sample_path = os.path.join(esmf_dir, 'output.pdb')

            motif_rmsd, global_rmsd, global_tm, motif_plddt, global_plddt = align_and_calc_rmsds(
                ref_pdbpath, esmf_sample_path, esmf_sample_path, 
                motif_mask, pocket_masks=pocket_mask
            )
            
            # 记录 Metric
            metric['motif_rmsd'] = motif_rmsd
            metric['rmsd'] = global_rmsd
            metric['tm_score'] = global_tm
            metric['plddt_whole'] = global_plddt
            metric['plddt_motif'] = motif_plddt
            metric['motif_mask'] = motif_mask 

            if self.LigandMode:
                affinity = self.smina_evaluator.my_smina(
                    esmf_sample_path, ligand_path, ligand_path, cpu=1
                )
                metric['vina_affinity'] = affinity
            else:
                metric['vina_affinity'] = 0.0
            
            misc['ref_pdbpath'] = ref_pdbpath
            misc['sample_path'] = esmf_sample_path
            misc['ligand_path'] = ligand_path
            misc['sequence'] = sequence
            misc.update(metric)

            exclude_keys = ['motifI_tosave', 'pocketI_tosave']
            kwargs_to_save_sp = {
                k: v if (k in exclude_keys or v is None) else v[i_sp] 
                for k, v in kwargs_to_save.items()
            }
            misc.update(kwargs_to_save_sp)

            misc_path = os.path.join(esmf_dir, 'misc.pt')
            torch.save(misc, misc_path)
            return metric

        task_args = list(enumerate(zip(esmf_dirs, seqs))) # [(0, (dir1, seq1)), (1, ...)]
        task_args = [(i, d, s) for i, (d, s) in task_args]
        results_list = []
        
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), len(task_args)) 
        # print(f"Starting parallel evaluation with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = executor.map(_process_single_sample, task_args)
            results_list = list(results_iterator)

        return results_list


