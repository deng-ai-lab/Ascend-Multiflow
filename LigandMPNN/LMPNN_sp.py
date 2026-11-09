import numpy as np
import torch
from LigandMPNN.data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    parse_PDB,
    restype_int_to_str,
)

import prody
prody.confProDy(verbosity='error')

def Lmpnn_grad_with_logprob(Wrapedmodel,accelerator,**kwargs):

    model=accelerator.unwrap_model(Wrapedmodel)
    S_sample=kwargs['S_sample']
    decoding_order=kwargs['decoding_order']
    feature_dict=kwargs['feature_dict']
    my_dict = model.forward_utils(feature_dict, True, decoding_order,S_sample)
    tmp_probs = torch.nn.functional.softmax(
        my_dict["logits"]/feature_dict["temperature"], dim=-1)
    probs_forward = tmp_probs[:,:, :20] / torch.sum(
        tmp_probs[:, :,:20], dim=-1, keepdim=True
    )
    log_probs = torch.log(probs_forward)
    log_prob = log_probs.gather(
        2, S_sample.unsqueeze(-1)).squeeze(-1)  # B,L 
    return log_prob,log_probs

def Lmpnn_with_logprob(mode,Wrapedmodel,accelerator,args,**kwargs):
    """
    Inference function
    """
    model=accelerator.unwrap_model(Wrapedmodel)
    device =accelerator.device
    pdb_paths = [kwargs['pdb_path']]
    sampling_num=kwargs['num_rest']
    atom_context_num=kwargs['atom_context_num']

    fixed_residues = kwargs['fix_index']
    fixed_residues_multi = {}
    for pdb in pdb_paths:
        fixed_residues_multi[pdb] = fixed_residues

    # deprecated default
    redesigned_residues = []
    redesigned_residues_multi = {}
    for pdb in pdb_paths:
        redesigned_residues_multi[pdb] = redesigned_residues
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=device,
    )
    parse_these_chains_only_list = []
    # ----

    # loop over PDB paths
    for pdb in pdb_paths:
        # ----------------------------------------------------
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]
        
        if args.verbose:
            print(f'Generating sequences: {kwargs["tag"]} - {name}')
            print("Designing protein from this path:", pdb)
        # ----------------------------------------------------

        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]
        parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
            args.pack_side_chains and not args.repack_everything
        )
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=parse_these_chains_only_list,
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )

        chain_letter=chain_letters_list[0]
        fixed_residues = [chain_letter + str(item) for item in fixed_residues]

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        buried_positions = torch.zeros_like(fixed_positions)
        interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        remapped_symmetry_residues = [[]]
        symmetry_weights = [[]]

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        with torch.no_grad():
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            # feature_dict["batch_size"] = args.batch_size
            feature_dict["batch_size"] = sampling_num

            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )

            feature_dict['mode']=mode
            output_dict = model.sample(feature_dict,eval_flag=kwargs.get('eval_flag',False))
            # -------------------------------

            S_sample=output_dict["S"]
            mask_scaffold = (feature_dict['chain_mask'] * feature_dict['mask']).squeeze().bool()
            feature_dict['mask_scaffold']=mask_scaffold

            log_prob,log_probs = Lmpnn_grad_with_logprob(
                Wrapedmodel=Wrapedmodel,
                accelerator=accelerator,
                S_sample=S_sample,
                decoding_order=output_dict["decoding_order"],
                feature_dict=feature_dict
            )

            diff= torch.norm(torch.log(output_dict['sampling_probs'][:,mask_scaffold])-log_probs[:,mask_scaffold])
            if diff > 1e-2:
                print(f"Warning diff>1e-2  - {pdb} - {name} - {diff}")
            if output_dict["S"][:,~mask_scaffold].ne(feature_dict["S"][:,~mask_scaffold]).any():
                print(f"Warning fixed part changed! - {name}")
                print("fixed part in output:", output_dict["S"][:,~mask_scaffold])
                print("fixed part in feature_dict:", feature_dict["S"][:,~mask_scaffold])

            full_seqs=[]
            for i_seq in range(feature_dict["batch_size"]):
                tmp_seq = "".join(
                    [restype_int_to_str[AA] for AA in output_dict["S"][i_seq].cpu().numpy()]
                )
                full_seqs.append(tmp_seq)

            my_tuple = (full_seqs, log_probs,log_prob,
                        output_dict["decoding_order"],mode,S_sample,feature_dict)
            
    return my_tuple





