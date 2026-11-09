from Bio.PDB import PDBParser, Superimposer, PDBIO
import numpy as np
from Bio.PDB.Polypeptide import three_to_one
from tmtools import tm_align

def align_and_calc_rmsds(fixed_pdb_file, moving_pdb_file, output_pdb_file, motif_mask,pocket_masks=None):
    parser = PDBParser(QUIET=True)
    fixed_structure = parser.get_structure("fixed", fixed_pdb_file)
    moving_structure = parser.get_structure("moving", moving_pdb_file)

    fixed_model = fixed_structure[0]
    moving_model = moving_structure[0]
    fixed_chain=next(fixed_model.get_chains())
    moving_chain=next(moving_model.get_chains())

    all_fixed_atoms = []
    all_moving_atoms = []
    motif_fixed_atoms = []
    motif_moving_atoms = []
    pocket_fixed_atoms = []
    pocket_moving_atoms = []
    # TM calculation
    all_fixed_atoms_coords = []
    all_moving_atoms_coords = []
    all_fixed_atoms_res = []
    all_moving_atoms_res = []
    # plddts
    motif_plddts = []
    global_plddts = []
    
    fixed_std_residues = [res for res in fixed_chain if res.id[0] == ' ']
    moving_std_residues = [res for res in moving_chain if res.id[0] == ' ']

    if len(fixed_std_residues) != motif_mask.shape[0]:
        print(f"Warning:{fixed_pdb_file} Length Mismatch: Chain has {len(fixed_std_residues)} standard residues, but mask shape is {motif_mask.shape[0]}. Truncating to shortest length.")
    min_len=min(len(fixed_std_residues),motif_mask.shape[0])
    fixed_std_residues=fixed_std_residues[:min_len]
    moving_std_residues=moving_std_residues[:min_len]
    motif_mask=motif_mask[:min_len]

    for i in range(len(fixed_std_residues)):
        try:
            fixed_res = fixed_std_residues[i]
            moving_res = moving_std_residues[i]
        except IndexError:
            print(fixed_pdb_file)
            print(moving_pdb_file)
            print(motif_mask)
            print(i,len(fixed_std_residues),len(moving_std_residues))

            # print()
            continue
            # raise IndexError
        mask=motif_mask[i]
        pocket_mask=pocket_masks[i] if pocket_masks is not None else False


        if 'CA' in moving_res:
            bfactor = moving_res['CA'].get_bfactor()
            global_plddts.append(bfactor)
            if mask:
                motif_plddts.append(bfactor)

        if 'CA' in fixed_res and 'CA' in moving_res:
            f_atom = fixed_res['CA']
            m_atom = moving_res['CA']
            

            all_fixed_atoms.append(f_atom)
            all_moving_atoms.append(m_atom)

            if mask:
                motif_fixed_atoms.append(f_atom)
                motif_moving_atoms.append(m_atom)
            if pocket_mask:
                pocket_fixed_atoms.append(f_atom)
                pocket_moving_atoms.append(m_atom)


            all_fixed_atoms_res.append(three_to_one(fixed_res.get_resname()))
            all_moving_atoms_res.append(three_to_one(moving_res.get_resname()))
            all_fixed_atoms_coords.append(f_atom.get_coord())
            all_moving_atoms_coords.append(m_atom.get_coord())

    motif_plddt = np.mean(motif_plddts) if motif_plddts else 0.0
    global_plddt = np.mean(global_plddts) if global_plddts else 0.0

    if not motif_fixed_atoms:
        raise ValueError("Mask error")

    # ---------------------------------------------------------
    # Imposer 1: Motif RMSD
    # ---------------------------------------------------------
    imposer_motif = Superimposer()
    imposer_motif.set_atoms(motif_fixed_atoms, motif_moving_atoms)
    motif_rmsd = imposer_motif.rms

    # ---------------------------------------------------------
    # Imposer 2: Global RMSD
    # ---------------------------------------------------------

    imposer_global = Superimposer()
    imposer_global.set_atoms(all_fixed_atoms, all_moving_atoms)
    global_rmsd = imposer_global.rms

    # --
    # TMalign
    # --
    fixed_coords=np.array(all_fixed_atoms_coords)
    moving_coords=np.array(all_moving_atoms_coords)
    fixed_seq=''.join(all_fixed_atoms_res)
    moving_seq=''.join(all_moving_atoms_res)
    tm_results = tm_align(fixed_coords, moving_coords, fixed_seq, moving_seq)
    global_tm=tm_results.tm_norm_chain2

    if pocket_masks is not None:
        imposer_ligand = Superimposer()
        imposer_ligand.set_atoms(pocket_fixed_atoms, pocket_moving_atoms)
        imposer_ligand.apply(moving_structure.get_atoms())
    else:
        imposer_motif.apply(moving_structure.get_atoms())

    io = PDBIO()
    io.set_structure(moving_structure)
    io.save(output_pdb_file)
    
    return motif_rmsd, global_rmsd,global_tm,motif_plddt,global_plddt



