"""Translate sequence-position motif masks to the identifiers used by the sampler."""
from functools import lru_cache, wraps
from pathlib import Path


@lru_cache(maxsize=512)
def _residue_numbers(filename, modified_ns, size):
    from Bio.PDB import PDBParser
    model = PDBParser(QUIET=True).get_structure("motif_reference", filename)[0]
    chains = list(model.get_chains())
    if len(chains) != 1:
        raise ValueError("The motif adapter currently supports single-chain reference backbones")
    residues = [r for r in chains[0] if r.id[0] == " "]
    if not residues or any("CA" not in r for r in residues):
        raise ValueError("Every reference residue must have a C-alpha atom")
    if any(r.id[2] != " " for r in residues):
        raise ValueError("Insertion-coded residue identifiers are not supported by the upstream sampler")
    numbers = tuple(int(r.id[1]) for r in residues)
    if len(set(numbers)) != len(numbers):
        raise ValueError("Duplicate PDB residue numbers cannot be mapped unambiguously")
    return numbers


def sequence_positions_to_pdb_numbers(pdb_path, positions):
    """CSV motif positions are 1-based sequence positions, not PDB residue numbers."""
    path = Path(pdb_path).expanduser().resolve()
    stat = path.stat()
    numbers = _residue_numbers(str(path), stat.st_mtime_ns, stat.st_size)
    result = []
    for position in positions:
        integer = int(position)
        if integer != position or not 1 <= integer <= len(numbers):
            raise ValueError(f"Motif position {position!r} is outside 1..{len(numbers)}")
        result.append(numbers[integer - 1])
    if len(set(result)) != len(result):
        raise ValueError("The motif contains repeated positions")
    return result


def wrap_motif_sampler(sampler, task_getter, indexing="sequence"):
    if indexing not in {"sequence", "pdb"}:
        raise ValueError("indexing must be 'sequence' or 'pdb'")

    @wraps(sampler)
    def wrapped(*args, **kwargs):
        if indexing == "sequence" and task_getter() == "motif" and kwargs.get("fix_index") is not None:
            translated = dict(kwargs)
            translated["fix_index"] = sequence_positions_to_pdb_numbers(kwargs["pdb_path"], kwargs["fix_index"])
            return sampler(*args, **translated)
        return sampler(*args, **kwargs)
    return wrapped
