import os
import subprocess
import re
import os
import shutil

class SminaEvaluator:
    def __init__(self, smina_binary="",adfr_binary=""):
        self.smina_binary=f"{os.getenv('SMINA')}" if not smina_binary else smina_binary
        self.adfr_binary=f"{os.getenv('ADFRTOOLS_DIR')}/prepare_receptor" if not adfr_binary else adfr_binary
        self.reduce_binary=f"{os.getenv('ADFRTOOLS_DIR')}/reduce" 
        self.docking_pattern = re.compile(r'^\s*1\s+(-?\d+\.\d+)', re.MULTILINE)

    def addH_by_reduce(self, pdb_path):
        temp_pdb = pdb_path + ".temp"
        command = [self.reduce_binary, pdb_path]
        try:
            with open(temp_pdb, 'w') as f_out:
                subprocess.run(command, stdout=f_out, stderr=subprocess.DEVNULL, check=True)

            shutil.move(temp_pdb, pdb_path)
        except subprocess.CalledProcessError:
            print(f"Reduce failed on {pdb_path}")
            if os.path.exists(temp_pdb):
                os.remove(temp_pdb)
        except Exception as e:
            print(f"Error during addH: {e}")

    def prepare_rec(self, pdb_path):
        out_pdbqt = pdb_path.replace('.pdb','.pdbqt')
        command = [
            self.adfr_binary,
            '-r', pdb_path,
            '-o', out_pdbqt,
            # '-A', 'hydrogens',
            # '-U', 'nphs_lps_waters'
        ]
        # 执行命令
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            check=True
        )
        return out_pdbqt

    def my_smina(self, receptor_file, ligand_file, autobox_ligand,dock_it=False,cpu=1):
        """
        运行 Smina 并返回 (affinity, rmsd)
        """
        self.addH_by_reduce(receptor_file)
        receptor_qt=self.prepare_rec(receptor_file)
        
        cmd = [
            self.smina_binary,
            "-r", receptor_qt,
            "-l", ligand_file,
            "--autobox_ligand", autobox_ligand,
            "--autobox_add", "4.0",
            "--cpu", "1",  
            "--exhaustiveness", "1",
        ]

        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            output_text = result.stdout
            affinity = self._parse_dock(output_text)
            
            return affinity

        except subprocess.CalledProcessError as e:
            print(f"Smina Error: {e.stderr}")
            return 0.0
        except Exception as e:
            print(f"Parsing Error: {e}")
            return 0.0

    def _parse_dock(self, text):
        match = self.docking_pattern.search(text)
        if match:
            return float(match.group(1))
        raise ValueError("Docking score not found in output.")

    
    
