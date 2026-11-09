"""CPU-only integration checks for the adapter; no model inference or training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fidia_ascend.mapping import sequence_positions_to_pdb_numbers, wrap_motif_sampler


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pdb(self, start):
        path = self.folder / f"reference_{start}.pdb"
        path.write_text("".join(
            f"ATOM  {i+1:5d}  CA  ALA A{start+i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 50.00           C\n"
            for i in range(5)) + "END\n")
        return path

    def test_zero_and_one_based_pdbs_represent_the_same_sequence_mask(self):
        self.assertEqual(sequence_positions_to_pdb_numbers(self.pdb(0), [1, 3, 5]), [0, 2, 4])
        self.assertEqual(sequence_positions_to_pdb_numbers(self.pdb(1), [1, 3, 5]), [1, 3, 5])

    def test_sampler_receives_pdb_identifiers_without_mutating_the_csv_mask(self):
        path, mask = self.pdb(0), [1, 3, 5]
        calls = []
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            return "original_result"
        wrapped = wrap_motif_sampler(native, lambda: "motif")
        result = wrapped("cond", pdb_path=str(path), fix_index=mask, num_rest=8)
        self.assertEqual(result, "original_result")
        self.assertEqual(mask, [1, 3, 5])
        self.assertEqual(calls[0][1]["fix_index"], [0, 2, 4])
        self.assertEqual(calls[0][1]["num_rest"], 8)
        legacy = wrap_motif_sampler(native, lambda: "motif", "pdb")
        legacy("cond", pdb_path=str(path), fix_index=mask)
        self.assertIs(calls[-1][1]["fix_index"], mask)

    def test_invalid_positions_are_rejected(self):
        path = self.pdb(0)
        for positions in ([0], [6], [1, 1], [1.5]):
            with self.assertRaises(ValueError):
                sequence_positions_to_pdb_numbers(path, positions)

    def test_cli_dry_run_preserves_native_parameters(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "fidia_ascend", "train", "--project-root", str(root),
            "--dry-run", "--config.sample.sampling_temp=0.3", "--config.wandb=False"],
            cwd=root, text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["motif_indexing"], "sequence")
        self.assertIn("--config=config/FIDIA.py:IAPO", value["native_arguments"])
        self.assertIn("--config.sample.sampling_temp=0.3", value["native_arguments"])
        self.assertIn("--config.wandb=False", value["native_arguments"])
        self.assertIn("--config.task=motif", value["native_arguments"])

    def test_native_bridge_installs_mapping_and_blocks_training_in_infer_mode(self):
        from fidia_ascend.runtime import launch_native
        root = Path(__file__).resolve().parents[1]
        reference = self.pdb(0)
        captured = []
        def sampler(*args, **kwargs):
            captured.append(kwargs["fix_index"])
        cfg = SimpleNamespace(task="motif", train=SimpleNamespace(mode="IAPO"), eval_bmk=False)
        namespace = {"FLAGS": SimpleNamespace(config=cfg), "Lmpnn_with_logprob": sampler,
                     "reference": str(reference)}
        exec("def main(argv):\n    return Lmpnn_with_logprob('cond', pdb_path=reference, fix_index=[1, 3])\n", namespace)
        namespace["app"] = SimpleNamespace(run=lambda callback: callback(None))
        old_argv = sys.argv[:]
        try:
            with patch("fidia_ascend.runtime.enable_backend"), patch("fidia_ascend.runtime.runpy.run_path", return_value=namespace):
                with self.assertRaises(ValueError):
                    launch_native(root, [], expected="infer")
            self.assertEqual(captured, [])
            # Recreate the untouched upstream binding for the successful call.
            namespace["Lmpnn_with_logprob"] = sampler
            cfg.train.mode, cfg.eval_bmk = "EVAL", True
            with patch("fidia_ascend.runtime.enable_backend"), patch("fidia_ascend.runtime.runpy.run_path", return_value=namespace):
                launch_native(root, [], expected="infer")
            self.assertEqual(captured, [[0, 2]])
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
