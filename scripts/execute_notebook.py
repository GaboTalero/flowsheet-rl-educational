"""Execute the tutorial in the project's Python kernel; optionally test its short training cell."""

import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument(
    "--notebook",
    type=Path,
    default=Path("notebooks/learn_flowsheet_rl_step_by_step.ipynb"),
    help="Notebook path relative to the repository root, or an absolute path",
)
parser.add_argument("--with-short-training", action="store_true")
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
runtime = project / "notebooks" / ".runtime"
runtime.mkdir(exist_ok=True)
os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
os.environ["IPYTHONDIR"] = str(runtime / "ipython")
os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
notebook_path = args.notebook if args.notebook.is_absolute() else project / args.notebook
nb = nbformat.read(notebook_path, as_version=4)
if args.with_short_training:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("RUN_TRAINING = False", "RUN_TRAINING = True")
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(nb, timeout=180, km=km, resources={"metadata": {"path": str(project)}})
client.execute()
args.out.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, args.out)
print("Executed", sum(c.cell_type == "code" for c in nb.cells), "code cells:", args.out)
