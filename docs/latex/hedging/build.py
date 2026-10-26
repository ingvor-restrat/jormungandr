#!/usr/bin/env python3
"""Build the stock-and-option hedging tutorial from its repository source."""
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
stem = "volt_jormungandr_option_hedging"
build = here / "build"
build.mkdir(exist_ok=True)
subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                "-outdir=" + str(build), stem + ".tex"], cwd=here, check=True,
               stdout=(build / "build.txt").open("w"), stderr=subprocess.STDOUT)
log = (build / (stem + ".log")).read_text(errors="replace")
if any(warning in log for warning in ["Overfull", "undefined references", "undefined citations"]):
    raise RuntimeError("Resolve layout or reference warnings before delivering the paper")
output = here / (stem + ".pdf")
shutil.copy2(build / output.name, output)
print(output.relative_to(here.parents[2]))
