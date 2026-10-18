"""Create a clean ZIP without environments, smoke-run outputs or local work files."""
import argparse
from pathlib import Path
import os
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','__pycache__','.work','runs'}

def files():
    for base,dirs,names in os.walk(ROOT):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDE and not d.startswith('.venv'))
        for name in sorted(names):
            path=Path(base)/name
            if path.suffix in ('.zip','.log','.pyc') or name=='MUJOCO_LOG.TXT':continue
            yield path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'ME639_Lab3_23110113_Submission.zip');a=p.parse_args()
    selected=list(files())
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in selected:z.write(file,Path('Lab 3')/file.relative_to(ROOT))
    print(f'{a.output}: {len(selected)} files, {a.output.stat().st_size/2**20:.1f} MiB')

if __name__=='__main__':main()
