# -*- coding: utf-8 -*-
"""Gera notebook .ipynb a partir de cells.py e executa com nbclient."""
import sys, json
import nbformat as nbf

sys.path.insert(0, '/tmp/opencode/t5')
from cells import celulas

def build(path, stage):
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                                 'name': 'python3'}
    for tipo, src in celulas(stage):
        nb.cells.append(nbf.v4.new_markdown_cell(src) if tipo == 'markdown'
                        else nbf.v4.new_code_cell(src))
    with open(path, 'w') as f:
        nbf.write(nb, f)

def run(path):
    from nbclient import NotebookClient
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name='python3',
                            resources={'metadata': {'path': repo}})
    client.execute()
    with open(path, 'w') as f:
        nbf.write(nb, f)
    # extrai SUMARIO
    for c in nb.cells:
        if c.cell_type == 'code':
            for out in (c.get('outputs') or []):
                txt = ''
                if out.get('output_type') == 'stream':
                    txt = ''.join(out.get('text', []))
                elif out.get('output_type') in ('execute_result', 'display_data'):
                    d = out.get('data', {})
                    txt = ''.join(d.get('text/plain', []))
                for ln in txt.splitlines():
                    if ln.startswith('SUMARIO'):
                        print(ln)

if __name__ == '__main__':
    repo = sys.argv[1]
    stage = int(sys.argv[2])
    path = f'{repo}/mlp_iris_scratch.ipynb'
    build(path, stage)
    run(path)
    print(f'OK stage {stage} -> {path}')
