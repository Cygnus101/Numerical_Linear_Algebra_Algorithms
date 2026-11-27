from pathlib import Path
import ast, contextlib, io, json, os, sys, time, warnings
import nbformat
import torch
root=Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0,str(root))
torch.set_num_threads(1)
report=[]
for p in sorted(root.glob('*.ipynb')):
    started=time.perf_counter()
    nb=nbformat.read(p,as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code': ast.parse(c.source)
    ns={'__name__':'__main__'}
    output=io.StringIO()
    status='passed'
    phase='small cases and real default dataset'
    try:
        with contextlib.redirect_stdout(output), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for i,c in enumerate(nb.cells):
                if c.cell_type!='code': continue
                src=c.source
                kaggle = p.name.endswith('_10^4.ipynb')
                if kaggle and src.startswith('MATRIX =') and '--full-kaggle' not in sys.argv:
                    phase = '3x3 checks only; bcsstk17 full run requires --full-kaggle'
                    break
                exec(compile(src,str(p)+f':cell{i}','exec'),ns)
        warning_text=sorted(set(str(w.message) for w in caught))
    except Exception as e:
        import traceback
        status='failed'
        output.write(traceback.format_exc())
        warning_text=[]
    item={'notebook':p.name,'status':status,'scope':phase,'seconds':time.perf_counter()-started,'output':output.getvalue(),'warnings':warning_text}
    report.append(item)
    print(p.name,status,round(item['seconds'],2),flush=True)
    if status=='failed': print(item['output'],flush=True)
    (root/'validation_notebooks.json').write_text(json.dumps(report,indent=2)+'\n')
if any(r['status']=='failed' for r in report): sys.exit(1)
