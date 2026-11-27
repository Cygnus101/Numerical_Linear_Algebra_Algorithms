from pathlib import Path
import sys, time, json
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from common import loader, data_path
from Iterative_Methods.conjugate_gradient import conjugate_gradient
from Iterative_Methods.newton import newton
from Eigenvalue_Methods.inverse_power_method import inverse_power_method
A=loader('suitesparse',path=data_path('bcsstk17'),dense=False).tocsr()
b=A @ np.ones(A.shape[0])
results=[]
for fn in [conjugate_gradient, newton, inverse_power_method]:
 start=time.perf_counter()
 if fn is inverse_power_method:
  lam,x,info=fn(A,maxiter=1000)
  info=dict(info,eigenvalue=lam)
 else:
  x,info=fn(A,b)
  info=dict(info,solution_error=float(np.linalg.norm(x-1)/np.sqrt(len(x))))
 item={'method':fn.__name__,'matrix':'bcsstk17','shape':list(A.shape),'nnz':A.nnz,'seconds':time.perf_counter()-start,**info}
 print(item,flush=True)
 results.append(item)
 (root / 'validation_sparse_10_4.json').write_text(json.dumps(results,indent=2)+'\n')
