# Numerical Linear Algebra Algorithms

15 implementations and 30 notebooks: one local notebook and one standalone Kaggle notebook per algorithm. The two existing Gaussian notebooks and their earlier outputs are retained. Datasets are unchanged.

| Algorithm | Implementation | Local notebook | Kaggle notebook |
| --- | --- | --- | --- |
| LU row format | `direct_methods/lu_row.py` | `LU_Row.ipynb` | `LU_Row_10^4.ipynb` |
| LU column format | `direct_methods/lu_column.py` | `LU_Column.ipynb` | `LU_Column_10^4.ipynb` |
| LU partial pivoting | `direct_methods/lu_partial_pivoting.py` | `LU_Partial_Pivoting.ipynb` | `LU_Partial_Pivoting_10^4.ipynb` |
| Gaussian, with and without pivoting | `direct_methods/gaussian_elim.py` | `Gaussian_Elimination.ipynb` | `Gaussian_elimination_10^4.ipynb` |
| Cholesky | `direct_methods/cholesky.py` | `Cholesky.ipynb` | `Cholesky_10^4.ipynb` |
| Power | `Eigenvalue_Methods/power_method.py` | `Power_Method.ipynb` | `Power_Method_10^4.ipynb` |
| Inverse power | `Eigenvalue_Methods/inverse_power_method.py` | `Inverse_Power_Method.ipynb` | `Inverse_Power_Method_10^4.ipynb` |
| Newton | `Iterative_Methods/newton.py` | `Newton_Method.ipynb` | `Newton_Method_10^4.ipynb` |
| Conjugate Gradient | `Iterative_Methods/conjugate_gradient.py` | `Conjugate_Gradient.ipynb` | `Conjugate_Gradient_10^4.ipynb` |
| Gradient Descent | `Iterative_Methods/gradient_descent.py` | `Gradient_Descent.ipynb` | `Gradient_Descent_10^4.ipynb` |
| Householder QR | `Orthogonalization/householder_qr.py` | `Householder_QR.ipynb` | `Householder_QR_10^4.ipynb` |
| Jacobi | `Iterative_Methods/jacobi.py` | `Jacobi.ipynb` | `Jacobi_10^4.ipynb` |
| Gauss–Seidel | `Iterative_Methods/gauss_seidel.py` | `Gauss_Seidel.ipynb` | `Gauss_Seidel_10^4.ipynb` |
| SOR | `Iterative_Methods/sor.py` | `SOR.ipynb` | `SOR_10^4.ipynb` |
| Classical Gram–Schmidt | `Orthogonalization/gram_schmidt.py` | `Gram_Schmidt.ipynb` | `Gram_Schmidt_10^4.ipynb` |

## Run

Use a Python environment with NumPy, SciPy and Numba. PyTorch is also needed for the tensor-based Kaggle notebooks. Open local notebooks from this folder or a subfolder. Their setup finds `common.py` and imports the implementations. The system's `python3` currently lacks SciPy; the Anaconda `python` environment has the required packages.

Every Kaggle notebook contains only 3×3 examples and one SuiteSparse case: bcsstk17. Upload a Kaggle notebook and attach only `bcsstk17.mat` using Add Input. The local notebooks retain their original dataset examples. File discovery searches `/kaggle/input` recursively; set `MAT_PATH` if there are multiple matches. A local `data/` fallback allows CPU testing outside Kaggle. No downloads are attempted. The Kaggle notebooks are self-contained and do not require uploading the repository's Python modules.

The tensor notebooks choose CUDA if available, otherwise CPU, use float64, and synchronize CUDA before and after timing. CG, gradient descent, Jacobi and power use sparse CSR matrix-vector products for real data. Gauss–Seidel and SOR use sequential Numba CSR sweeps on CPU. Newton and inverse power use SciPy CPU sparse factorizations for reusable inner solves; neither densifies a large sparse matrix. CPU sparse work is deliberate even in those Kaggle notebooks.

All new in-place factorization work uses copies. The existing Gaussian functions retain their in-place API; notebook calls pass copies. LU permutation convention is `A[p, :] = L @ U`. `lu_solve` permutes `b` using `p`.

## Data and choices

| Existing file | Shape | Nonzeros | Use |
| --- | --- | --- | --- |
| `data/bcsstk09.mat` | 1,083 × 1,083 | 18,437 | Manageable SPD factorization and iterative examples |
| `data/bcsstk17.mat` | 10,974 × 10,974 | 428,650 | Sparse iterative and eigenvalue tests; optional dense experiments |
| `data/west1505.mat` | 1,505 × 1,505 | 5,414 | Nonsymmetric pivoting example |
| `data/bcspwr09.mat` | 1,723 × 1,723 | 6,511 | Expected Cholesky failure: symmetric, not positive definite |

There is no 10^5 matrix or lecture-note file in this repository. No dataset was invented or downloaded. Dense 10^4 experiments in the local notebooks are disabled by default with `RUN_10_4 = False`. Kaggle notebooks run bcsstk17 directly, with no duplicate optional dataset section; a single dense bcsstk17 float64 array already needs about 0.90 GiB, with several arrays required by these methods. The existing Gaussian Kaggle notebook retains its bcsstk17 default and its earlier recorded Tesla T4 results. Those old outputs are explicitly labeled as prior runs.

LU uses right-looking elimination: row format updates `i` then `j`; column format updates `j` then `i`, after computing the multiplier column. The local implementations keep literal loops. PyTorch batches the inner row or column operation; these literal loop-order demonstrations can make many small GPU launches. No lecture convention beyond these explicit choices was available.

Newton and gradient descent use `f(x) = 0.5*x.T@A@x - b.T@x` with SPD A. Newton solves `A*p = b-A*x` using a reused inner factorization; an exact quadratic Newton step normally takes one iteration. Gradient descent uses the exact line-search step `(r.T@r)/(r.T@A@r)`. CG uses the standard recurrence, checks the true residual before accepting convergence, and restarts only if recursive convergence was inaccurate. SPD is an input assumption for these methods; sparse Newton does not certify positive definiteness.

Power iteration targets the eigenvalue largest in magnitude. Inverse iteration targets the eigenvalue nearest `shift`, default zero, with the shifted inner factorization reused. These are explicit iterations; library solvers appear only in the mathematically required Newton/inverse inner solves and small reference checks.

Gram–Schmidt is classical: each projection uses the original column. Householder stores reflectors and applies Q implicitly. Three random probes check real-data factorizations and orthogonality without another cubic verification or an explicit full Q. Probe checks are estimates, not full matrix-norm certificates.

Timings exclude loading and error measurements. Direct-method timings measure factorization; triangular substitutions are performed afterward. Iterative/eigenvalue timings include their inner factorizations. Small examples warm Numba/tensor operations first; first-use sparse backend setup may still be included. Residuals, solution errors and convergence status are printed separately. New notebooks are delivered without execution outputs; actual validation output is saved in the reports below.

## Validation

Run from this folder:

```sh
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/validate_notebooks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/validate_sparse.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/validate_kaggle_setup.py
```

The notebook validator runs only the small Kaggle examples by default; `--full-kaggle` also executes every bcsstk17 case, including expensive dense factorizations. The separate Kaggle setup validator checks the actual bcsstk17 data path and sparse load.

`tests/test_algorithms.py` covers all 15 algorithms in seven test groups, including dense/CSR systems, multiple row swaps and permutation recovery, singular/final-zero-pivot rejection, indefinite Cholesky rejection, zero RHS, iteration limits, nonconvergence, negative dominant eigenvalues, shifted inverse iteration, rectangular QR, dependent columns, and input preservation.

Before the Kaggle dataset-only revision, all 30 notebooks passed format validation, code parsing, small-case checks and manageable real-data execution on CPU. The existing Gaussian Kaggle notebook was tested with bcsstk09 instead of executing its bcsstk17 dense default; its saved source and earlier outputs retain the original choice. Those prior real pivoting tests used west1505, and both Cholesky notebooks rejected bcspwr09. The Kaggle notebooks now use only bcsstk17; the extra datasets and failure case remain in the local notebooks. `validation_notebooks.json` retains the prior actual output and scope for every notebook. `validation_kaggle_setup.json` records the revised Kaggle checks: all 15 notebooks pass syntax/format and 3×3 execution checks, and load/discover the existing bcsstk17 file in sparse form. Dense bcsstk17 factorizations were not rerun. The Kaggle discovery fallback and actual local data loading were exercised; a live Kaggle attachment and CUDA hardware were not available for execution.

`validation_sparse_10_4.json` records additional sparse bcsstk17 runs. Power was already run on sparse bcsstk17 by both power notebooks. No dense 10^4 factorization, live Kaggle session or 10^5 benchmark was run.

Observed limitations are retained in the reports:

- bcsstk09 CG converged in 208 iterations to relative residual about 6.8e-9. Jacobi, Gauss–Seidel, SOR and gradient descent did not reach 1e-8 in 10,000 iterations.
- Classical Gram–Schmidt reconstructed bcsstk09 accurately but lost orthogonality: probe error about 5.2e-6 locally and 4.8e-7 in the PyTorch CPU implementation. Householder orthogonality probe errors were about 2e-15.
- bcsstk17 unpreconditioned CG did not reach tolerance in 10,000 iterations: residual about 1.3e-6 and solution error about 0.22. Sparse Newton reached residual 2.1e-15 in one step. Inverse iteration found eigenvalue 1 with residual 6.5e-11; power found the largest eigenvalue about 1.2961e10.
- west1505 pivoted solves had residuals near 1e-16 but solution errors around 1e-9. A tiny residual alone does not establish an equally tiny forward error.
