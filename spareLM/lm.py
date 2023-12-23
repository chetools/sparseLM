import pypardiso
import numpy as np
from collections import namedtuple
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import scipy as sp
from copy import deepcopy
jnp.set_printoptions(precision=2,linewidth=120)
np.set_printoptions(precision=2,linewidth=120)


solNT = namedtuple('solNT', 'success niter f x')

def get_combos(mat):
    mat=sp.sparse.csr_array(mat)
    nz = (mat!=0).astype(np.int8)
    c = (nz @ nz.T).astype(bool)
    N=mat.shape[0]
    cand = set(range(N))
    cols=[]
    rows=[]
    vecs=[]
    ls=[]
    while cand:
        vec = np.zeros(N)
        k = cand.pop()
        cand2 = deepcopy(cand)
        v=set(c.indices[c.indptr[k]:c.indptr[k+1]])
        matcols = mat.indices[mat.indptr[k]:mat.indptr[k+1]]
        col1=[matcols]
        row1=[np.full_like(matcols,k)]
        ls1=[k]
        while True:
            cand2-=v
            if not(cand2):
                break
            first=cand2.pop()
            v = set(c.indices[c.indptr[first]:c.indptr[first+1]])
            ls1.append(first)
            matcols=mat.indices[mat.indptr[first]:mat.indptr[first+1]]
            col1.append(matcols)
            row1.append(np.full_like(matcols,first))
            cand.remove(first)
        vec[ls1]=1.
        vecs.append(vec)
        cols.append(np.concatenate(col1))
        rows.append(np.concatenate(row1))
        ls.append(ls1)
    return ls, vecs, rows, cols


def LM(f, x0, L=1000, rho_tol = 0.1, maxiter=1000, xtol=1e-12, ftol=1e-12):
    expected = jax.jit(jax.jacobian(f))(x0)
    ls, vecs,rows,cols= get_combos(expected)
    

    def jac(x):
        _, vjp = jax.vjp(f, x)
        data=[]
        for vec,col in zip(vecs,cols):
            data.append(vjp(vec)[0][col])

        data = np.concatenate(data)
        data_rows = np.concatenate(rows)
        data_cols=np.concatenate(cols)
        return sp.sparse.coo_matrix((data,(data_rows, data_cols)))

    def solve(x0=x0,L=L):

        x=x0
        f1 = f(x)
        j = jac(x).tocsr()
        jj = (j.T @ j)
        jTf = j.T @ f(x)
        
        for i in range(maxiter):
            Ljj = L*jj.diagonal()
            h = pypardiso.spsolve(jj + sp.sparse.diags(Ljj,format='csr'), -jTf)
            if np.max(np.abs(h/x))<xtol or np.max(np.abs(f1))<ftol:
                res=solNT(success=True, x=x, f=f1, niter=i)
                return res

            f2 = f(x+h)
            rho=(np.sum(f1**2)-np.sum(f2**2))/np.sum(h*(Ljj*h - j.T @ f1))
            if rho>rho_tol:
                f1=f2
                x+=h
                j = jac(x).tocsr()
                jj = (j.T @ j)
                jTf = j.T @ f(x)
                L = max(L/9, 1e-7)
            else:
                L = min(L*11, 1e7)        

        res=solNT(success=False, x=x, f=f1, niter=i)
        return res

    res=solve()
    return res
     

     

     

     

    