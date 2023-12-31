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


solNT = namedtuple('solNT', 'success niter L f x')

def get_combos(mat):
    mat=sp.sparse.csr_array(mat)
    nz = (mat!=0).astype(np.int8)
    c = (nz @ nz.T).astype(bool)
    N=mat.shape[0]
    cand = set(range(N))
    cols=[]
    rows=[]
    vecs=[]
    while cand:
        vec = np.zeros(N)
        k = cand.pop()
        cand2 = deepcopy(cand)
        v=set(c.indices[c.indptr[k]:c.indptr[k+1]])  #rows that have common non-zero columns with row k
        matcols = mat.indices[mat.indptr[k]:mat.indptr[k+1]]   #non-zero columns of row k
        col1=[matcols]  #indices of non-zero columns to be concatenated via cols into a 1D array for COO
        row1=[np.full_like(matcols,k)]  #index of row k repeated as many times as non-zero columns for COO
        ls1=[k]  # eventually this will be a list of rows with non-common non-zero columns that can be combined
        while True:
            cand2-=v  #remove rows with common non-zero columns from candidate rows to be combined with row k
            if not(cand2):
                break  #no more candidate rows to check
            k2=cand2.pop()  #take a row k2 from candidates to combine with row k
            v = set(c.indices[c.indptr[k2]:c.indptr[k2+1]]) # find it's non-zero columns
            ls1.append(k2)  
            matcols=mat.indices[mat.indptr[k2]:mat.indptr[k2+1]]
            col1.append(matcols)
            row1.append(np.full_like(matcols,k2))
            cand.remove(k2)
        vec[ls1]=1.
        vecs.append(vec)
        cols.append(np.concatenate(col1))
        rows.append(np.concatenate(row1))
    return vecs, rows, cols



def LM_print(res):
    print(f'Iteration: {res.niter}  L: {res.L:0.2e}')
    print(f'f abs mean: {np.mean(np.abs(res.f)):0.2e}  max: {np.max(np.abs(res.f)):0.2e}')


def LM(f, x0):
    expected = jax.jacobian(f)(x0)
    vecs,rows,cols= get_combos(expected)
    data_rows = np.concatenate(rows)
    data_cols=np.concatenate(cols)

    def jac(x):
        _, vjp = jax.vjp(f, x)
        data=[]
        for vec,col in zip(vecs,cols):
            data.append(vjp(vec)[0][col])

        data = np.concatenate(data)
        return sp.sparse.coo_matrix((data,(data_rows, data_cols)))
    
    
    

    def solve(x0=x0,L=1000, rho_tol = 0.1, maxiter=1000, xtol=1e-12, ftol=1e-12, printfunc=LM_print, iter_print=int(1e9)):

        x=x0
        f1 = f(x)
        j = jac(x).tocsr()
        print(f'Levenberg-Marquadrt Jacobian shape: {j.shape}')
        jj = (j.T @ j)
        jTf = j.T @ f(x)
        
        for i in range(maxiter):
            Ljj = L*jj.diagonal()
            h = pypardiso.spsolve(jj + sp.sparse.diags(Ljj,format='csr'), -jTf)
            if i%iter_print == 0 and i > 0:
                res=solNT(success=False, x=x, f=f1, niter=i, L=L)
                printfunc(res)   
            if np.max(np.abs(h)/(np.abs(x)+1e-15))<xtol or np.max(np.abs(f1))<ftol:
                res=solNT(success=True, x=x, L=L, f=f1, niter=i)
                if iter_print < 1e9:
                    printfunc(res) 
                return res

            f2 = f(x+h)
            rho=(np.sum(f1**2)-np.sum(f2**2))/np.sum(h*(Ljj*h - j.T @ f1))
            if rho>rho_tol:
                f1=f2
                x+=h
                j = jac(x).tocsr()
                jj = (j.T @ j)
                jTf = j.T @ f(x)
                L = max(L/9, 1e-12)
            else:
                L = min(L*11, 1e7)
            
   

        res=solNT(success=False, x=x, f=f1, niter=i, L=L)
        return res

    return solve
     

     

     

     

    