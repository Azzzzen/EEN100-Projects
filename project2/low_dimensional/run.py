"""Fixed-p Lasso experiment: only the two required success/failure settings.

Compare a predetermined penalty with existence of a successful model anywhere
on the complete piecewise-linear Lasso path. All outputs concern p=3, q=2.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import warnings
import numpy as np
from sklearn.linear_model import lasso_path, lars_path
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
CASES={'low_easy':{'r':.3,'seed_id':0},'low_fail':{'r':.7,'seed_id':2}}
SAMPLE_SIZES=[50,100,200,400,600,800]
SEED=20261012
ZERO_TOL=1e-6


def specification(name,n=None):
    r=CASES[name]['r']
    cov=np.array([[1.,.2,r],[.2,1.,r],[r,r,1.]])
    beta=np.array([1.,1.,0.])
    margin=1-abs(cov[2,:2]@np.linalg.solve(cov[:2,:2],np.sign(beta[:2])))
    assert np.linalg.eigvalsh(cov).min()>0
    return 3,2,beta,float(margin),cov


def make_seed(name,n,repeat):
    # Stable identifiers preserve the original independent evaluation samples.
    return np.random.SeedSequence([SEED,CASES[name]['seed_id'],n,1,repeat])


def generate(name,n,seed):
    _,q,beta,margin,cov=specification(name)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,3))@np.linalg.cholesky(cov).T
    x-=x.mean(axis=0)
    x/=np.sqrt(np.mean(x*x,axis=0))
    x=np.asfortranarray(x)
    # Original errors are iid N(0,1). Centering removes the intercept.
    noise=rng.normal(size=n);noise-=noise.mean()
    y=x@beta+noise
    c=x.T@x/n
    sample_margin=1-abs(c[2,:2]@np.linalg.solve(c[:2,:2],np.sign(beta[:q])))
    return x,y,beta,margin,float(sample_margin),float(np.linalg.eigvalsh(c[:2,:2]).min())


def sign_success(coefficients,beta):
    sign=np.where(abs(coefficients)>ZERO_TOL,np.sign(coefficients),0)
    return np.all(sign==np.sign(beta)[:,None],axis=0)


def kkt_residual(x,y,coef,alphas):
    grad=x.T@(y[:,None]-x@coef)/len(y)
    residual=np.maximum(abs(grad)-np.asarray(alphas)[None,:],0)
    active=abs(coef)>1e-8
    residual[active]=abs(grad-np.asarray(alphas)[None,:]*np.sign(coef))[active]
    return float(residual.max())


def fit(name,n,repeat):
    x,y,beta,margin,sample_margin,eigenmin=generate(name,n,make_seed(name,n,repeat))
    alpha=.75*n**(-.15)  # Paper lambda_n=1.5*n**.85.
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        _,fixed,dual_gap=lasso_path(x,y,alphas=[alpha],tol=1e-11,max_iter=30000)
        alphas,_,coef=lars_path(x,y,method='lasso',alpha_min=0.,eps=1e-12)
    assert np.isclose(alphas[-1],0), 'LARS did not reach the unpenalized endpoint'
    assert np.all(np.diff(alphas)<=1e-12), 'Unordered path knots'
    # On each open segment, coefficients are affine and exact support/sign
    # stay fixed. Include segment midpoints, since endpoints may have zeros.
    midpoint_coef=(coef[:,:-1]+coef[:,1:])/2
    midpoint_alpha=(alphas[:-1]+alphas[1:])/2
    states=np.concatenate([coef,midpoint_coef],axis=1)
    state_alphas=np.r_[alphas,midpoint_alpha]
    fixed_success=bool(sign_success(fixed,beta)[0])
    path_success=bool(sign_success(states,beta).any())
    assert not fixed_success or path_success
    # Independent solvers must agree at the predetermined penalty.
    interp=np.array([np.interp(alpha,alphas[::-1],c[::-1],left=c[-1],right=0.) for c in coef])
    assert np.allclose(interp,fixed[:,0],atol=2e-7), 'CD/LARS disagree'
    certificate=max(kkt_residual(x,y,fixed,[alpha]),kkt_residual(x,y,states,state_alphas))
    assert certificate<2e-4, f'KKT violation: {certificate}'
    selected=abs(fixed[:,0])>ZERO_TOL
    return dict(scenario=name,n=n,repeat=repeat,alpha=alpha,lambda_paper=2*n*alpha,
        population_margin=margin,sample_margin=sample_margin,active_eigenmin=eigenmin,
        fixed_success=fixed_success,path_success=path_success,
        false_positive=int(selected[2]),false_negative=int((~selected[:2]).sum()),
        max_kkt_residual=certificate,dual_gap=float(dual_gap[0])),fixed[:,0],alphas,coef


def wilson(count,total):
    z=1.95996398454;p=count/total
    center=(p+z*z/(2*total))/(1+z*z/total)
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return float(center-radius),float(center+radius)


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(repeats=50,output='results'):
    out=ROOT/output;out.mkdir(parents=True,exist_ok=True)
    start=time.time();summary=[];runs=[];examples=[]
    for name in CASES:
        for n in SAMPLE_SIZES:
            cells=[];fixed_coefficients=[]
            for repeat in range(repeats):
                row,fixed,alphas,coef=fit(name,n,repeat)
                cells.append(row);runs.append(row);fixed_coefficients.append(fixed)
                if n==800 and repeat==0:
                    examples.append({**row,'true_beta':[1.,1.,0.],'estimated_beta':fixed.tolist(),
                        'path_alphas':alphas.tolist(),'path_coefficients':coef.tolist()})
            r=CASES[name]['r'];_,_,_,margin,cov=specification(name)
            row=dict(scenario=name,r=r,n=n,p=3,q=2,repeats=repeats,
                population_ic_score=1-margin,population_margin=margin,
                population_eigenmin=float(np.linalg.eigvalsh(cov).min()),
                fixed_alpha=.75*n**(-.15),fixed_lambda=1.5*n**.85,
                sample_margin_mean=float(np.mean([c['sample_margin'] for c in cells])),
                sample_condition_fraction=float(np.mean([c['sample_margin']>0 for c in cells])),
                false_positive_mean=float(np.mean([c['false_positive'] for c in cells])),
                false_negative_mean=float(np.mean([c['false_negative'] for c in cells])))
            for metric in ['fixed','path']:
                count=sum(c[f'{metric}_success'] for c in cells)
                lo,hi=wilson(count,repeats)
                row.update({f'{metric}_successes':count,f'{metric}_success':count/repeats,
                            f'{metric}_ci_low':lo,f'{metric}_ci_high':hi})
            summary.append(row)
            np.savez_compressed(out/f'{name}_n{n}.npz',
                fixed_success=[c['fixed_success'] for c in cells],
                path_success=[c['path_success'] for c in cells],
                fixed_coefficients=fixed_coefficients,
                sample_margin=[c['sample_margin'] for c in cells])
            print(f'{name} n={n}: fixed={row["fixed_success"]:.3f}, any path={row["path_success"]:.3f}',flush=True)
    write_csv(out/'summary.csv',summary);write_csv(out/'runs.csv',runs)
    (out/'example_fits.json').write_text(json.dumps(examples,indent=2)+'\n')
    import sklearn
    metadata=dict(seed=SEED,cases=CASES,sample_sizes=SAMPLE_SIZES,repeats=repeats,
        realizations=len(runs),p=3,q=2,beta=[1,1,0],original_noise='iid N(0,1)',
        intercept='X and response centered; feature columns scaled to mean square 1',
        penalty='paper lambda=1.5*n**.85; sklearn alpha=lambda/(2*n)',
        path_check='LARS complete path knots and segment midpoints; alpha_min=0',
        path_metric='exists a sign-correct model on path; truth-based diagnostic, not practical tuning',
        coefficient_zero_tolerance=ZERO_TOL,
        max_kkt_residual=max(c['max_kkt_residual'] for c in runs),
        validation='coordinate descent compared with LARS interpolation for every realization',
        seconds=time.time()-start,sklearn=sklearn.__version__,numpy=np.__version__)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=50)
    parser.add_argument('--output',default='results');args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.repeats,args.output)
