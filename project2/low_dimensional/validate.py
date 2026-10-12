"""Check low-dimensional results and independently verify penalty scaling."""
from pathlib import Path
import csv
import json
import numpy as np
from sklearn.linear_model import lasso_path
from threadpoolctl import threadpool_limits
from run import CASES,ZERO_TOL,specification,fit

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'


def validate():
    meta=json.loads((RESULTS/'metadata.json').read_text())
    summary=list(csv.DictReader((RESULTS/'summary.csv').open()))
    runs=list(csv.DictReader((RESULTS/'runs.csv').open()))
    assert len(runs)==meta['realizations']
    assert set(r['scenario'] for r in runs)==set(CASES)
    for name in CASES:
        _,_,_,margin,cov=specification(name)
        assert np.isclose(1-margin,2*CASES[name]['r']/1.2)
        assert np.linalg.eigvalsh(cov).min()>0
        assert np.allclose(np.linalg.eigvalsh(cov[:2,:2]),[.8,1.2])
    # Independent closed-form solution under orthogonal columns.
    rng=np.random.default_rng(42);a=rng.normal(size=(32,3));a-=a.mean(0)
    x=np.asfortranarray(np.linalg.qr(a)[0]*np.sqrt(32))
    beta=np.array([1.,-.8,.1]);alpha=.2
    _,coef,_=lasso_path(x,x@beta,alphas=[alpha],tol=1e-12)
    expected=np.sign(beta)*np.maximum(abs(beta)-alpha,0)
    assert np.allclose(coef[:,0],expected,atol=1e-10)
    threshold_changes=0
    for row in summary:
        name=row['scenario'];n=int(row['n']);repeats=int(row['repeats'])
        assert int(row['p'])==3 and int(row['q'])==2
        assert np.isclose(float(row['fixed_lambda']),2*n*float(row['fixed_alpha']))
        subset=[r for r in runs if r['scenario']==name and int(r['n'])==n]
        assert len(subset)==repeats
        with np.load(RESULTS/f'{name}_n{n}.npz') as data:
            assert len(data['fixed_success'])==repeats
            assert np.all(~data['fixed_success']|data['path_success'])
            for kind in ('fixed','path'):
                assert np.isclose(data[f'{kind}_success'].mean(),float(row[f'{kind}_success']))
                assert int(data[f'{kind}_success'].sum())==int(row[f'{kind}_successes'])
                assert sum(r[f'{kind}_success']=='True' for r in subset)==int(row[f'{kind}_successes'])
                lo,hi=float(row[f'{kind}_ci_low']),float(row[f'{kind}_ci_high'])
                assert -1e-12<=lo<=float(row[f'{kind}_success'])+1e-12<=hi+1e-12<=1+1e-12
            coef=data['fixed_coefficients']
            for cutoff in [1e-7,1e-5]:
                signs=np.where(abs(coef)>cutoff,np.sign(coef),0)
                success=np.all(signs==np.array([1,1,0]),axis=1)
                threshold_changes+=int(np.count_nonzero(success!=data['fixed_success']))
    examples=json.loads((RESULTS/'example_fits.json').read_text())
    for ex in examples:
        row,coef,alphas,path=fit(ex['scenario'],800,0)
        assert np.allclose(coef,ex['estimated_beta'],atol=1e-9)
        assert np.allclose(alphas,ex['path_alphas'])
        assert np.allclose(path,ex['path_coefficients'])
    with (RESULTS/'validation.json').open('w') as f:
        report=dict(settings_checked=len(summary),realizations_checked=len(runs),
            positive_definite_covariances=True,analytic_ic_scores=True,
            independent_soft_threshold_check=True,csv_npz_consistency=True,
            fixed_success_implies_path_success=True,example_reproduction=True,
            zero_threshold_sensitivity_changes=threshold_changes,
            threshold_values=[1e-7,ZERO_TOL,1e-5],max_kkt_residual=meta['max_kkt_residual'])
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):validate()
