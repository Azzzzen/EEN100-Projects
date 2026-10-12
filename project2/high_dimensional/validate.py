"""Audit saved statistics and independently certify predetermined fitted models."""
from pathlib import Path
import csv
import json
import numpy as np
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits
from run import CASES,NS,generate,choose,path_metrics,wilson

ROOT=Path(__file__).resolve().parent


def validate():
    rows=list(csv.DictReader((ROOT/'results/summary.csv').open()))
    meta=json.loads((ROOT/'results/metadata.json').read_text())
    assert len(rows)==len(CASES)*len(NS)
    boundary=[]; solver_checks=[]
    for case,spec in CASES.items():
        # Independent covariance construction, rather than using the factor
        # generator, certifies the analytic IC and positive definiteness.
        q=spec['q'];g=spec['ic'];cov=np.eye(q+2)
        cov[q,:q]=g/q;cov[:q,q]=g/q
        ic=np.max(np.abs(cov[q:,:q]@np.linalg.solve(cov[:q,:q],np.ones(q))))
        assert np.isclose(ic,g)
        assert np.isclose(np.linalg.eigvalsh(cov).min(),1-g/np.sqrt(q))
        assert np.linalg.eigvalsh(cov).min()>0
    for row in rows:
        case=row['case'];n=int(row['n']);p=int(row['p']);q=int(row['q'])
        assert p>n>q
        with np.load(ROOT/f'results/{case}_n{n}.npz') as d:
            idx=int(d['selected_index']);alpha=d['alphas'][idx]
            assert idx==choose(d['tuning_errors'],d['tuning_coefficient_error'])
            assert np.isclose(float(row['selected_lambda']),2*n*alpha)
            assert np.isclose(alpha,d['multipliers'][idx]*np.sqrt(2*np.log(p)/n))
            assert np.isclose(float(row['evaluation_mean_sign_errors']),d['evaluation_errors'][:,idx].mean())
            successes=int(np.sum(d['evaluation_errors'][:,idx]==0));total=d['evaluation_errors'].shape[0]
            assert successes==int(row['evaluation_success_count'])
            assert np.isclose(float(row['evaluation_success']),successes/total)
            lo,hi=wilson(successes,total)
            assert np.isclose(float(row['success_ci_low']),lo)
            assert np.isclose(float(row['success_ci_high']),hi)
            assert np.all(d['evaluation_errors']==d['evaluation_fp']+d['evaluation_fn']+d['evaluation_wrong'])
            assert np.isclose(float(row['sample_ic_holds_fraction']),np.mean(d['evaluation_sample_ic']<1))
            if idx in (0,len(d['alphas'])-1):boundary.append([case,n])
            if n not in (NS[0],NS[-1]):continue
            x,y,beta,sample_ic,_=generate(case,n,1,0)
            assert np.allclose(x.mean(axis=0),0,atol=1e-12)
            assert np.allclose(np.mean(x*x,axis=0),1)
            c=x.T@x/n
            direct=np.max(np.abs(c[q:,:q]@np.linalg.solve(c[:q,:q],np.ones(q))))
            assert np.isclose(sample_ic,direct)
            assert np.isclose(sample_ic,d['evaluation_sample_ic'][0])
            fit=Lasso(alpha=float(alpha),fit_intercept=False,tol=1e-11,max_iter=30000).fit(x,y)
            coef=fit.coef_;active=np.flatnonzero(np.abs(coef)>1e-6)
            if len(active):
                xa=x[:,active]
                closed=np.linalg.solve(xa.T@xa/n,xa.T@y/n-alpha*np.sign(coef[active]))
                assert np.allclose(coef[active],closed,atol=2e-7)
            gradient=x.T@(y-x@coef)/n
            inactive=np.setdiff1d(np.arange(p),active)
            assert np.max(np.abs(gradient[inactive]))<=alpha+2e-7
            signs=np.where(abs(coef)>1e-6,np.sign(coef),0)
            error=int(np.sum(signs!=np.sign(beta)))
            assert error==d['evaluation_errors'][0,idx]
            m=path_metrics(x,y,beta,np.array([alpha]))
            assert error==m['errors'][0]
            solver_checks.append(dict(case=case,n=n,sign_errors=error,active_count=len(active),
                active_closed_form_verified=True,inactive_kkt_verified=True))
    assert not boundary,f'Grid expansion needed: {boundary}'
    report=dict(cells_checked=len(rows),datasets=meta['datasets'],p_exceeds_n_in_all_cells=True,
        analytic_ic_and_covariance_verified=True,independent_tuning_and_evaluation=True,
        saved_counts_and_intervals_verified=True,sample_ic_direct_gram_verified=True,
        selected_grid_boundaries=boundary,predetermined_model_certificates=solver_checks,
        maximum_path_kkt_residual=meta['max_kkt_residual'])
    (ROOT/'results/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(1):validate()
