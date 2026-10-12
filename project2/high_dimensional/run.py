"""Three growing-p Lasso cases: select a grid-optimal penalty, then evaluate it.

The selection target is expected coefficient-sign Hamming error. Known truth
is used for a simulation diagnostic, not a deployable data-tuning procedure.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import warnings
import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CASES = {
    'easy': dict(id=0, ratio=4, q=5, ic=0.0),
    'correlated': dict(id=2, ratio=4, q=5, ic=0.9),
    'difficult': dict(id=1, ratio=8, q=10, ic=0.9),
}
NS = [50, 80, 120, 180, 270, 400]
MULTIPLIERS = np.geomspace(0.15, 8.0, 49)[::-1]
SEED = 20261012
ZERO_TOL = 1e-6


def generate(case, n, batch, repeat):
    spec = CASES[case]; p = spec['ratio']*n; q = spec['q']; g = spec['ic']
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 610, spec['id'], n, batch, repeat]))
    x = rng.normal(size=(n, p))
    if g:
        # One inactive proxy; all remaining inactive predictors are independent.
        x[:, q] = g/q*x[:, :q].sum(axis=1) + np.sqrt(1-g*g/q)*x[:, q]
    x -= x.mean(axis=0)
    x /= np.sqrt(np.mean(x*x, axis=0))
    x = np.asfortranarray(x)
    beta = np.r_[np.ones(q), np.zeros(p-q)]
    noise = rng.normal(size=n); noise -= noise.mean()
    y = x[:, :q].sum(axis=1) + noise
    active = x[:, :q]
    c11 = active.T@active/n
    v = np.linalg.solve(c11, np.ones(q))
    sample_ic = float(np.max(np.abs(x[:, q:].T@(active@v)/n)))
    return x, y, beta, sample_ic, float(np.linalg.eigvalsh(c11).min())


def path_metrics(x, y, beta, alphas):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        returned, coef, gaps = lasso_path(x, y, alphas=alphas, tol=1e-10, max_iter=50000)
    assert np.allclose(returned, alphas)
    q = int(np.count_nonzero(beta))
    signs = np.where(np.abs(coef)>ZERO_TOL, np.sign(coef), 0)
    errors = np.sum(signs != np.sign(beta)[:, None], axis=0)
    fp = np.sum(signs[q:] != 0, axis=0)
    fn = np.sum(signs[:q] == 0, axis=0)
    wrong = np.sum(signs[:q] == -1, axis=0)
    assert np.array_equal(errors, fp+fn+wrong)
    gradient = x.T@(y[:, None]-x@coef)/len(y)
    active = np.abs(coef)>1e-8
    residual = np.maximum(np.abs(gradient)-alphas[None, :], 0)
    residual[active] = np.abs(gradient-alphas[None, :]*np.sign(coef))[active]
    kkt = float(residual.max())
    assert kkt < 2e-5, f'KKT residual {kkt}'
    return dict(errors=errors, fp=fp, fn=fn, wrong=wrong,
        coefficient_error=np.sum((coef-beta[:, None])**2, axis=0),
        kkt=kkt, dual_gap=float(np.max(np.abs(gaps))))


def choose(errors, coefficient_error):
    risk = np.mean(errors, axis=0)
    mse = np.mean(coefficient_error, axis=0)
    # Prefer smaller coefficient error when several penalties have equal
    # empirical sign risk; a final exact tie chooses the smaller penalty.
    return min(range(len(risk)), key=lambda i: (risk[i], mse[i], -i))


def wilson(k, total):
    z=1.95996398454; p=k/total; den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return float(max(0, center-radius)), float(min(1, center+radius))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(tuning_repeats=80, evaluation_repeats=160, output='results', cases=None):
    out=ROOT/output; out.mkdir(parents=True, exist_ok=True)
    summaries=[]; curves=[]; runs=[]; start=time.time(); max_kkt=0.0
    selected_cases=list(CASES) if cases is None else cases
    assert selected_cases and len(set(selected_cases))==len(selected_cases)
    assert all(case in CASES for case in selected_cases)
    if set(selected_cases)!=set(CASES):
        previous=json.loads((out/'metadata.json').read_text())
        assert previous['tuning_repeats']==tuning_repeats
        assert previous['evaluation_repeats']==evaluation_repeats
        assert previous['sample_sizes']==NS
        for filename,target in [('summary.csv',summaries),('grid.csv',curves),('runs.csv',runs)]:
            with (out/filename).open() as f:
                target.extend(r for r in csv.DictReader(f) if r['case'] not in selected_cases)
        max_kkt=max(float(r['max_kkt_residual']) for r in runs)
    for case, spec in CASES.items():
        if case not in selected_cases:continue
        for n in NS:
            p=spec['ratio']*n; q=spec['q']
            alphas=MULTIPLIERS*np.sqrt(2*np.log(p)/n)
            data={}
            for batch, repeats in [(0, tuning_repeats), (1, evaluation_repeats)]:
                metrics=[]; sample_ic=[]; eigenmin=[]
                for repeat in range(repeats):
                    x,y,beta,ic,eig=generate(case,n,batch,repeat)
                    m=path_metrics(x,y,beta,alphas)
                    metrics.append(m); sample_ic.append(ic); eigenmin.append(eig)
                    max_kkt=max(max_kkt,m['kkt'])
                    runs.append(dict(case=case,n=n,p=p,q=q,batch='tuning' if batch==0 else 'evaluation',
                        repeat=repeat,sample_ic=ic,sample_margin=1-ic,active_eigenmin=eig,
                        max_kkt_residual=m['kkt']))
                data[batch]={key:np.array([m[key] for m in metrics]) for key in
                             ['errors','fp','fn','wrong','coefficient_error']}
                data[batch]['sample_ic']=np.array(sample_ic)
                data[batch]['active_eigenmin']=np.array(eigenmin)
            tune=data[0]; evaluate=data[1]
            selected=choose(tune['errors'],tune['coefficient_error'])
            rng=np.random.default_rng(np.random.SeedSequence([SEED,611,spec['id'],n]))
            bootstrap=[]
            for _ in range(500):
                inds=rng.integers(tuning_repeats,size=tuning_repeats)
                bootstrap.append(choose(tune['errors'][inds],tune['coefficient_error'][inds]))
            boot_lambdas=2*n*alphas[np.array(bootstrap)]
            risk=tune['errors'].mean(axis=0)
            ties=np.flatnonzero(np.isclose(risk,risk[selected],rtol=0,atol=1e-12))
            successes=int(np.sum(evaluate['errors'][:,selected]==0))
            lo,hi=wilson(successes,evaluation_repeats)
            ss=dict(case=case,n=n,p=p,q=q,population_ic=spec['ic'],population_eta=1-spec['ic'],
                selected_grid_index=selected,selected_multiplier=float(MULTIPLIERS[selected]),
                selected_alpha=float(alphas[selected]),selected_lambda=float(2*n*alphas[selected]),
                lambda_bootstrap_low=float(np.quantile(boot_lambdas,.025)),
                lambda_bootstrap_high=float(np.quantile(boot_lambdas,.975)),
                minimum_risk_lambda_low=float(2*n*alphas[ties[-1]]),
                minimum_risk_lambda_high=float(2*n*alphas[ties[0]]),
                selected_at_grid_boundary=bool(selected in (0,len(alphas)-1)),
                tuning_mean_sign_errors=float(risk[selected]),
                evaluation_mean_sign_errors=float(evaluate['errors'][:,selected].mean()),
                evaluation_false_positives=float(evaluate['fp'][:,selected].mean()),
                evaluation_false_negatives=float(evaluate['fn'][:,selected].mean()),
                evaluation_wrong_signs=float(evaluate['wrong'][:,selected].mean()),
                evaluation_success_count=successes,evaluation_repeats=evaluation_repeats,
                evaluation_success=successes/evaluation_repeats,success_ci_low=lo,success_ci_high=hi,
                evaluation_any_grid_success=float(np.mean(np.any(evaluate['errors']==0,axis=1))),
                sample_ic_median=float(np.median(evaluate['sample_ic'])),
                sample_margin_median=float(np.median(1-evaluate['sample_ic'])),
                sample_ic_holds_fraction=float(np.mean(evaluate['sample_ic']<1)),
                tuning_repeats=tuning_repeats)
            summaries.append(ss)
            for i,alpha in enumerate(alphas):
                curves.append(dict(case=case,n=n,p=p,q=q,grid_index=i,multiplier=float(MULTIPLIERS[i]),
                    alpha=float(alpha),lambda_paper=float(2*n*alpha),selected=bool(i==selected),
                    tuning_mean_sign_errors=float(risk[i]),
                    tuning_mean_coefficient_error=float(tune['coefficient_error'][:,i].mean()),
                    tuning_success=float(np.mean(tune['errors'][:,i]==0)),
                    evaluation_mean_sign_errors=float(evaluate['errors'][:,i].mean()),
                    evaluation_success=float(np.mean(evaluate['errors'][:,i]==0))))
            arrays={f'{"tuning" if b==0 else "evaluation"}_{key}':value
                for b,values in data.items() for key,value in values.items()}
            np.savez_compressed(out/f'{case}_n{n}.npz',alphas=alphas,multipliers=MULTIPLIERS,
                selected_index=selected,bootstrap_indices=bootstrap,**arrays)
            print(f'{case} n={n} p={p}: lambda={ss["selected_lambda"]:.2f}, '
                  f'alpha={ss["selected_alpha"]:.3f}, errors={ss["evaluation_mean_sign_errors"]:.3f}, '
                  f'recovery={ss["evaluation_success"]:.3f}, sample IC holds={ss["sample_ic_holds_fraction"]:.3f}',flush=True)
    order={case:i for i,case in enumerate(CASES)}
    for collection in [summaries,curves,runs]:
        collection.sort(key=lambda r:(order[r['case']],int(r['n'])))
    write_csv(out/'summary.csv',summaries); write_csv(out/'grid.csv',curves); write_csv(out/'runs.csv',runs)
    import sklearn
    metadata=dict(cases=CASES,sample_sizes=NS,seed=SEED,zero_tolerance=ZERO_TOL,
        tuning_repeats=tuning_repeats,evaluation_repeats=evaluation_repeats,
        datasets=len(runs),grid_size=len(MULTIPLIERS),grid_multipliers=MULTIPLIERS.tolist(),
        penalty_grid='alpha=k*sqrt(2*log(p)/n); paper lambda=2*n*alpha',
        objective='Minimize tuning mean coefficient-sign Hamming error; break ties by coefficient squared error, then smaller alpha',
        interpretation='Known-truth simulation diagnostic over a finite grid; not practical CV tuning or a unique population optimum',
        noise='Original iid N(0,1), independent of predictors; centered to remove intercept',
        signal='First q coefficients are +1, all remaining coefficients zero',
        correlated_proxy='X_(q+1)=0.9/q*sum(X_active)+sqrt(1-0.9^2/q)*Z; all other variables independent',
        eta='Population eta=1-IC; sample margin=1-max(abs(C21 inv(C11) sign(beta_S))) checked on every realization',
        uncertainty='500 bootstrap resamples of tuning replicates; lambda range measures tuning variability',
        independence='Separate seed streams for tuning and evaluation; evaluation is not used to choose alpha',
        solver_tolerance=1e-10,max_iterations=50000,kkt_limit=2e-5,
        max_kkt_residual=max_kkt,sklearn=sklearn.__version__,numpy=np.__version__,seconds=time.time()-start,
        computed_cases_this_run=selected_cases,
        scope='Three settings: independent, matched correlated, and larger correlated')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tuning-repeats',type=int,default=80)
    parser.add_argument('--evaluation-repeats',type=int,default=160)
    parser.add_argument('--output',default='results')
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=None)
    args=parser.parse_args()
    with threadpool_limits(1): run(args.tuning_repeats,args.evaluation_repeats,args.output,args.cases)
