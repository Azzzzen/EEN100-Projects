"""Finite-sample recovery with violated IC: a paired-noise counterexample.

Illustration selection is explicit: first fixed-penalty success among the
existing r=0.7, n=50 replicates. Frequencies use every run in each setting.
"""
from pathlib import Path
import csv
import json
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path,lars_path
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from run import generate,make_seed,sign_success,kkt_residual

ROOT=Path(__file__).resolve().parent
BLUE='#32688C'; RED='#9B4E5B'; GRAY='#DCE3E8'


def save(fig,name):
    for item in fig.findobj(match=matplotlib.text.Text):
        assert not re.search(r'[\u4e00-\u9fff]',item.get_text())
    fig.savefig(ROOT/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def compute_example():
    rows=list(csv.DictReader((ROOT/'results/runs.csv').open()))
    target=next(r for r in rows if r['scenario']=='low_fail' and int(r['n'])==50
                and r['fixed_success']=='True')
    n=50;repeat=int(target['repeat']);alpha=float(target['alpha'])
    x,y,beta,_,_,_=generate('low_fail',n,make_seed('low_fail',n,repeat))
    noise=y-x@beta;c=x.T@x/n;active=x[:,:2];c11=c[:2,:2]
    signed_ic=float(c[2,:2]@np.linalg.solve(c11,np.ones(2)))
    residual_noise=noise-active@np.linalg.solve(c11,active.T@noise/n)
    w=float(x[:,2]@residual_noise/n)
    assert signed_ic>1 and w<0
    variants=[]
    for direction in [1,-1]:
        yy=x@beta+direction*noise
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            _,coef,_=lasso_path(x,yy,alphas=[alpha],tol=1e-12,max_iter=30000)
            path_alpha,_,path_coef=lars_path(x,yy,method='lasso',alpha_min=0,eps=1e-12)
        candidate=beta[:2]+np.linalg.solve(c11,active.T@(direction*noise)/n-alpha*np.ones(2))
        assert np.all(candidate>0)
        candidate_residual=yy-active@candidate
        direct=float(x[:,2]@candidate_residual/n)
        decomposition=alpha*signed_ic+direction*w
        assert np.isclose(direct,decomposition,atol=1e-12)
        candidate_kkt=bool(abs(direct)<=alpha)
        success=bool(sign_success(coef,beta)[0])
        assert success==candidate_kkt
        states=np.c_[path_coef,(path_coef[:,:-1]+path_coef[:,1:])/2]
        full_success=bool(sign_success(states,beta).any())
        certificate=kkt_residual(x,yy,coef,[alpha]);assert certificate<1e-9
        variants.append(dict(noise_sign=direction,estimated_beta=coef[:,0].tolist(),
            candidate_active_beta=candidate.tolist(),candidate_residual_correlation=direct,
            noise_component=direction*w,shrinkage_component=alpha*signed_ic,
            fixed_sign_recovery=success,any_path_recovery=full_success,kkt_residual=certificate))
    assert variants[0]['fixed_sign_recovery'] and not variants[1]['fixed_sign_recovery']
    assert variants[0]['any_path_recovery'] and not variants[1]['any_path_recovery']
    bounds=[-w/(signed_ic+1),-w/(signed_ic-1)]
    # In this example the active candidate remains positive throughout the
    # entire inactive-KKT interval. Endpoints are included mathematically.
    for a in bounds:
        b=beta[:2]+np.linalg.solve(c11,active.T@noise/n-a*np.ones(2))
        assert np.all(b>0)
    result=dict(n=n,repeat=repeat,r=.7,true_beta=beta.tolist(),alpha=alpha,
        paper_lambda=2*n*alpha,population_ic=7/6,sample_ic=signed_ic,
        selection_rule='First fixed-penalty success in existing n=50, r=0.7 replicates; illustrative, not representative',
        comparison='Same X, beta and alpha; compare original centered noise with its negative',
        noise_component=w,shrinkage_component=alpha*signed_ic,
        successful_alpha_interval=bounds,variants=variants)
    (ROOT/'results/finite_sample_explanation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def make():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    contract=dict(conclusion='Violation of sample IC can coexist with correct finite-sample selection because noise offsets shrinkage bias.',
        archetype='quantitative grid',backend='Python / matplotlib',export='English PNG only',
        panels=['Paired-noise coefficient fits with the same X',
                'Inactive KKT decomposition for the candidate with beta3=0',
                'Fixed-penalty recovery versus full-path existence across all replicates'],
        review_risks=['Explicitly selected counterexample; use all replicates for frequencies',
                      'Candidate residual is not the residual of the unconstrained failed fit',
                      'Full-path existence is truth-based and not a practical tuning method',
                      'Finite Monte Carlo results do not prove an exact asymptotic limit'])
    (ROOT/'finite_sample_figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    ex=compute_example();alpha=ex['alpha'];bias=ex['shrinkage_component']
    fig,axes=plt.subplots(1,2,figsize=(12.0,4.8),layout='constrained',
        gridspec_kw={'width_ratios':[1.1,1]})
    ax=axes[0];pos=np.arange(3)
    ax.bar(pos-.24,[1,1,0],width=.23,color=GRAY,label='True coefficient')
    for offset,v,color,label in zip([0,.24],ex['variants'],[BLUE,RED],
            ['Noise epsilon: correct','Noise -epsilon: incorrect']):
        ax.bar(pos+offset,v['estimated_beta'],width=.23,color=color,label=label)
        for j,val in enumerate(v['estimated_beta']):
            ax.text(j+offset,val+.025,f'{val:.3f}',ha='center',fontsize=10,color=color)
    ax.set(xticks=pos,xticklabels=['X1 (active)','X2 (active)','X3 (inactive)'],
        ylabel='Estimated / true coefficient',ylim=(-.03,1.4),title='A  Same X and penalty, different noise')
    ax.axhline(0,color='#657682',lw=.7);ax.legend(loc='upper right',fontsize=9)
    ax=axes[1]
    ax.axvspan(0,alpha,color=BLUE,alpha=.07)
    ax.axvline(alpha,color='#243746',ls='--',lw=1.5)
    ax.axvline(bias,color='#94A1AA',ls=':',lw=1)
    for y,v,color in zip([1,0],ex['variants'],[BLUE,RED]):
        total=v['candidate_residual_correlation'];component=v['noise_component']
        ax.plot([0,bias],[y,y],color=GRAY,lw=9,solid_capstyle='butt')
        ax.annotate('',xy=(total,y),xytext=(bias,y),
            arrowprops={'arrowstyle':'->','color':color,'lw':2.5,'mutation_scale':16})
        ax.plot(total,y,'o',color=color,ms=7,zorder=5)
        ax.text(.18,y+.17,f'Noise term: {component:+.4f}',color=color,fontsize=11)
        ax.text(total,y-.18,f'g3 = {total:.4f}',ha='center',color=color,fontsize=11)
    ax.text(alpha-.018,1.52,f'KKT upper limit\nalpha = {alpha:.4f}',ha='right',fontsize=10,color='#243746')
    ax.text(bias+.015,1.52,f'Shrinkage term\nalpha A = {bias:.4f}',ha='left',fontsize=10,color='#657682')
    ax.set(xlim=(0,.66),ylim=(-.42,1.85),yticks=[1,0],
        yticklabels=['epsilon','-epsilon'],xticks=[0,.2,.4,.6],
        xlabel='Residual correlation if beta3 is constrained to zero',
        title='B  Noise can offset shrinkage bias')
    fig.suptitle(f'Sample IC = {ex["sample_ic"]:.3f} > 1 in BOTH fits; n = 50; r = 0.7',fontsize=14)
    fig.supxlabel('Counterexample selected explicitly: first successful original replicate (repeat 1).\n'
        'Panel B uses the positive-active-coefficient candidate with beta3 = 0.',fontsize=10)
    save(fig,'03_finite_sample_counterexample')

    rows=list(csv.DictReader((ROOT/'results/summary.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    for ax,metric,title in zip(axes,['fixed','path'],
            ['A  Predetermined penalty','B  Any correct model on the full path']):
        for name,color,label in [('low_easy',BLUE,'IC holds: r = 0.3'),('low_fail',RED,'IC fails: r = 0.7')]:
            rr=[r for r in rows if r['scenario']==name];ns=[int(r['n']) for r in rr]
            ax.plot(ns,[100*float(r[f'{metric}_success']) for r in rr],'o-',color=color,lw=2,label=label)
            ax.fill_between(ns,[100*float(r[f'{metric}_ci_low']) for r in rr],
                [100*float(r[f'{metric}_ci_high']) for r in rr],color=color,alpha=.13)
        ax.set(xlabel='Sample size n',ylabel='Correct sign recovery (%)',
            ylim=(-3,105),title=title,xlim=(25,825),xticks=[50,200,400,600,800])
        ax.minorticks_off();ax.grid(axis='y',alpha=.15)
    axes[0].legend(loc='center right',fontsize=10)
    repeat_counts={int(r['repeats']) for r in rows}
    assert len(repeat_counts)==1
    repeats=repeat_counts.pop()
    fig.supxlabel(f'{repeats} independent runs per point; shaded bands: 95% Wilson intervals\n'
        r'Fixed penalty: $\alpha_n=0.75n^{-0.15}$; full-path existence uses known truth.',fontsize=10)
    save(fig,'04_finite_sample_recovery')
    qa=dict(language='English-only text checked',export='Two supplementary PNGs',
        numeric_checks='Paired fits: KKT <1e-9; direct residual agrees with decomposition; active candidates positive',
        path_checks='Original noise has a successful path point; reversed noise has none',
        source='Original runs.csv and summary.csv; no original simulation values changed',
        selection=ex['selection_rule'],visual_review='Pending inspection')
    (ROOT/'finite_sample_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(ex,indent=2))


if __name__=='__main__':
    with threadpool_limits(1):make()
