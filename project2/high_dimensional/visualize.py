"""English PNG figures for grid-optimal high-dimensional Lasso penalties."""
from pathlib import Path
import csv
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
COLORS={'easy':'#32688C','correlated':'#6C8060','difficult':'#9B4E5B'}
LABELS={'easy':'A  Independent: p = 4n, q = 5, population eta = 1',
        'correlated':'B  Correlated: p = 4n, q = 5, population eta = 0.1',
        'difficult':'C  Difficult: p = 8n, q = 10, population eta = 0.1'}


def save(fig,name):
    for item in fig.findobj(match=matplotlib.text.Text):
        assert not re.search(r'[\u4e00-\u9fff]',item.get_text())
    fig.savefig(ROOT/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make():
    rows=list(csv.DictReader((ROOT/'results/summary.csv').open()))
    meta=json.loads((ROOT/'results/metadata.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    contract=dict(conclusion='Correlation and sparsity change the useful penalty and held-out sign recovery as n grows.',
        panels=['Grid-optimal paper lambda versus n','Independent sign recovery versus n',
                'Tuning sign-error surfaces for all three cases'],
        archetype='quantitative grid',backend='Python / matplotlib',export='English PNG only, 220 dpi',
        uncertainty='Lambda: 500 tuning bootstrap resamples; recovery: 95% Wilson intervals',
        review_risks=['Known-truth selection diagnostic, not practical CV',
                      'Matched A/B compare correlations; B/C jointly change p and q',
                      'Population eta is not the actual sample margin',
                      'Grid optimum and finite repetitions do not prove a population optimum or asymptotic rate'])
    (ROOT/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    for case in COLORS:
        rr=[r for r in rows if r['case']==case]; ns=np.array([int(r['n']) for r in rr])
        color=COLORS[case]
        axes[0].plot(ns,[float(r['selected_lambda']) for r in rr],'o-',color=color,lw=2,label=LABELS[case])
        axes[0].fill_between(ns,[float(r['lambda_bootstrap_low']) for r in rr],
            [float(r['lambda_bootstrap_high']) for r in rr],color=color,alpha=.15)
        axes[1].plot(ns,[100*float(r['evaluation_success']) for r in rr],'o-',color=color,lw=2)
        axes[1].fill_between(ns,[100*float(r['success_ci_low']) for r in rr],
            [100*float(r['success_ci_high']) for r in rr],color=color,alpha=.15)
    axes[0].set(title='Grid-optimal penalty',xlabel='Sample size n',ylabel=r'Paper penalty $\lambda^* = 2n\alpha^*$')
    axes[1].set(title='Independent evaluation',xlabel='Sample size n',ylabel='Correct sign recovery (%)',ylim=(-3,105))
    for ax in axes:
        ax.set_xticks(meta['sample_sizes']); ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncols=1,fontsize=10)
    fig.supxlabel(f'{meta["tuning_repeats"]} tuning + {meta["evaluation_repeats"]} independent evaluation runs per point\n'
        'Shading: left, tuning bootstrap range; right, 95% Wilson intervals',fontsize=10)
    save(fig,'01_optimal_lambda')

    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,case in zip(axes,COLORS):
        ns=meta['sample_sizes']; risks=[]; chosen=[]
        for n in ns:
            with np.load(ROOT/f'results/{case}_n{n}.npz') as d:
                risks.append(d['tuning_errors'].mean(axis=0)[::-1])
                chosen.append(d['multipliers'][int(d['selected_index'])])
                multipliers=d['multipliers'][::-1]
        risks=np.array(risks)
        # Color saturation is explicit: 10 or more sign errors share the upper color.
        mesh=ax.pcolormesh(multipliers,ns,risks,shading='nearest',cmap='Blues',vmin=0,vmax=10)
        ax.plot(chosen,ns,'o-',color=COLORS[case],lw=2,markeredgecolor='white',label='Selected multiplier')
        ax.set(xscale='log',
            title={'easy':'A  Independent','correlated':'B  Correlated','difficult':'C  Difficult'}[case],
            xticks=[.15,.5,1,2,4,8],xticklabels=['0.15','0.5','1','2','4','8'],yticks=ns)
        ax.tick_params(axis='x',labelsize=10)
        ax.minorticks_off()
        ax.legend(loc='upper left',fontsize=9,frameon=True,facecolor='white',framealpha=.9)
    axes[0].set_ylabel('Sample size n')
    cbar=fig.colorbar(mesh,ax=axes,shrink=.9,extend='max')
    cbar.set_label('Mean sign errors (colors saturate at 10)',fontsize=10)
    fig.supxlabel(r'Multiplier $k$ in $\alpha=k\sqrt{2\log(p)/n}$ (log scale)'
        '\nSelection minimizes mean sign errors; ties use coefficient squared error.',fontsize=11)
    save(fig,'02_penalty_tradeoff')
    qa=dict(language='English-only text asserted',export='Two PNG files only',
        uncertainty=contract['uncertainty'],data='results/summary.csv and per-cell NPZ arrays',
        scope='Three high-dimensional cases; low-dimensional outputs preserved',
        visual_review='Pending visual inspection',integrity='Underlying numerical values unchanged')
    (ROOT/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print('Two English PNG figures generated.')


if __name__=='__main__':make()
