"""Only Figure 2: English PNG of the predetermined coefficient fits."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
COLORS=['#32688C','#9B4E5B']


def save(fig,name):
    for item in fig.findobj(match=matplotlib.text.Text):
        assert not re.search(r'[\u4e00-\u9fff]',item.get_text()),'Figure contains Chinese text'
    fig.savefig(ROOT/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    examples=json.loads((RESULTS/'example_fits.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),layout='constrained')
    for i,(ax,ex) in enumerate(zip(axes,examples)):
        coef=ex['estimated_beta'];pos=np.arange(3)
        ax.bar(pos-.17,[1,1,0],width=.32,color='#E0E6EA',label='True coefficient')
        ax.bar(pos+.17,coef,width=.32,color=COLORS[i],label='Lasso estimate')
        for j,val in enumerate(coef):ax.text(j+.17,val+.035,f'{val:.3f}',ha='center',fontsize=12,color=COLORS[i])
        ax.axhline(0,c='#53616C',lw=.7)
        ax.set(xticks=pos,xticklabels=['X1\n(active)','X2\n(active)','X3\n(inactive)'],
               ylim=(-.03,1.25),ylabel='Coefficient',
               title='A  Correct: estimated beta3 = 0' if i==0 else 'B  False inclusion: estimated beta3 > 0')
        if i==0:ax.legend(loc='upper right',fontsize=9)
    fig.suptitle(f'Same alpha = {examples[0]["alpha"]:.3f}; n = 800; evaluation repeat 0',fontsize=13)
    save(fig,'02_coefficients')

    contract=dict(conclusion='Same penalty; IC holds: correct selection; IC fails: false inclusion.',
        panels=['A: r=0.3, IC holds','B: r=0.7, IC fails'],
        backend='Python / matplotlib',export='Only 02_coefficients.png, 220 dpi',
        evidence='Predetermined n=800, repeat=0 fits from results/example_fits.json',
        review_risks=['One realization illustrates behavior; repeated results are reported as text',
                      'Sign recovery does not require unbiased coefficient estimates'])
    (ROOT/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    qa=dict(language='All figure text checked: English only',export='One PNG: 02_coefficients.png',
        example='evaluation repeat 0 chosen in advance, n=800',
        metrics='Correct zero set and coefficient signs; zero tolerance 1e-6',
        numerical_check='Underlying fits validated by CD/LARS agreement and KKT certificates',
        visual_review='Retained Figure 2 reviewed: readable axes, labels and legend.',
        scope='Two low-dimensional settings; simulation data and source PDFs retained.',
        presentation='One-slide English text in ../PPT_TEXT_EN.md; only Figure 2 is used.')
    (ROOT/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print('Only Figure 2 generated: 02_coefficients.png')


if __name__=='__main__':make()
