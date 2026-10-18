"""Constructed, bounded sensitivity laws; no observed-effect power forecast."""
from pathlib import Path
import json, math
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]

def bounded_power(n, s, delta=.05, alpha=.025, draws=20000, seed=20261016):
    mass=s*s
    if not (abs(delta)<=mass<=1):raise ValueError('Law must have nonnegative probabilities')
    probs=[(mass-delta)/2,1-mass,(mass+delta)/2]
    counts=np.random.default_rng(seed).multinomial(n,probs,size=draws)
    means=(counts[:,2]-counts[:,0])/n
    variance=((counts[:,0]+counts[:,2])-n*means*means)/(n-1)
    stats=np.divide(np.sqrt(n)*means,np.sqrt(variance),out=np.zeros_like(means),where=variance>0)
    power=float(np.mean(abs(stats)>t.ppf(1-alpha/2,n-1)))
    return dict(n=n, nominal_sd_parameter=s, actual_sd=math.sqrt(mass-delta*delta),mean=delta,
                probabilities=probs,support=[-1,0,1],alpha_two_sided=alpha,draws=draws,
                rejection_rate=power,mc_standard_error=math.sqrt(power*(1-power)/draws),
                type='CONSTRUCTED_SENSITIVITY_NOT_FORECAST',seed=seed)

def main():
    out=[bounded_power(n,s) for n,s in [(238,.25),(951,.5)]]
    (ROOT/'results/bounded_power_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
