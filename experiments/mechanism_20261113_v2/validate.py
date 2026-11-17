"""Synthetic mechanism controls, never counted as scientific model responses."""
import math
import numpy as np
from score import distribution_metrics,decode,quantiles
from common import *


def main():
    assert distribution_metrics([1,0],[0,1],['x','y'])==(1.,math.log(2))
    assert distribution_metrics([.3,.7],[.3,.7],[0,10])==(0.,0.)
    assert abs(distribution_metrics([1,0,0],[0,0,1],[0,1,10])[0]-1)<1e-12
    assert abs(distribution_metrics([1,0,0],[0,1,0],[10,0,1])[0]-1)<1e-12
    rng=np.random.default_rng(20261113)
    # Independently split unit mass across sorted locations (1D optimal transport).
    for _ in range(200):
        x=np.sort(rng.choice(100,5,replace=False)).astype(float)
        p=rng.dirichlet(np.ones(5));q=rng.dirichlet(np.ones(5));a=p.copy();b=q.copy();i=j=0;cost=0.
        while i<5 and j<5:
            mass=min(a[i],b[j]);cost+=mass*abs(x[i]-x[j]);a[i]-=mass;b[j]-=mass
            if a[i]<1e-14:i+=1
            if b[j]<1e-14:j+=1
        assert abs(distribution_metrics(p,q,x.tolist())[0]-cost/(x[-1]-x[0]))<1e-12
    row=dict(field='action',candidates=[0,1],task_type='goal')
    for text in ('{"action":true}','{"action":0.0}','{"action":2}','prefix {"action":0}','{"action":0,"x":1}'):
        assert decode(row,dict(response_text=text,finish_reason='stop'))[0] is None
    assert decode(row,dict(response_text='{"action":0}',finish_reason='length'))[1]=='truncated'
    assert decode(row,dict(response_text='{"action":0}',finish_reason='stop'))==(0,None)
    assert quantiles([None]*500)==dict(defined=0,undefined=500,empirical_range=None)
    assert quantiles([0.,0.])['empirical_range']==[0.,0.]
    # Missing mass replacement cannot move a bounded W1 by more than both missing fractions.
    for _ in range(200):
        p=rng.dirichlet(np.ones(3));q=rng.dirichlet(np.ones(3));u=rng.dirichlet(np.ones(3));v=rng.dirichlet(np.ones(3))
        r,s=rng.random(2);observed=distribution_metrics(p,q,[0,2,10])[0]
        latent=distribution_metrics((1-r)*p+r*u,(1-s)*q+s*v,[0,2,10])[0]
        assert max(0,observed-r-s)-1e-12<=latent<=min(1,observed+r+s)+1e-12
    assert seed('m','s','a','p',1)!=seed('m','s','b','p',1)
    assert seed('m','s','a','p',1)!=seed('m','s','pilot_a','p',1)
    save(R/'synthetic_validation.json',dict(status='passed',random_transport_controls=200,random_missing_bound_controls=200,
        checks=['independent_W1_transport','categorical_TV_and_JS','numeric_order',
        'strict_types_and_all_attempt_failures','all_missing_not_zero','missing_quality_bound',
        'independent_checkpoint_and_pilot_seed_streams'],scientific_model_answers=0))
    print('synthetic controls passed')


if __name__=='__main__':main()
