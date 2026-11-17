"""Unchanged scalar metrics from score_joint_reasoning_v2.py; provenance in freeze."""
import collections
import statistics

def w1(a,b):
    if not a or not b:return None
    ca,cb=collections.Counter(a),collections.Counter(b);xs=sorted(set(ca)|set(cb))
    fa=fb=value=0.
    for x,y in zip(xs,xs[1:]):
        fa+=ca[x]/len(a);fb+=cb[x]/len(b);value+=abs(fa-fb)*(y-x)
    return value

def w1_check(a,b):
    # Independent quantile-grid calculation, including unequal parse counts.
    if not a or not b:return None
    a=sorted(a);b=sorted(b);cuts=sorted({i/len(a) for i in range(len(a)+1)}|{i/len(b) for i in range(len(b)+1)})
    return sum((y-x)*abs(a[min(int(((x+y)/2)*len(a)),len(a)-1)]-b[min(int(((x+y)/2)*len(b)),len(b)-1)]) for x,y in zip(cuts,cuts[1:]))

def distance(a,b,n):
    point=w1(a,b)
    if point is None:return None,0.,1.
    assert abs(point-w1_check(a,b))<1e-10
    missing=2-(len(a)+len(b))/n
    return point,max(0.,point-missing),min(1.,point+missing)

def metrics(q,values,n):
    v={k:values[q[k]] for k in 'ABCD'}
    result={}
    for label,edges in [('E',['AB','CD']),('R',['AC','BD'])]:
        ds=[distance(v[a],v[b],n) if q[a]!=q[b] else (0.,0.,0.) for a,b in edges]
        result[label+'_conditional']=statistics.mean(d[0] for d in ds) if all(d[0] is not None for d in ds) else None
        for i,suffix in [(1,'low'),(2,'high')]:result[label+'_'+suffix]=statistics.mean(d[i] for d in ds)
    coeff=collections.Counter()
    for k,c in [('A',1),('B',-1),('C',-1),('D',1)]:coeff[q[k]]+=c
    lo=hi=0.
    for pid,c in coeff.items():
        vl=sum(values[pid])/n;vh=vl+(n-len(values[pid]))/n
        lo+=c*(vl if c>=0 else vh);hi+=c*(vh if c>=0 else vl)
    result['G_low']=0. if lo<=0<=hi else min(abs(lo),abs(hi))
    result['G_high']=min(max(abs(lo),abs(hi)),2*result['E_high'])
    result['G_conditional']=abs(sum(c*statistics.mean(values[pid]) for pid,c in coeff.items() if c)) if all(values[pid] for pid,c in coeff.items() if c) else None
    if result['G_conditional'] is not None and result['E_conditional'] is not None:assert result['G_conditional']<=2*result['E_conditional']+1e-10
    return result

def aggregate(rows,key=None):
    # First average variants within each condition pair; sensitivity weights pairs
    # equally within source or task, then groups equally.
    pairs=collections.defaultdict(list)
    for r in rows:pairs[r['pair_id']].append(r)
    cols=[k for k in rows[0] if k.startswith(('E_','G_','R_'))]
    pp=[]
    for rs in pairs.values():
        p={k:rs[0][k] for k in ['pair_id','source_id','task']}
        p.update({k:statistics.mean(r[k] for r in rs) if all(r[k] is not None for r in rs) else None for k in cols});pp.append(p)
    groups=collections.defaultdict(list)
    for p in pp:groups[p[key] if key else 'all'].append(p)
    return {k:statistics.mean(statistics.mean(p[k] for p in ps) for ps in groups.values()) if all(p[k] is not None for p in pp) else None for k in cols}
