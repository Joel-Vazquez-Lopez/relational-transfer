"""Unmodified source-balanced B-D15 render function; only numpy imported."""
import numpy as np

def render(src,tgt,event,distractor,rng,latent=None):
    if latent is None:
        p=rng.uniform(-.25,.25,2).astype(np.float32)
        ang=rng.uniform(0,2*np.pi);u=np.array([np.cos(ang),np.sin(ang)],np.float32)
        speed=np.float32(rng.uniform(.10,.15));dist=np.float32(rng.uniform(.58,.85))
        tv=rng.normal(0,.006,2).astype(np.float32)
        dp=rng.uniform(-.9,.9,2).astype(np.float32)
        dv=rng.normal(0,.055,2).astype(np.float32)
        order=rng.permutation(3)
        latent=(p,u,speed,dist,tv,dp,dv,order)
    p,u,speed,dist,tv,dp,dv,order=latent
    source=np.stack([p+k*speed*u for k in range(4)])
    t0=p+(1 if event==0 else -1)*dist*u
    target=np.stack([t0+k*tv for k in range(4)])
    distract=np.stack([dp+k*dv for k in range(4)])
    traj=[]
    for ident,xy in ((src,source),(tgt,target),(distractor,distract)):
        row=np.concatenate([np.eye(6,dtype=np.float32)[ident],xy.reshape(-1)]).astype(np.float32)
        traj.append(row)
    scene=np.stack(traj)[order]
    s=scene[scene[:,:6].argmax(1)==src][0,6:].reshape(4,2)
    t=scene[scene[:,:6].argmax(1)==tgt][0,6:].reshape(4,2)
    g=float(np.linalg.norm(s[-1]-t[-1])-np.linalg.norm(s[0]-t[0]))
    assert np.sign(g)==(-1 if event==0 else 1) and abs(g)>.2,(src,tgt,event,distractor,g,source,target)
    return scene,latent
