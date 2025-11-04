# XR-side geometry + smoothing+latch for third-person camera boom.
# EXACT per-mesh BVHs for BOTH static and dynamic. No proxies, no fallback.

import math, time
from typing import List, Tuple, Dict, Any

# =========================
# Global state (XR process)
# =========================

# Static world-space triangles (single BVH)
_STAT_TRIS: List[Tuple[Tuple[float,float,float],
                       Tuple[float,float,float],
                       Tuple[float,float,float]]] = []
_STAT_BVH: List[dict] = []
_STAT_ROOT = -1

# Dynamic meshes: id -> {"tris":[(v0,v1,v2) in LOCAL], "bvh":[nodes], "root":int}
_DYN: Dict[str, Dict[str, Any]] = {}

# Per-operator smoothing/latch
_STATE: Dict[int, Dict[str, Any]] = {}

# Tunables (mirrors Blender-side intent)
_MIN_CAM_ABS       = 0.0006
_EXTRA_PULL_METERS = 0.25
_EXTRA_PULL_R_K    = 2.0
_LOS_EPS           = 1.0e-3
_PUSH_ITERS        = 1
_EPS               = 1.0e-9
_OUTWARD_RATE_MPS  = 10.0
_LATCH_HOLD_S      = 0.14
_LATCH_PAD_MIN     = 0.06
_LATCH_PAD_K       = 1.6

# =========================
# Small vector helpers
# =========================
def _sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _mul(a,k): return (a[0]*k, a[1]*k, a[2]*k)
def _dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def _len(a):   return math.sqrt(max(0.0, _dot(a,a)))
def _norm(a):
    L = _len(a)
    return (0.0,0.0,0.0) if L<=_EPS else (a[0]/L, a[1]/L, a[2]/L)

def _bmin3(a,b): return (min(a[0],b[0]), min(a[1],b[1]), min(a[2],b[2]))
def _bmax3(a,b): return (max(a[0],b[0]), max(a[1],b[1]), max(a[2],b[2]))

def _tri_bbox(t):
    v0,v1,v2=t
    mn=_bmin3(_bmin3(v0,v1),v2)
    mx=_bmax3(_bmax3(v0,v1),v2)
    return mn,mx

def _centroid(t):
    v0,v1,v2=t
    return ((v0[0]+v1[0]+v2[0])/3.0, (v0[1]+v1[1]+v2[1])/3.0, (v0[2]+v1[2]+v2[2])/3.0)

# nearest point on triangle (Ericson + fallbacks)
def _pt_tri_nearest(p, v0, v1, v2):
    ab=_sub(v1,v0); ac=_sub(v2,v0); ap=_sub(p,v0)
    d1=_dot(ab,ap); d2=_dot(ac,ap)
    if d1<=0.0 and d2<=0.0: return v0
    bp=_sub(p,v1); d3=_dot(ab,bp); d4=_dot(ac,bp)
    if d3>=0.0 and d4<=d3: return v1
    vc = d1*d4 - d3*d2
    if vc<=0.0 and d1>=0.0 and d3<=0.0:
        v = d1/(d1 - d3); return _add(v0,_mul(ab,v))
    cp=_sub(p,v2); d5=_dot(ab,cp); d6=_dot(ac,cp)
    if d6>=0.0 and d5<=d6: return v2
    vb = d5*d2 - d1*d6
    if vb<=0.0 and d2>=0.0 and d6<=0.0:
        w = d2/(d2 - d6); return _add(v0,_mul(ac,w))
    va = d3*d6 - d5*d4
    if va<=0.0 and (d4-d3)>=0.0 and (d5-d6)>=0.0:
        w = (d4 - d3)/((d4 - d3) + (d5 - d6))
        return _add(v1,_mul(_sub(v2,v1), w))
    # bary fallback
    v0v1=_sub(v1,v0); v0v2=_sub(v2,v0)
    d00=_dot(v0v1,v0v1); d01=_dot(v0v1,v0v2); d11=_dot(v0v2,v0v2)
    v0p=_sub(p,v0); d20=_dot(v0p,v0v1); d21=_dot(v0p,v0v2)
    denom = max(_EPS, d00*d11 - d01*d01)
    v = (d11*d20 - d01*d21)/denom
    w = (d00*d21 - d01*d20)/denom
    u = 1.0 - v - w
    return (v0[0]*u + v1[0]*v + v2[0]*w,
            v0[1]*u + v1[1]*v + v2[1]*w,
            v0[2]*u + v1[2]*v + v2[2]*w)

# =========================
# Ray / BVH helpers
# =========================
def _ray_aabb(o,d,mn,mx,tmax):
    tmin=0.0; tmax_=tmax
    for i in (0,1,2):
        di=d[i]
        if abs(di)<_EPS:
            if o[i]<mn[i] or o[i]>mx[i]: return None
        else:
            inv=1.0/di; t0=(mn[i]-o[i])*inv; t1=(mx[i]-o[i])*inv
            if inv<0.0: t0,t1=t1,t0
            if t0>tmin: tmin=t0
            if t1<tmax_: tmax_=t1
            if tmax_<tmin: return None
    return (tmin,tmax_)

def _ray_tri(o,d,v0,v1,v2):
    e1=_sub(v1,v0); e2=_sub(v2,v0)
    p=(d[1]*e2[2]-d[2]*e2[1], d[2]*e2[0]-d[0]*e2[2], d[0]*e2[1]-d[1]*e2[0])
    det=_dot(e1,p)
    if abs(det)<1.0e-10: return None
    inv=1.0/det
    t=_sub(o,v0)
    u=_dot(t,p)*inv
    if u<0.0 or u>1.0: return None
    q=(t[1]*e1[2]-t[2]*e1[1], t[2]*e1[0]-t[0]*e1[2], t[0]*e1[1]-t[1]*e1[0])
    v=_dot(d,q)*inv
    if v<0.0 or u+v>1.0: return None
    tt=_dot(e2,q)*inv
    return tt if tt>=0.0 else None

def _bvh_build_array(tris):
    nodes=[]
    def rec(idxs):
        mn=(+1e30,+1e30,+1e30); mx=(-1e30,-1e30,-1e30)
        for i in idxs:
            a,b=_tri_bbox(tris[i]); mn=_bmin3(mn,a); mx=_bmax3(mx,b)
        node={"bmin":mn,"bmax":mx,"left":-1,"right":-1,"tris":None}
        if len(idxs)<=16:
            node["tris"]=idxs; nodes.append(node); return len(nodes)-1
        ext=(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2])
        axis=0 if ext[0]>=ext[1] and ext[0]>=ext[2] else (1 if ext[1]>=ext[2] else 2)
        cents=[(i,_centroid(tris[i])[axis]) for i in idxs]
        cents.sort(key=lambda t:t[1])
        mid=len(cents)//2
        L=[i for (i,_) in cents[:mid]] or idxs[:len(idxs)//2]
        R=[i for (i,_) in cents[mid:]] or idxs[len(idxs)//2:]
        nodes.append(node); me=len(nodes)-1
        nodes[me]["left"]=rec(L); nodes[me]["right"]=rec(R)
        return me
    root = rec(list(range(len(tris)))) if tris else -1
    return nodes, root

def _raycast_bvh(tris, nodes, root, o, d, tmax, best=None):
    if root<0: return best
    stack=[root]
    while stack:
        n = nodes[stack.pop()]
        box=_ray_aabb(o,d,n["bmin"],n["bmax"], tmax if best is None else min(tmax,best))
        if box is None: continue
        if n["tris"] is not None:
            for i in n["tris"]:
                t=_ray_tri(o,d,*tris[i])
                if t is not None and 0.0<=t<= (best if best is not None else tmax):
                    best=t
        else:
            stack.append(n["left"]); stack.append(n["right"])
    return best

def _nearest_bvh(tris, nodes, root, p, best_d2):
    if root<0: return best_d2
    stack=[root]
    while stack:
        n=nodes[stack.pop()]
        mn=n["bmin"]; mx=n["bmax"]
        dx=0.0 if mn[0]<=p[0]<=mx[0] else (mn[0]-p[0] if p[0]<mn[0] else p[0]-mx[0])
        dy=0.0 if mn[1]<=p[1]<=mx[1] else (mn[1]-p[1] if p[1]<mn[1] else p[1]-mx[1])
        dz=0.0 if mn[2]<=p[2]<=mx[2] else (mn[2]-p[2] if p[2]<mn[2] else p[2]-mx[2])
        if (dx*dx+dy*dy+dz*dz) > best_d2 + 1.0e-12:
            continue
        if n["tris"] is not None:
            for i in n["tris"]:
                v0,v1,v2 = tris[i]
                q=_pt_tri_nearest(p, v0,v1,v2)
                d=_sub(p,q); d2=_dot(d,d)
                if d2<best_d2: best_d2=d2
        else:
            stack.append(n["left"]); stack.append(n["right"])
    return best_d2

# =========================
# Dynamic transforms (4x4)
# =========================
def _mat_split(M):
    return (M[0:4], M[4:8], M[8:12], M[12:16])

def _mat3(M):
    return ((M[0][0],M[0][1],M[0][2]),
            (M[1][0],M[1][1],M[1][2]),
            (M[2][0],M[2][1],M[2][2]))

def _mat_mul_vec3(M3, v):
    return (M3[0][0]*v[0]+M3[0][1]*v[1]+M3[0][2]*v[2],
            M3[1][0]*v[0]+M3[1][1]*v[1]+M3[1][2]*v[2],
            M3[2][0]*v[0]+M3[2][1]*v[1]+M3[2][2]*v[2])

def _mat_apply(M, v):
    return (M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2]+M[0][3],
            M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2]+M[1][3],
            M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]+M[2][3])

def _mat_inv(M):
    # affine inverse
    m=_mat_split(M); A=_mat3(m); tx=m[0][3]; ty=m[1][3]; tz=m[2][3]
    a,b,c=A
    det = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))
    if abs(det)<1e-12:  # fallback identity
        Ai=((1,0,0),(0,1,0),(0,0,1))
    else:
        inv=1.0/det
        Ai = (
          ((b[1]*c[2]-b[2]*c[1])*inv, (a[2]*c[1]-a[1]*c[2])*inv, (a[1]*b[2]-a[2]*b[1])*inv),
          ((b[2]*c[0]-b[0]*c[2])*inv, (a[0]*c[2]-a[2]*c[0])*inv, (a[2]*b[0]-a[0]*b[2])*inv),
          ((b[0]*c[1]-b[1]*c[0])*inv, (a[1]*c[0]-a[0]*c[1])*inv, (a[0]*b[1]-a[1]*b[0])*inv),
        )
    it = (- (Ai[0][0]*tx + Ai[0][1]*ty + Ai[0][2]*tz),
          - (Ai[1][0]*tx + Ai[1][1]*ty + Ai[1][2]*tz),
          - (Ai[2][0]*tx + Ai[2][1]*ty + Ai[2][2]*tz))
    Minv = ((Ai[0][0],Ai[0][1],Ai[0][2],it[0]),
            (Ai[1][0],Ai[1][1],Ai[1][2],it[1]),
            (Ai[2][0],Ai[2][1],Ai[2][2],it[2]),
            (0.0,0.0,0.0,1.0))
    Minv3 = Ai
    MinvT3 = ((Ai[0][0],Ai[1][0],Ai[2][0]),
              (Ai[0][1],Ai[1][1],Ai[2][1]),
              (Ai[0][2],Ai[1][2],Ai[2][2]))
    M3 = A
    return Minv, Minv3, MinvT3, M3

# =========================
# Latch + Smoothing
# =========================
def _st(op_id:int):
    s=_STATE.get(op_id)
    if s is None:
        s={"last_allowed":None,"t_last":time.perf_counter(),
           "latched_obj":None,"latched_until":0.0,"latched_allowed":None}
        _STATE[op_id]=s
    return s

def _apply_latch(s, hit_token, allowed_now, r_cam):
    now=time.perf_counter()
    if isinstance(hit_token,str) and hit_token:
        s["latched_obj"]=hit_token
        s["latched_until"]=now+_LATCH_HOLD_S
        s["latched_allowed"]=float(allowed_now)
        return float(allowed_now)
    lo=s.get("latched_obj"); until=float(s.get("latched_until",0.0)); la=s.get("latched_allowed")
    if lo is not None and la is not None:
        pad=max(_LATCH_PAD_MIN,_LATCH_PAD_K*float(r_cam))
        if (now<until) and (allowed_now<(float(la)+pad)):
            return float(la)
        s["latched_obj"]=None; s["latched_until"]=0.0; s["latched_allowed"]=None
    return allowed_now

def _apply_smoothing(s,target):
    now=time.perf_counter(); last=s.get("last_allowed",None)
    dt=max(1.0e-6, now - float(s.get("t_last",now)))
    if last is None:
        s["last_allowed"]=float(target)
    elif target>=last:
        s["last_allowed"]=min(float(target), float(last)+_OUTWARD_RATE_MPS*dt)
    else:
        s["last_allowed"]=float(target)
    s["t_last"]=now
    return float(s["last_allowed"])

# =========================
# LoS + Pushout (static+dyn)
# =========================
def _los_blocked_static(anchor, cam):
    if _STAT_ROOT<0: return False
    d=_sub(cam,anchor); dist=_len(d)
    if dist<=_EPS: return False
    dn=_mul(d,1.0/max(dist,_EPS))
    t=_raycast_bvh(_STAT_TRIS,_STAT_BVH,_STAT_ROOT, anchor, dn, max(0.0,dist-_LOS_EPS), None)
    return (t is not None)

def _los_blocked_dynamic(anchor, cam, dyn_xforms):
    d=_sub(cam,anchor); dist=_len(d)
    if dist<=_EPS: return False
    dn=_mul(d,1.0/max(dist,_EPS))
    for rec in dyn_xforms:
        rid=rec["id"]; M=rec["M"]
        dm=_DYN.get(rid)
        if not dm: continue
        Minv,Minv3,_,_ = _mat_inv(M)
        oL=_mat_apply(Minv, anchor)
        dL=_mat_mul_vec3(Minv3, dn)
        stepL=_mat_mul_vec3(Minv3, _mul(dn, dist))
        distL=_len(stepL)
        tL=_raycast_bvh(dm["tris"], dm["bvh"], dm["root"], oL, _norm(dL), max(0.0,distL-_LOS_EPS), None)
        if tL is not None:
            return True
    return False

def _pushout_any(pos, r_cam, dyn_xforms):
    p=pos
    for _ in range(max(1,_PUSH_ITERS)):
        moved=False
        # static
        if _STAT_ROOT>=0:
            d2=_nearest_bvh(_STAT_TRIS,_STAT_BVH,_STAT_ROOT, p, 1e18)
            if d2 < (r_cam*r_cam) - 1.0e-12:
                need = max(0.0, r_cam - math.sqrt(max(0.0,d2))) + 1.0e-4
                p=(p[0], p[1], p[2]+need)
                moved=True
        # dynamic
        for rec in dyn_xforms:
            rid=rec["id"]; M=rec["M"]
            dm=_DYN.get(rid)
            if not dm: continue
            Minv,_,_,_ = _mat_inv(M)
            pL=_mat_apply(Minv, p)
            d2=_nearest_bvh(dm["tris"], dm["bvh"], dm["root"], pL, 1e18)
            if d2 < (r_cam*r_cam) - 1.0e-12:
                need = max(0.0, r_cam - math.sqrt(max(0.0,d2))) + 1.0e-4
                p=(p[0], p[1], p[2]+need)
                moved=True
        if not moved: break
    return p

# =========================
# Jobs (EXACT only)
# =========================
def _job_load_static_meshes_exact(payload: dict) -> dict:
    """payload: {"meshes":[{"id":"name","tris":[x0..x2,y2,z2]}, ...]} world-space"""
    global _STAT_TRIS, _STAT_BVH, _STAT_ROOT
    _STAT_TRIS.clear(); _STAT_BVH.clear(); _STAT_ROOT=-1
    meshes = payload.get("meshes") or []
    tri_total=0
    for m in meshes:
        tris = m.get("tris") or []
        for t in tris:
            _STAT_TRIS.append(((float(t[0]),float(t[1]),float(t[2])),
                               (float(t[3]),float(t[4]),float(t[5])),
                               (float(t[6]),float(t[7]),float(t[8]))))
            tri_total += 1
    if _STAT_TRIS:
        _STAT_BVH, _STAT_ROOT = _bvh_build_array(_STAT_TRIS)
    print(f"[VIEW][XR] static EXACT synced: {len(meshes)} objs, {tri_total} tris")
    return {"ok": True, "objs": len(meshes), "tris": tri_total}

def _job_load_dynamic_meshes_exact(payload: dict) -> dict:
    """payload: {"meshes":[{"id":"name","tris":[...local...]}, ...]}"""
    meshes = payload.get("meshes") or []
    added=0
    for m in meshes:
        rid = str(m.get("id",""))
        if not rid: continue
        if rid in _DYN:  # idempotent
            continue
        local_tris=[]
        for t in (m.get("tris") or []):
            local_tris.append(((float(t[0]),float(t[1]),float(t[2])),
                               (float(t[3]),float(t[4]),float(t[5])),
                               (float(t[6]),float(t[7]),float(t[8]))))
        nodes, root = _bvh_build_array(local_tris)
        _DYN[rid] = {"tris": local_tris, "bvh": nodes, "root": root}
        added += 1
    print(f"[VIEW][XR] dynamic EXACT synced: +{added} (total {len(_DYN)})")
    return {"ok": True, "added": added, "total": len(_DYN)}

def _job_solve_third_full_exact(payload: dict) -> dict:
    """
    payload:
      op_id:int, anchor:[3], dir:[3], desired_max:float, min_cam:float, r_cam:float,
      dyn_xforms:[{"id":str,"M":[16 floats row-major]}, ...]
    returns: {"allowed":float,"candidate":float,"hit":str|None}
    """
    op_id      = int(payload.get("op_id",0))
    anchor     = tuple(float(x) for x in (payload.get("anchor") or (0,0,0)))
    dir_in     = tuple(float(x) for x in (payload.get("dir") or (0,0,1)))
    desiredMax = max(0.0, float(payload.get("desired_max",3.0)))
    min_cam    = max(_MIN_CAM_ABS, float(payload.get("min_cam",0.05)))
    r_cam      = max(0.0, float(payload.get("r_cam",0.02)))
    dyn_xforms = payload.get("dyn_xforms") or []

    d = _norm(dir_in)

    # ---- nearest hit across static + dynamic (together) ----
    best_t = desiredMax
    best_token = None

    if _STAT_ROOT>=0:
        t = _raycast_bvh(_STAT_TRIS,_STAT_BVH,_STAT_ROOT, anchor, d, desiredMax, None)
        if t is not None and 0.0<=t<=best_t:
            best_t = t; best_token = "__STATIC__"

    for rec in dyn_xforms:
        rid=rec["id"]; M=rec["M"]
        dm=_DYN.get(rid)
        if not dm: continue
        Minv,Minv3,_,M3 = _mat_inv(M)
        oL = _mat_apply(Minv, anchor)
        stepW = _mul(d, desiredMax)
        stepL = _mat_mul_vec3(Minv3, stepW)
        distL = _len(stepL)
        if distL <= 1.0e-12:
            continue
        dL = _norm(_mat_mul_vec3(Minv3, d))
        tL = _raycast_bvh(dm["tris"], dm["bvh"], dm["root"], oL, dL, distL, None)
        if tL is not None and 0.0<=tL<=distL:
            vL = _mul(dL, tL)
            vW = _mat_mul_vec3(M3, vL)
            tW = _len(vW)
            if 0.0<=tW<=best_t:
                best_t = tW; best_token = rid

    # base inward pad
    if best_token is None:
        cand = desiredMax
    else:
        base = max(min_cam, min(desiredMax, best_t - r_cam))
        cand = max(min_cam, base - (_EXTRA_PULL_METERS + _EXTRA_PULL_R_K * r_cam))

    # LoS inward refine
    lo, hi = min_cam, cand
    for _ in range(1):  # LOS_STEPS = 1, matches Blender
        mid = 0.5*(lo+hi)
        cam = _add(anchor, _mul(d, mid))
        if _los_blocked_static(anchor, cam) or _los_blocked_dynamic(anchor, cam, dyn_xforms):
            hi = mid
        else:
            lo = mid
    cand = lo

    # tiny pushout against both
    cam_pos = _add(anchor, _mul(d, cand))
    cam_pos = _pushout_any(cam_pos, r_cam, dyn_xforms)
    cand = max(min_cam, min(desiredMax, _len(_sub(cam_pos, anchor))))

    # latch + smoothing
    s=_st(op_id)
    latched = _apply_latch(s, best_token, cand, r_cam)
    final   = _apply_smoothing(s, latched)
    final   = max(min_cam, min(desiredMax, final))

    # console proof
    print(f"[VIEW][XR] full EXACT dist {final:.3f} m  cand {cand:.3f} m  hit {best_token or 'NONE'}")

    return {"allowed": float(final), "candidate": float(cand), "hit": (best_token or None)}

# =========================
# Proxies are forbidden
# =========================
def _job_reject_proxies(_payload: dict) -> dict:
    print("[VIEW][XR][ERROR] Proxy inputs rejected (AABB/sphere). Send exact mesh triangles + per-frame transforms.")
    return {"ok": False, "error": "proxies_rejected"}

# =========================
# Registry
# =========================
def register(register_job):
    # Preferred exact endpoints
    register_job("view.load_static_meshes_exact.v2", _job_load_static_meshes_exact)
    register_job("view.load_dynamic_meshes_exact.v2", _job_load_dynamic_meshes_exact)
    register_job("view.solve_third_full_exact.v1", _job_solve_third_full_exact)

    # Alias the older "full" name to the exact solver so callers can keep the name
    register_job("view.solve_third_full.v1", _job_solve_third_full_exact)

    # Hard-reject the old proxy loaders to enforce no fallback/no legacy
    register_job("view.load_static_aabbs.v1", _job_reject_proxies)
    register_job("view.update_dynamic_spheres.v1", _job_reject_proxies)
