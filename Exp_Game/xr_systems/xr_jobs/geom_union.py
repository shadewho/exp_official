# Exp_Game/xr_systems/xr_jobs/geom_union.py
import math
from .geom_dynamic import _DYN
from .geom_static  import _STATIC

# Build-once coarse blocks over static triangles (no changes to geom_static.py needed)
_BLOCKS = {"gen": -1, "blocks": [], "block_tris": 2048}

def _dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _m3_mul_v(m,v):
    return (m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
            m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
            m[6]*v[0]+m[7]*v[1]+m[8]*v[2])

def _m4_mul_point_rowmajor(m,p):
    x = m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3]
    y = m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7]
    z = m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11]
    w = m[12]*p[0]+ m[13]*p[1]+ m[14]*p[2] + m[15]
    if abs(w) > 1.0e-12:
        inv = 1.0/w; return (x*inv,y*inv,z*inv)
    return (x,y,z)

def _ray_tri_mt(o, d, v0, v1, v2, tmax):
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    pvec = _cross(d, e2)
    det  = _dot(e1, pvec)
    if abs(det) < 1.0e-9: return None
    inv_det = 1.0/det
    tvec = (o[0]-v0[0], o[1]-v0[1], o[2]-v0[2])
    u = _dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0: return None
    qvec = _cross(tvec, e1)
    v = _dot(d, qvec) * inv_det
    if v < 0.0 or (u+v) > 1.0: return None
    t = _dot(e2, qvec) * inv_det
    if t < 0.0 or t > tmax: return None
    return t

def _prepare_blocks():
    tris = _STATIC.get("tris")
    if not isinstance(tris, list):
        _BLOCKS["gen"] = -1; _BLOCKS["blocks"] = []; return
    n_tris = len(tris) // 9
    if _BLOCKS["gen"] == n_tris:
        return
    blocks = []
    B = int(max(256, _BLOCKS["block_tris"]))
    for start in range(0, n_tris, B):
        end = min(start + B, n_tris)
        minx=miny=minz= 1e30
        maxx=maxy=maxz=-1e30
        for i in range(start, end):
            base = i*9
            x0,y0,z0 = tris[base],   tris[base+1], tris[base+2]
            x1,y1,z1 = tris[base+3], tris[base+4], tris[base+5]
            x2,y2,z2 = tris[base+6], tris[base+7], tris[base+8]
            minx=min(minx, x0,x1,x2); miny=min(miny, y0,y1,y2); minz=min(minz, z0,z1,z2)
            maxx=max(maxx, x0,x1,x2); maxy=max(maxy, y0,y1,y2); maxz=max(maxz, z0,z1,z2)
        blocks.append({"start": start, "end": end, "aabb": (minx,miny,minz,maxx,maxy,maxz)})
    _BLOCKS["gen"] = n_tris
    _BLOCKS["blocks"] = blocks

def _ray_aabb(origin, dirn, max_d, aabb):
    ox,oy,oz = origin; dx,dy,dz = dirn
    minx,miny,minz,maxx,maxy,maxz = aabb
    tmin = 0.0; tmax = max_d
    for o,d,minv,maxv in ((ox,dx,minx,maxx),(oy,dy,miny,maxy),(oz,dz,minz,maxz)):
        if abs(d) < 1.0e-12:
            if o < minv or o > maxv: return False
            continue
        invd = 1.0/d
        t1 = (minv - o) * invd
        t2 = (maxv - o) * invd
        if t1 > t2: t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
        if tmax < tmin: return False
    return tmax >= 0.0 and tmin <= max_d

def _static_best(o, d, max_d):
    tris = _STATIC.get("tris")
    if not isinstance(tris, list) or not tris:
        return None
    _prepare_blocks()
    best_t = None; best_n = None
    for blk in _BLOCKS["blocks"]:
        if not _ray_aabb(o, d, max_d, blk["aabb"]):
            continue
        for i in range(blk["start"], blk["end"]):
            b = i*9
            v0 = (tris[b],   tris[b+1], tris[b+2])
            v1 = (tris[b+3], tris[b+4], tris[b+5])
            v2 = (tris[b+6], tris[b+7], tris[b+8])
            t = _ray_tri_mt(o, d, v0, v1, v2, max_d)
            if t is None: continue
            if best_t is None or t < best_t:
                e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
                e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
                n  = _cross(e1, e2)
                ln = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) or 1.0
                n  = (n[0]/ln, n[1]/ln, n[2]/ln)
                if _dot(n, d) > 0.0:  # face against the ray
                    n = (-n[0], -n[1], -n[2])
                best_t, best_n = t, n
    if best_t is None:
        return None
    return ("static", best_t, best_n)

# --- REPLACE _dynamic_best entirely ---
def _dynamic_best(o, d, max_d):
    best_dist = None; best_n = None; best_id = None
    for _id, ent in _DYN.items():
        tris = ent.get("tris"); Rinv=ent.get("Minv3"); RinvT=ent.get("MinvT3"); M3=ent.get("M3"); Minv=ent.get("Minv")
        if not tris or not Rinv or not RinvT or not M3 or not Minv:
            continue
        o_l = _m4_mul_point_rowmajor(Minv, o)
        d_l = _m3_mul_v(Rinv, d)
        dl2 = d_l[0]*d_l[0] + d_l[1]*d_l[1] + d_l[2]*d_l[2]
        if dl2 <= 1.0e-24: continue
        inv = 1.0 / math.sqrt(dl2); d_l = (d_l[0]*inv, d_l[1]*inv, d_l[2]*inv)

        step_l = _m3_mul_v(Rinv, (d[0]*max_d, d[1]*max_d, d[2]*max_d))
        max_l  = math.sqrt(step_l[0]*step_l[0] + step_l[1]*step_l[1] + step_l[2]*step_l[2])
        if max_l <= 1.0e-12: continue

        n_tris = len(tris)//9
        best_tl = None; best_nl = None
        for i in range(n_tris):
            b=i*9
            v0=(tris[b],   tris[b+1], tris[b+2])
            v1=(tris[b+3], tris[b+4], tris[b+5])
            v2=(tris[b+6], tris[b+7], tris[b+8])
            t = _ray_tri_mt(o_l, d_l, v0, v1, v2, max_l)
            if t is None: continue
            if best_tl is None or t < best_tl:
                e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
                e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
                best_tl = t
                best_nl = _cross(e1, e2)

        if best_tl is None: continue

        vec_l = (d_l[0]*best_tl, d_l[1]*best_tl, d_l[2]*best_tl)
        vec_w = _m3_mul_v(M3, vec_l)
        dist_w = math.sqrt(vec_w[0]*vec_w[0] + vec_w[1]*vec_w[1] + vec_w[2]*vec_w[2])
        if dist_w <= 1.0e-12: continue

        n_w = _m3_mul_v(RinvT, best_nl)
        ln = math.sqrt(n_w[0]*n_w[0] + n_w[1]*n_w[1] + n_w[2]*n_w[2]) or 1.0
        n_w = (n_w[0]/ln, n_w[1]/ln, n_w[2]/ln)
        if _dot(n_w, d) > 0.0: n_w = (-n_w[0], -n_w[1], -n_w[2])

        if best_dist is None or dist_w < best_dist:
            best_dist, best_n, best_id = dist_w, n_w, int(_id)

    if best_dist is None:
        return None
    return ("dynamic", best_dist, best_n, best_id)


# --- REPLACE _ray_union entirely (adds id when src=dynamic) ---
def _ray_union(payload: dict) -> dict:
    o = payload.get("origin"); d = payload.get("dir"); max_d = float(payload.get("max_d", 0.0))
    if (not isinstance(o,(list,tuple)) or len(o)!=3 or
        not isinstance(d,(list,tuple)) or len(d)!=3 or
        max_d <= 1.0e-9):
        return {"hit": False}

    ox,oy,oz = float(o[0]), float(o[1]), float(o[2])
    dx,dy,dz = float(d[0]), float(d[1]), float(d[2])
    dl = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dl <= 1.0e-12: return {"hit": False}
    d = (dx/dl, dy/dl, dz/dl); o = (ox, oy, oz)

    sb = _static_best(o, d, max_d)                 # -> ("static", dist, n)
    db = _dynamic_best(o, d, max_d)                # -> ("dynamic", dist, n, id)

    if not sb and not db:
        return {"hit": False}
    if sb and db:
        pick = sb if sb[1] <= db[1] else db
    else:
        pick = sb if sb else db

    if pick[0] == "static":
        return {"hit": True, "dist": float(pick[1]), "normal": (float(pick[2][0]), float(pick[2][1]), float(pick[2][2])), "src": "STATIC"}
    else:
        # dynamic with id
        return {"hit": True, "dist": float(pick[1]), "normal": (float(pick[2][0]), float(pick[2][1]), float(pick[2][2])), "src": "DYNAMIC", "id": int(pick[3])}


# --- ADD: new job returning both static and dynamic details + union pick ---
def _ray_both(payload: dict) -> dict:
    o = payload.get("origin"); d = payload.get("dir"); max_d = float(payload.get("max_d", 0.0))
    if (not isinstance(o,(list,tuple)) or len(o)!=3 or
        not isinstance(d,(list,tuple)) or len(d)!=3 or
        max_d <= 1.0e-9):
        return {"hit": False}

    ox,oy,oz = float(o[0]), float(o[1]), float(o[2])
    dx,dy,dz = float(d[0]), float(d[1]), float(d[2])
    dl = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dl <= 1.0e-12: return {"hit": False}
    d = (dx/dl, dy/dl, dz/dl); o = (ox, oy, oz)

    sb = _static_best(o, d, max_d)          # -> ("static", dist, n)
    db = _dynamic_best(o, d, max_d)         # -> ("dynamic", dist, n, id)

    out = {"hit": False}
    # pack stat
    if sb:
        out["stat"] = {"hit": True, "dist": float(sb[1]), "normal": (float(sb[2][0]), float(sb[2][1]), float(sb[2][2]))}
    else:
        out["stat"] = {"hit": False}
    # pack dyn
    if db:
        out["dyn"] = {"hit": True, "dist": float(db[1]), "normal": (float(db[2][0]), float(db[2][1]), float(db[2][2])), "id": int(db[3])}
    else:
        out["dyn"] = {"hit": False}

    # union pick
    if sb and db:
        pick = sb if sb[1] <= db[1] else db
    elif sb:
        pick = sb
    elif db:
        pick = db
    else:
        return out  # no hit anywhere

    out["hit"] = True
    if pick[0] == "static":
        out["src"]  = "STATIC"
        out["dist"] = float(pick[1])
        out["normal"] = (float(pick[2][0]), float(pick[2][1]), float(pick[2][2]))
    else:
        out["src"]  = "DYNAMIC"
        out["dist"] = float(pick[1])
        out["normal"] = (float(pick[2][0]), float(pick[2][1]), float(pick[2][2]))
        out["id"]   = int(pick[3])
    return out

# --- NEW: 3-band forward union ray (feet/mid/head) --------------------------
def _clamp(v, lo, hi):  # local helper
    return lo if v < lo else (hi if v > hi else v)

def _ray_triplet(payload: dict) -> dict:
    # Inputs:
    #   origin: (x,y,z) base position (character base)
    #   dir:    (x,y,z) world direction (need not be normalized)
    #   r:      capsule radius
    #   h:      capsule height
    #   step_len: forward step length for this sub-step
    o = payload.get("origin"); d = payload.get("dir")
    r = float(payload.get("r", 0.3))
    h = float(payload.get("h", 1.8))
    step_len = float(payload.get("step_len", 0.0))
    if (not isinstance(o,(list,tuple)) or len(o)!=3 or
        not isinstance(d,(list,tuple)) or len(d)!=3 or
        step_len <= 1.0e-9 or r <= 0.0 or h <= (2.0*r + 1.0e-9)):
        return {"hit": False}

    ox,oy,oz = float(o[0]), float(o[1]), float(o[2])
    dx,dy,dz = float(d[0]), float(d[1]), float(d[2])
    dl = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dl <= 1.0e-12:
        return {"hit": False}
    d = (dx/dl, dy/dl, dz/dl)
    base = (ox, oy, oz)
    midz = _clamp(h*0.5, r, h-r)
    bands = [("low", r), ("mid", midz), ("high", h-r)]
    max_d = step_len + r

    best = None  # tuple (label, src, dist, normal)
    details = {}

    for label, z in bands:
        o_z = (base[0], base[1], base[2] + z)
        sb = _static_best(o_z, d, max_d)      # -> ("static", dist, n) or None
        db = _dynamic_best(o_z, d, max_d)     # -> ("dynamic", dist, n) or None

        pick = None
        if sb and db:
            pick = sb if sb[1] <= db[1] else db
        elif sb:
            pick = sb
        elif db:
            pick = db

        if pick:
            src, dist, n = pick
            details[label] = {"hit": True, "dist": float(dist),
                              "normal": (float(n[0]), float(n[1]), float(n[2])),
                              "src": ("STATIC" if src=="static" else "DYNAMIC")}
            if (best is None) or (dist < best[2]):
                best = (label, details[label]["src"], dist, n)
        else:
            details[label] = {"hit": False}

    if best is None:
        return {"hit": False, "bands": details}

    label, src, dist, n = best
    return {
        "hit": True,
        "dist": float(dist),
        "normal": (float(n[0]), float(n[1]), float(n[2])),
        "src": src,
        "band": label,
        "bands": details,
    }


# --- REPLACE register entirely (adds geom.ray_both.v1) ---
def register(register_job):
    register_job("geom.ray_union.v1", _ray_union)
    register_job("geom.ray_both.v1",  _ray_both)
    register_job("geom.ray_triplet.v1", _ray_triplet)



