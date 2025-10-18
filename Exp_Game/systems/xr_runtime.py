# Exploratory/Exp_Game/systems/xr_runtime.py
import json, socket, threading, sys, time, math, traceback
from typing import Optional, Tuple, List, Dict

HOST = "127.0.0.1"

# --------------------------
# Small math helpers (no bpy)
# --------------------------
def _v_len(v):      return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
def _v_norm(v):
    L = _v_len(v)
    if L <= 1.0e-12: return (0.0, 0.0, 0.0)
    inv = 1.0 / L
    return (v[0]*inv, v[1]*inv, v[2]*inv)
def _v_add(a,b):    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _v_sub(a,b):    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _v_mul(v,s):    return (v[0]*s, v[1]*s, v[2]*s)
def _dot(a,b):      return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

def _segment_sphere_first_hit(origin, dnorm, max_dist, center, radius) -> Optional[float]:
    """
    Ray/segment vs sphere, return distance to first hit within [0, max_dist] or None.
    dnorm must be normalized.
    """
    oc = _v_sub(origin, center)
    b = 2.0 * _dot(dnorm, oc)
    c = _dot(oc, oc) - radius*radius
    disc = b*b - 4.0*c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t1 = (-b - sq) * 0.5
    t2 = (-b + sq) * 0.5
    # take the smallest non-negative within the segment
    hit = None
    if 0.0 <= t1 <= max_dist: hit = t1
    if 0.0 <= t2 <= max_dist: hit = min(hit, t2) if hit is not None else t2
    return hit

# --------------------------
# Socket helpers
# --------------------------
def _recv_lines(sock):
    buf = b""
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                return
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl == -1:
                    break
                line = buf[:nl].decode("utf-8", "replace")
                buf = buf[nl+1:]
                yield line
        except socket.timeout:
            continue
        except (ConnectionResetError, OSError):
            return
        except Exception:
            traceback.print_exc()

def _send(sock, obj):
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(data)

# --------------------------
# Server
# --------------------------
def run_server(port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"XR_READY {HOST}:{port}", flush=True)

    while True:
        try:
            conn, addr = srv.accept()
            conn.settimeout(1.0)

            # simple authoritative tick at 30 Hz
            tick = 0
            alive = True

            def ticker():
                nonlocal tick, alive
                period = 1.0 / 30.0
                next_t = time.perf_counter() + period
                while alive:
                    now = time.perf_counter()
                    if now >= next_t:
                        tick += 1
                        next_t += period
                    else:
                        time.sleep(min(0.005, max(0.0, next_t - now)))

            th = threading.Thread(target=ticker, daemon=True)
            th.start()

            try:
                for line in _recv_lines(conn):
                    try:
                        msg = json.loads(line)
                    except Exception:
                        _send(conn, {"ok": False, "error": "bad_json"})
                        continue

                    t = msg.get("type")

                    if t == "hello":
                        _send(conn, {"ok": True, "type": "hello_ack", "runtime": "XR", "version": 2})

                    elif t == "ping":
                        _send(conn, {"ok": True, "type": "pong", "tick": tick})

                    elif t == "shutdown":
                        _send(conn, {"ok": True, "type": "bye"})
                        alive = False
                        try: conn.shutdown(socket.SHUT_RDWR)
                        except Exception: pass
                        try: conn.close()
                        except Exception: pass
                        return  # clean exit

                    elif t == "frame_input":
                        # Keep your existing demo HUD / transform echo to prove the pipe.
                        now_t = msg.get("t", 0.0)
                        cmds = {
                            "type": "frame_cmds",
                            "t": now_t,
                            "set": [
                                {"kind": "text", "id": "hud_tip", "value": "XR alive — commands flowing"},
                            ]
                        }
                        _send(conn, cmds)

                    else:
                        _send(conn, {"ok": True, "type": "ack", "seen": t, "tick": tick})

            finally:
                alive = False
                try: conn.close()
                except Exception: pass

        except Exception:
            traceback.print_exc()
            time.sleep(0.1)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)
