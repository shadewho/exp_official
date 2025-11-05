# Generic XR runtime: socket server + tiny job router with plugin loading.
import json, socket, threading, sys, time, os, importlib, traceback
from typing import Callable, Dict

"""Developers manifesto: develop a strong output of the character, actions, 
environment, utilties etc so that we can eliminate guess work from XR development.
Visualize and output statistics and save time debugging and confusion in the 
future. Gate everything by ON/OFF toggles so that we can quickly enable/disable 
and don't bloat the system when not needed. It's critical every new component 
and is backed by data and visualization to help us understand the system and develop it.
Use /Developers module and the framework within to build these systems out.
Use DevHud to output text and graphs as needed. 
"""


HOST = "127.0.0.1"

# ---------------- Registry ----------------
_JOB_REGISTRY: Dict[str, Callable[[dict], dict]] = {}

def register_job(name: str, fn: Callable[[dict], dict]):
    _JOB_REGISTRY[name] = fn

def _load_plugins():
    """
    Load modules from ./xr_jobs/*.py that expose:  def register(registry_fn): ...
    This keeps xr_runtime.py tiny while allowing small, focused job files.
    """
    here = os.path.dirname(__file__)
    jobs_dir = os.path.join(here, "xr_jobs")
    if not os.path.isdir(jobs_dir):
        return
    for fname in os.listdir(jobs_dir):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        mod_name = f"{__package__}.xr_jobs.{fname[:-3]}" if __package__ else f"xr_jobs.{fname[:-3]}"
        try:
            mod = importlib.import_module(mod_name)
            reg_fn = getattr(mod, "register", None)
            if callable(reg_fn):
                reg_fn(register_job)
        except Exception:
            traceback.print_exc()

# ---------------- Socket helpers ----------------
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

# ---------------- Server ----------------
def run_server(port: int):
    _load_plugins()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"XR_READY {HOST}:{port}", flush=True)

    while True:
        try:
            conn, addr = srv.accept()
            conn.settimeout(1.0)

            # Authoritative tick at 30 Hz (for debugging)
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
                        return

                    elif t == "frame_input":
                        sim_t    = msg.get("t", 0.0)
                        frame_seq= msg.get("frame_seq", -1)
                        jobs_req = msg.get("jobs", []) or []

                        jobs_out = []
                        for j in jobs_req:
                            jid = j.get("id")
                            name = j.get("name")
                            payload = j.get("payload", {})
                            try:
                                fn = _JOB_REGISTRY.get(name)
                                if fn is None:
                                    jobs_out.append({"id": jid, "ok": False, "error": f"unknown_job:{name}"})
                                else:
                                    res = fn(payload)
                                    jobs_out.append({"id": jid, "ok": True, "result": res})
                            except Exception:
                                jobs_out.append({"id": jid, "ok": False, "error": "exception"})

                        reply = {
                            "type": "frame_cmds",
                            "t": sim_t,
                            "frame_seq": frame_seq,
                            "tick": int(tick),
                            "period": 1.0 / 30.0,
                            "set": [
                                {"kind": "text", "id": "hud_tip", "value": "XR alive — commands flowing"},
                            ],
                            "jobs": jobs_out,
                        }
                        _send(conn, reply)


                    else:
                        _send(conn, {"ok": True, "type": "ack", "seen": t, "tick": tick})
            finally:
                alive = False
                try: conn.close()
                except Exception:
                    pass

        except Exception:
            traceback.print_exc()
            time.sleep(0.1)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)
