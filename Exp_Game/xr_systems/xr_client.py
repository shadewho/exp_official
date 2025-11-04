# Exploratory/Exp_Game/systems/xr_client.py
import json, os, socket, subprocess, sys, time, threading
from typing import Optional, Tuple, Any, Dict

HOST = "127.0.0.1"

def _find_free_port(start=8765, limit=50) -> int:
    for p in range(start, start + limit):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((HOST, p))
            s.close()
            return p
        except OSError:
            s.close()
    raise RuntimeError("No free localhost port found")

class XRClient:
    """
    """
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self.port: Optional[int] = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    # ---------------- Internals ----------------

    def _connect_with_retry(self, deadline_s=2.0) -> bool:
        end = time.perf_counter() + float(deadline_s)
        while time.perf_counter() < end:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.25)
                s.connect((HOST, self.port))
                self._sock = s
                return True
            except Exception:
                time.sleep(0.05)
        return False

    def _readline(self, timeout_s=0.25) -> Optional[dict]:
        if not self._sock:
            return None
        self._sock.settimeout(timeout_s)
        buf = b""
        try:
            while True:
                ch = self._sock.recv(1)
                if not ch:
                    return None
                if ch == b"\n":
                    break
                buf += ch
        except socket.timeout:
            return None
        try:
            return json.loads(buf.decode("utf-8", "replace"))
        except Exception:
            return None

    def _send(self, obj: dict) -> bool:
        if not self._sock:
            return False
        try:
            data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
            self._sock.sendall(data)
            return True
        except Exception:
            return False

    # ---------------- Lifecycle ----------------

    def start(self) -> bool:
        """
        Start the runtime, wait for XR_READY, connect, send hello.
        Uses Blender's Python and resolves xr_runtime.py in the SAME folder.
        """
        self.last_error = None
        try:
            try:
                import bpy
                py_bin = bpy.app.binary_path_python or sys.executable
            except Exception:
                py_bin = sys.executable
            print(f"[XR] python bin: {py_bin}")

            self.port = _find_free_port()
            print(f"[XR] choosing free port {self.port}")

            here = os.path.dirname(__file__)  # .../Exp_Game/systems
            script_path = os.path.normpath(os.path.join(here, "xr_runtime.py"))
            print(f"[XR] runtime script: {script_path}")
            if not os.path.isfile(script_path):
                self.last_error = f"runtime_script_missing:{script_path}"
                print(f"[XR][error] {self.last_error}")
                return False

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            print("[XR] launching runtime process...")
            self._proc = subprocess.Popen(
                [py_bin, "-u", script_path, str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

            # wait for XR_READY
            ready = False
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < 3.0:
                if self._proc.poll() is not None:
                    break
                line = self._proc.stdout.readline() if self._proc.stdout else ""
                if line:
                    print(f"[XR][child] {line.strip()}")
                if line.startswith("XR_READY"):
                    ready = True
                    break
                time.sleep(0.01)

            if not ready:
                err_tail = ""
                try:
                    if self._proc and self._proc.stderr:
                        time.sleep(0.05)
                        err_tail = self._proc.stderr.read() or ""
                except Exception:
                    pass
                self.last_error = "XR_READY not seen"
                print(f"[XR][error] {self.last_error}")
                if err_tail.strip():
                    print(f"[XR][stderr]\n{err_tail.strip()}")
                self.stop(force=True)
                return False

            print("[XR] child signaled ready; attempting socket connect...")
            if not self._connect_with_retry(deadline_s=3.0):
                self.last_error = "socket_connect_failed"
                print("[XR][error] failed to connect socket.")
                self.stop(force=True)
                return False

            print("[XR] socket connected. sending hello...")
            self._send({"type": "hello", "from": "Blender", "pid": os.getpid()})
            ack = self._readline(timeout_s=0.5)
            print(f"[XR] hello_ack: {ack}")

            return True
        except Exception as e:
            self.last_error = f"exception: {e}"
            print(f"[XR][error] {self.last_error}")
            try:
                self.stop(force=True)
            except Exception:
                pass
            return False

    def stop(self, force=False):
        with self._lock:
            try:
                if self._sock and not force:
                    self._send({"type": "shutdown"})
                    _ = self._readline(timeout_s=0.25)
            except Exception:
                pass

            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None

            if self._proc:
                try:
                    self._proc.wait(timeout=0.5)
                except Exception:
                    pass
                if self._proc.poll() is None:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                self._proc = None

    # ---------------- Convenience RPCs ----------------

    def ping(self) -> Tuple[bool, Optional[int]]:
        """
        Returns (ok, tick|None). Single quick reconnect attempt if socket died.
        """
        with self._lock:
            def _do_ping():
                if not self._sock:
                    return (False, None)
                if not self._send({"type": "ping"}):
                    return (False, None)
                resp = self._readline(timeout_s=0.5)
                if not isinstance(resp, dict) or resp.get("type") != "pong":
                    return (False, None)
                return (True, int(resp.get("tick", 0)))

            ok, tick = _do_ping()
            if ok:
                return (True, tick)

            # soft reconnect once
            try:
                if self._sock:
                    try: self._sock.close()
                    except Exception: pass
                self._sock = None
                if not self._connect_with_retry(deadline_s=1.0):
                    return (False, None)
                self._send({"type": "hello", "from": "Blender", "pid": os.getpid()})
                _ = self._readline(timeout_s=0.25)
            except Exception:
                return (False, None)

            return _do_ping()

    def request_frame(self, payload: dict, timeout_s: float = 0.01) -> Optional[dict]:
        """
        Send one 'frame_input' request and wait briefly for a 'frame_cmds' reply.
        Returns dict or None if no reply (non-fatal).
        """
        with self._lock:
            if not self._sock:
                return None
            if not self._send({"type": "frame_input", **payload}):
                return None
            return self._readline(timeout_s=timeout_s)