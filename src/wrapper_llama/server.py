import subprocess
import os
import sys
import time
import atexit
import multiprocessing
import threading
import socket
import urllib.request
from typing import Optional

class LlamaServerManager:
    """
    Gestionnaire de cycle de vie pour llama-server.exe / llama-server (Windows/Linux).
    Utilisation recommandée :
        with LlamaServerManager(...) as server:
            # utilisation du serveur
    """

    def __init__(
        self,
        exe_path: str,
        model_path: str,
        port: int = 8080,
        host: str = "127.0.0.1",
        gpu_layers: int = 0,
        ctx_size: int = 2048,
        threads: Optional[int] = None,
        timeout_start: float = 30.0,
        debug: bool = False
    ):
        self.exe_path = exe_path
        self.model_path = model_path
        self.port = port
        self.host = host
        self.gpu_layers = gpu_layers
        self.ctx_size = ctx_size
        self.threads = threads or multiprocessing.cpu_count()
        self.timeout_start = timeout_start
        self.debug = debug

        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._started = False

        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"Exécutable introuvable : {self.exe_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        atexit.register(self.stop)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        with self._lock:
            if self._started:
                print("Serveur déjà en cours d'exécution.")
                return

            if self._is_port_in_use(self.host, self.port):
                raise RuntimeError(f"Le port {self.port} est déjà utilisé.")

            command = [
                self.exe_path,
                "-m", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "-ngl", str(self.gpu_layers),
                "-t", str(self.threads),
                "-c", str(self.ctx_size)
            ]

            print(f"Lancement de {os.path.basename(self.exe_path)} sur {self.host}:{self.port}")

            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            if self.debug:
                kwargs["stdout"] = None
                kwargs["stderr"] = None
            else:
                kwargs["stdout"] = subprocess.DEVNULL
                kwargs["stderr"] = subprocess.DEVNULL

            self.process = subprocess.Popen(command, **kwargs)

            time.sleep(0.5)
            if self.process.poll() is not None:
                raise RuntimeError("Le serveur a échoué au démarrage (processus terminé).")

            if not self._wait_for_server():
                self.stop()
                raise RuntimeError("Le serveur n'a pas répondu dans le délai imparti.")

            self._started = True
            print("Serveur prêt.")

    def stop(self):
        with self._lock:
            if not self._started or self.process is None:
                return
            print("Arrêt du serveur...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Le serveur ne répond pas, tentative de kill...")
                self.process.kill()
                self.process.wait()
            self.process = None
            self._started = False
            print("Serveur arrêté.")

    def is_running(self) -> bool:
        return self._started and self._health_check()

    def _health_check(self) -> bool:
        try:
            url = f"http://{self.host}:{self.port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _wait_for_server(self) -> bool:
        deadline = time.time() + self.timeout_start
        while time.time() < deadline:
            if self.process.poll() is not None:
                return False
            if self._health_check():
                return True
            time.sleep(0.5)
        return False

    @staticmethod
    def _is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
