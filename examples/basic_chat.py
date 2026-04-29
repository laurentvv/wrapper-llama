import urllib.request
import json
import os
import sys
import time
from dotenv import load_dotenv
from wrapper_llama import LlamaServerManager

if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CHEMIN_EXE = os.getenv("LLAMA_EXE_PATH", "llama-server")
CHEMIN_MODELE = os.getenv("LLAMA_MODEL_PATH")
GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "0"))

if not CHEMIN_MODELE:
    raise ValueError("LLAMA_MODEL_PATH doit être défini dans le fichier .env")

t_total_start = time.perf_counter()

print(f"Modèle  : {os.path.basename(CHEMIN_MODELE)}")
print(f"GPU     : {GPU_LAYERS} layers")
print(f"Exe     : {CHEMIN_EXE}")
print()

t_start = time.perf_counter()
with LlamaServerManager(
    exe_path=CHEMIN_EXE,
    model_path=CHEMIN_MODELE,
    gpu_layers=GPU_LAYERS,
    debug=False,
) as server:
    t_server_ready = time.perf_counter()
    print(f"⏱  Démarrage serveur : {t_server_ready - t_start:.2f}s")

    url = f"http://{server.host}:{server.port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {"role": "user", "content": "Raconte-moi une courte blague sur les développeurs."}
        ]
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    print("Envoi de la requête...")
    t_req = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as reponse:
            data = json.loads(reponse.read().decode("utf-8"))
            t_resp = time.perf_counter()
            print(f"⏱  Inférence         : {t_resp - t_req:.2f}s")
            print(f"\nRéponse :\n{data['choices'][0]['message']['content']}")
    except Exception as e:
        print("Erreur de requête :", e)

t_stop = time.perf_counter()
print(f"\n⏱  Arrêt serveur     : {t_stop - t_resp:.2f}s")
print(f"⏱  Total             : {t_stop - t_total_start:.2f}s")
