import uuid
import subprocess
from importlib.metadata import version

def get_eval_uuid():
    return str(uuid.uuid4())

def get_sae_lens_version():
    try:
        return version('sae_lens')
    except Exception:
        return "Unknown"

def get_sae_bench_version():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "Unknown"