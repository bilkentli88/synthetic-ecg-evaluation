import json, platform, subprocess, sys
from pathlib import Path
try:
    import torch
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    gpu=None
payload={'python':sys.version,'platform':platform.platform(),'gpu':gpu}
try: payload['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
except Exception: payload['pip_freeze']=[]
out=Path('environment_snapshot.json'); out.write_text(json.dumps(payload,indent=2)); print(out)
