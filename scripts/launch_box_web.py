import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
server = root / "minicpm_v46_box_web.py"
env = os.environ.copy()
env["ADB"] = r"D:\PhD_LetGoo\PhD_Farming\edge-ai\.tools\platform-tools\adb.exe"
subprocess.Popen(
    [sys.executable, str(server), "--host", "127.0.0.1", "--port", "7877"],
    env=env,
    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
)
print("started")
