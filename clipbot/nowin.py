"""Never open a console window for ffmpeg/ffprobe/gallery-dl/nvidia-smi child processes.

Run from a process without a console (scheduled, pythonw, or Claude's test shell), every subprocess
call pops up a black terminal window. Import this first; it adds CREATE_NO_WINDOW by default.
"""
import subprocess
import sys

if sys.platform == "win32" and not getattr(subprocess.Popen, "_nowin", False):
    _orig_init = subprocess.Popen.__init__

    def _init(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        _orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _init
    subprocess.Popen._nowin = True
