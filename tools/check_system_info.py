"""
check_system_info.py
--------------------
Collects the hardware and software details that the manuscript needs
(Methods, "Computational accounting and reproducibility") and writes them
to system_info.txt next to this script.

Run it ON THE SAME COMPUTER that produced the run times reported in the paper,
from the same Python environment (the one Spyder uses).

How to run in Spyder:  open this file and press F5.
How to run in Anaconda Prompt:  python check_system_info.py

Optional: if you pass the path of a config file used for the runs, the script
also prints its n_jobs (number of parallel workers):
    python check_system_info.py path\\to\\config_used.json
"""

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

lines = []


def out(text=""):
    print(text)
    lines.append(text)


def run(cmd):
    """Run a shell command and return its output (empty string on failure)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              shell=isinstance(cmd, str)).stdout.strip()
    except Exception:
        return ""


def cpu_name():
    # 1) Windows registry (most reliable, no extra packages)
    if sys.platform.startswith("win"):
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            mhz = winreg.QueryValueEx(key, "~MHz")[0]
            return name, mhz
        except Exception:
            pass
    # 2) Linux
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip(), None
    # 3) macOS / fallback
    name = run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    return name, None


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        txt = run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | '
                  'Measure-Object -Property NumberOfCores -Sum).Sum"')
        try:
            return int(txt), os.cpu_count()
        except ValueError:
            pass
    return None, os.cpu_count()


def total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    if sys.platform.startswith("win"):
        txt = run('powershell -NoProfile -Command "(Get-CimInstance '
                  'Win32_ComputerSystem).TotalPhysicalMemory"')
        try:
            return int(txt) / 1024 ** 3
        except ValueError:
            pass
    return None


def cpu_sockets_and_model_count():
    if sys.platform.startswith("win"):
        return run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Count"')
    return ""


def gpu_names():
    if sys.platform.startswith("win"):
        return run('powershell -NoProfile -Command "(Get-CimInstance '
                   'Win32_VideoController).Name"').replace("\n", "; ")
    return run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])


def version_of(module_name, dist_name=None):
    """Return the installed version of a package, or 'not installed'."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(dist_name or module_name)
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "installed (version unknown)")
    except Exception:
        return "not installed"


# --------------------------------------------------------------------------
out("System information for the manuscript")
out("Collected: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
out("=" * 60)

name, mhz = cpu_name()
phys, logical = physical_cores()
ram = total_ram_gb()

out("CPU model          : %s" % name)
if mhz:
    out("CPU base clock     : %.2f GHz (from registry)" % (mhz / 1000.0))
out("CPU packages       : %s" % (cpu_sockets_and_model_count() or "n/a"))
out("Physical cores     : %s" % phys)
out("Logical processors : %s" % logical)
out("RAM                : %s" % ("%.1f GB" % ram if ram else "n/a"))
out("GPU(s) present     : %s" % (gpu_names() or "none found"))
out("Operating system   : %s" % platform.platform())
if sys.platform.startswith("win"):
    try:
        out("Windows edition    : %s" % platform.win32_edition())
    except Exception:
        pass
out("Machine name       : %s" % platform.node())
out("Python             : %s (%s)" % (platform.python_version(), sys.executable))

out("")
out("Package versions")
out("-" * 60)
packages = [
    ("numpy", None), ("scipy", None), ("scikit-image", "scikit-image"),
    ("SimpleITK", "SimpleITK"), ("nibabel", None), ("pydicom", None),
    ("cv2", "opencv-python"), ("pwlf", None), ("pandas", None),
    ("joblib", None), ("psutil", None), ("matplotlib", None),
    ("torch", None), ("lungmask", None), ("totalsegmentator", "TotalSegmentator"),
    ("nnunetv2", None),
]
for mod, dist in packages:
    if mod == "scikit-image":
        v = version_of("skimage", "scikit-image")
    elif mod == "cv2":
        v = version_of("cv2", "opencv-python")
        if v == "not installed":
            v = version_of("cv2", "opencv-python-headless")
    else:
        v = version_of(mod, dist)
    out("%-18s %s" % (mod, v))

# Was PyTorch able to see a GPU? (The paper says every method ran on the CPU.)
try:
    import torch
    out("")
    out("torch.cuda.is_available() : %s" % torch.cuda.is_available())
    out("torch CPU threads         : %s" % torch.get_num_threads())
    if torch.cuda.is_available():
        out("CUDA device               : %s" % torch.cuda.get_device_name(0))
        out("NOTE: a GPU is visible to PyTorch. Check that lungmask was run with "
            "force_cpu=True and TotalSegmentator with device='cpu' (or -d cpu).")
except Exception:
    pass

# Optional: n_jobs from the config file used for the runs
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    try:
        with open(sys.argv[1]) as f:
            cfg = json.load(f)
        out("")
        out("Config file        : %s" % sys.argv[1])
        out("n_jobs (workers)   : %s" % cfg.get("n_jobs", "not in file"))
        out("config version     : %s" % cfg.get("version", cfg.get("config_version", "n/a")))
    except Exception as e:
        out("Could not read config: %s" % e)

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_info.txt")
with open(target, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
out("")
out("Saved to: " + target)
