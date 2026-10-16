"""Accelerator probe: which device did this machine_shape get, and how fast did it start?"""
import glob, os, subprocess, time
print("START_UTC", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)
print("accel devices", sorted(glob.glob("/dev/accel*")) + sorted(glob.glob("/dev/vfio/*")), flush=True)
print("TPU env", {k: v for k, v in os.environ.items() if "TPU" in k or "XLA" in k or "PJRT" in k}, flush=True)
try:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout, flush=True)
except Exception as e:
    print("nvidia-smi:", repr(e)[:100])
try:
    import torch, torch_xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
    print("torch", torch.__version__, "torch_xla", torch_xla.__version__, flush=True)
    print("xla device", xm.xla_device(), "global devices", xr.global_runtime_device_count(), xm.xla_device_hw(xm.xla_device()), flush=True)
except Exception as e:
    print("torch_xla:", repr(e)[:300], flush=True)
print("cpus", os.cpu_count(), flush=True)
print("DONE", flush=True)
