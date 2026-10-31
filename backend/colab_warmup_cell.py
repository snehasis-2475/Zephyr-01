# ===== Step 5 · Models + RAM-limited warm-up (keeps this process under 10 GB) =====
import os, sys, gc, time, ctypes, threading
import psutil, torch

RAM_LIMIT_GB = 10.0      # hard cap: warm-up aborts cleanly instead of crashing Colab
RAM_SOFT_GB = 8.5        # above this: force GC and hand freed memory back to the OS
LOAD_TEST_PAINT = True   # False = only download Hunyuan3D-Paint (lowest RAM warm-up)

os.environ.update({
    "S2S_SHAPE_MODEL": "tencent/Hunyuan3D-2mini",          # L4/A100: "tencent/Hunyuan3D-2"
    "S2S_SHAPE_SUBFOLDER": "hunyuan3d-dit-v2-mini-turbo",  # L4/A100: "hunyuan3d-dit-v2-0-turbo"
    "S2S_TEX_MODEL": "tencent/Hunyuan3D-2",
    "S2S_SD_MODEL": "Lykon/dreamshaper-8",
    "S2S_CN_MODEL": "lllyasviel/control_v11p_sd15_scribble",
    "S2S_ENABLE_TEXTURE": "1",
    "S2S_SEQUENTIAL": "1",   # REQUIRED on 12 GB RAM: only one model in memory at a time
})
os.chdir("/content/Hunyuan3D-2")
for p in ("/content", "/content/Hunyuan3D-2"):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- RAM helpers ----------
_libc = ctypes.CDLL("libc.so.6")
_proc = psutil.Process()

def rss_gb():
    return _proc.memory_info().rss / 1024**3

def sys_gb():
    return psutil.virtual_memory().used / 1024**3

def trim():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _libc.malloc_trim(0)  # glibc keeps freed blocks; this returns them to the OS

class RamLimiter:
    """Background watchdog: trims at the soft limit, raises MemoryError in the
    main thread at the hard limit so loading stops before the kernel is killed."""

    def __init__(self, hard_gb, soft_gb, interval=0.25):
        self.hard, self.soft, self.interval = hard_gb, soft_gb, interval
        self.peak, self.tripped = 0.0, False
        self._stop = threading.Event()
        self._main = threading.main_thread().ident

    def _run(self):
        while not self._stop.is_set():
            r = rss_gb()
            self.peak = max(self.peak, r)
            if r > self.hard and not self.tripped:
                self.tripped = True
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._main), ctypes.py_object(MemoryError))
            elif r > self.soft:
                trim()
            time.sleep(self.interval)

    def __enter__(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False

print(f"Start: process {rss_gb():.1f} GB | system {sys_gb():.1f} / "
      f"{psutil.virtual_memory().total / 1024**3:.1f} GB | limit {RAM_LIMIT_GB} GB")

# ---------- Phase 1: download weights to disk (no RAM needed) ----------
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline

E = os.environ
downloads = [
    ("SD 1.5 (DreamShaper-8)", lambda: DiffusionPipeline.download(E["S2S_SD_MODEL"], use_safetensors=True)),
    ("ControlNet Scribble", lambda: snapshot_download(
        E["S2S_CN_MODEL"], allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])),
    ("Hunyuan3D shape", lambda: snapshot_download(
        E["S2S_SHAPE_MODEL"], allow_patterns=[f"{E['S2S_SHAPE_SUBFOLDER']}/*"])),
    ("Hunyuan3D paint + delight", lambda: snapshot_download(
        E["S2S_TEX_MODEL"], allow_patterns=["hunyuan3d-paint-v2-0-turbo/*", "hunyuan3d-delight-v2-0/*"])),
]
for name, fn in downloads:
    t0 = time.time()
    fn()
    trim()
    print(f"  downloaded {name:28s} {time.time() - t0:6.0f}s | process {rss_gb():.1f} GB")

# ---------- Phase 2: load-test each model one at a time under the limiter ----------
import importlib
if "s2s_server" in sys.modules:  # step 4 rewrote the file: pick up the new code
    importlib.reload(sys.modules["s2s_server"])
import s2s_server as S
S.SEQUENTIAL = True  # enforce even if the module was imported earlier with "0"

tests = [("sd", S.MODELS.get_sd), ("rembg", S.MODELS.get_rembg), ("shape", S.MODELS.get_shape)]
if LOAD_TEST_PAINT and S.texture_available():
    tests.append(("paint", S.MODELS.get_paint))

results = {}
with RamLimiter(RAM_LIMIT_GB, RAM_SOFT_GB) as lim:
    for name, load in tests:
        t0 = time.time()
        try:
            model = load()
            del model
            results[name] = "OK"
            print(f"  OK   {name:6s} loaded in {time.time() - t0:5.0f}s | peak {lim.peak:.1f} GB")
        except MemoryError:
            results[name] = "over limit"
            lim.tripped = False
            print(f"  FAIL {name:6s} exceeded {RAM_LIMIT_GB} GB - aborted cleanly")
        except Exception as exc:  # report and continue with the next model
            msg = str(exc)
            if "MemoryError" in msg:  # watchdog fired inside a wrapped call
                results[name] = "over limit"
                lim.tripped = False
                print(f"  FAIL {name:6s} exceeded {RAM_LIMIT_GB} GB - aborted cleanly")
            else:
                results[name] = f"error: {type(exc).__name__}"
                short = msg if len(msg) < 500 else msg[:250] + " ... " + msg[-250:]
                print(f"  FAIL {name:6s} {type(exc).__name__}: {short}")
        finally:
            if name != "rembg":          # rembg is tiny, keep it loaded
                S.MODELS.release(name)
            trim()
            print(f"       after release: process {rss_gb():.1f} GB | system {sys_gb():.1f} GB")

    print(f"\nWarm-up done | peak process RAM {lim.peak:.1f} GB (limit {RAM_LIMIT_GB} GB) | {results}")

if results.get("sd") != "OK" or results.get("shape") != "OK":
    raise RuntimeError("Core models did not fit - restart the session and re-run steps 4-5")
if results.get("paint", "OK") != "OK":
    print("Hunyuan3D-Paint failed to load: turn OFF 'Paint texture' in the app (shapes still work).")
