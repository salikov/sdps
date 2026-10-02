# !pip install -q -U transformers diffusers accelerate gradio
# !pip install -q Cython
# !pip install -q insightface onnxruntime --no-build-isolation || (pip install -q "numpy<2" && pip install -q insightface onnxruntime --no-build-isolation)
# !pip install -q -U kaggle   # === ИЗМЕНЕНО: образ Kaggle несёт старый CLI 2.0.2, который тихо падает на больших заливках ===

import warnings, logging, os, gc, re, sys, math, glob, inspect, shutil, zipfile, traceback, subprocess, threading, time
import json, importlib.util
import torch, gradio as gr, urllib.request
import numpy as np, cv2
from PIL import Image, ImageOps
from diffusers import (StableDiffusionXLPipeline, ControlNetModel,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

custom_css = """
.gallery-container .grid-wrap { justify-content: flex-start !important; gap: 10px !important; }
.gallery-container .thumbnail-item { width: 120px !important; height: 120px !important; object-fit: contain !important; }
"""

# ---------- 0. Автоопределение окружения ----------
def detect_env():
    if os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except Exception:
        return "local"

ENV = detect_env()
if ENV == "kaggle":
    WORK = "/kaggle/working"
    def _get_token():
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            return None
elif ENV == "colab":
    WORK = "/content"
    def _get_token():
        try:
            from google.colab import userdata
            return userdata.get("HF_TOKEN")
        except Exception:
            return None
else:
    WORK = os.path.join(os.path.expanduser("~"), "aistudio")
    os.makedirs(WORK, exist_ok=True)
    def _get_token():
        return os.environ.get("HF_TOKEN")

hf_token = _get_token()
assert torch.cuda.is_available(), "GPU не подключена!"
print(f"🖥️ Окружение: {ENV} | GPU: {torch.cuda.get_device_name(0)}")
print("✅ HF_TOKEN:", "найден" if hf_token else "не найден (не обязателен)")

def _p(prog, frac, desc):
    if prog is not None:
        try: prog(frac, desc=desc)
        except Exception: pass

# ---------- 0.5. Кеш тяжелых файлов в личном Kaggle-датасете ----------
# Приоритет: 1) датасет-кеш подключён как Input (мгновенно) → 2) уже собран
# в этой сессии → 3) скачивание кеш-датасета через Kaggle API (мимо HF) →
# 4) обычная загрузка с HF + (один раз) сборка своего кеш-датасета.
# Работает только на Kaggle и только при наличии секрета KAGGLE_API.
KAGGLE_CACHE_SLUG = "sd-photostudio-cache"
KAGGLE_CACHE_INPUT = f"/kaggle/input/{KAGGLE_CACHE_SLUG}"
KAGGLE_CACHE_LOCAL = "/kaggle/tmp/sd_cache" if ENV == "kaggle" else os.path.join(WORK, "sd_cache")
CACHE_MARKER = "base/model.safetensors"   # главный признак валидности кеша

_cache = {"mode": None, "root": None, "upload": False, "dataset_id": None}

def _run_kaggle(*args, timeout=1800):
    try:
        r = subprocess.run(["kaggle"] + list(args), capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return False, "kaggle CLI не найден"
    except subprocess.TimeoutExpired:
        return False, "таймаут команды kaggle"

def _kaggle_username():
    """Секрет KAGGLE_API (JSON {"username":...,"key":...}) -> настройка CLI + username."""
    if importlib.util.find_spec("kaggle") is None:
        return None
    try:
        from kaggle_secrets import UserSecretsClient
        raw = UserSecretsClient().get_secret("KAGGLE_API")
    except Exception:
        return None
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        creds = json.loads(raw)
        username, key = creds["username"], creds["key"]
    except Exception:
        print("⚠️ Секрет KAGGLE_API не похож на {\"username\":...,\"key\":...} — игнорируем.")
        return None
    kdir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kdir, exist_ok=True)
    kjson = os.path.join(kdir, "kaggle.json")
    with open(kjson, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kjson, 0o600)
    return username

def _init_cache():
    # 1) датасет-кеш подключён как Input — идеально, ничего качать не надо
    if os.path.isfile(os.path.join(KAGGLE_CACHE_INPUT, CACHE_MARKER)):
        _cache.update(mode="input", root=KAGGLE_CACHE_INPUT, upload=False)
        print("✅ Кеш-датасет подключён как Input — все модели уже на диске, качать нечего")
        return
    # 2) кеш уже собран в этой сессии (перезапуск ячейки)
    if os.path.isfile(os.path.join(KAGGLE_CACHE_LOCAL, CACHE_MARKER)):
        _cache.update(mode="local", root=KAGGLE_CACHE_LOCAL, upload=False)
        username = _kaggle_username()
        if username:
            _cache["dataset_id"] = f"{username}/{KAGGLE_CACHE_SLUG}"
            ok, _ = _run_kaggle("datasets", "files", _cache["dataset_id"], timeout=60)
            if not ok:
                _cache["upload"] = True   # файлы собрали, а датасет создать не успели
        print("✅ Кеш уже на месте в этой сессии" + (" (датасет создадим в конце)" if _cache["upload"] else ""))
        return
    # 3) есть секрет KAGGLE_API?
    username = _kaggle_username()
    if not username:
        _cache.update(mode="none", root=None, upload=False)
        print("ℹ️ Секрета KAGGLE_API нет — кеш не используется, обычная загрузка моделей.")
        return
    dataset_id = f"{username}/{KAGGLE_CACHE_SLUG}"
    _cache["dataset_id"] = dataset_id
    ok, _ = _run_kaggle("datasets", "files", dataset_id, timeout=60)
    if ok:
        # 3а) датасет существует — скачиваем его (мимо Hugging Face)
        print(f"📦 Кеш-датасет {dataset_id} существует — скачиваем его (мимо Hugging Face)...")
        os.makedirs(KAGGLE_CACHE_LOCAL, exist_ok=True)
        ok2, out2 = _run_kaggle("datasets", "download", dataset_id,
                                "-p", KAGGLE_CACHE_LOCAL, "--unzip", timeout=3600)
        for z in glob.glob(os.path.join(KAGGLE_CACHE_LOCAL, "*.zip")):
            try: os.remove(z)
            except OSError: pass
        if ok2 and os.path.isfile(os.path.join(KAGGLE_CACHE_LOCAL, CACHE_MARKER)):
            _cache.update(mode="kaggle", root=KAGGLE_CACHE_LOCAL, upload=False)
            print("✅ Кеш скачан с Kaggle — тяжёлые модели грузятся без Hugging Face.")
        else:
            _cache.update(mode="none", root=None, upload=False)
            print(f"⚠️ Кеш не скачался ({out2[:200]}) — грузим модели обычным путём.")
        return
    # 3б) датасета нет: модели качаем как обычно, в конце соберём кеш-датасет
    _cache.update(mode="none", root=KAGGLE_CACHE_LOCAL, upload=True)
    print("ℹ️ Кеш-датасета ещё нет: модели скачаем как обычно и один раз соберём кеш.")

_init_cache()

def _cache_path(rel):
    return os.path.join(_cache["root"], rel) if _cache["root"] else None

def _cache_writable():
    return _cache["mode"] in ("local", "kaggle", "none") and _cache["root"] is not None

def cached_hf_file(rel, repo_id, filename):
    """Одиночный файл с HF: из кеша, иначе скачать и (если кеш пишется) положить в него."""
    p = _cache_path(rel)
    if p and os.path.isfile(p):
        return p
    src = hf_hub_download(repo_id, filename, token=hf_token)
    if p and _cache_writable():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy2(src, p)
        return p
    return src

def cached_repo_dir(rel, repo_id, allow_patterns, marker, weights=(), ignore_patterns=None):
    """Папка с файлами HF-репозитория (для from_pretrained): из кеша или snapshot в кеш."""
    def _complete(d):
        if not os.path.isfile(os.path.join(d, marker)):
            return False
        if weights and not any(os.path.isfile(os.path.join(d, w)) for w in weights):
            return False
        return True
    p = _cache_path(rel)
    if p and _complete(p):
        return p
    if p and _cache_writable():
        try:
            os.makedirs(p, exist_ok=True)
            snapshot_download(repo_id, allow_patterns=allow_patterns,
                              ignore_patterns=ignore_patterns, local_dir=p, token=hf_token)
            # старые версии huggingface_hub кладут симлинки — заменяем на реальные файлы
            for root_dir, _, files in os.walk(p):
                for fn in files:
                    fp = os.path.join(root_dir, fn)
                    if os.path.islink(fp):
                        real = os.path.realpath(fp)
                        os.remove(fp)
                        shutil.copy2(real, fp)
            shutil.rmtree(os.path.join(p, ".cache"), ignore_errors=True)
            if _complete(p):
                return p
        except Exception:
            pass
    return None   # кеш недоступен — вызывающий код грузит как раньше

# ---------- 1. Базовая модель ----------
BASE_MODEL = "realvis"   # "realvis" = RealVisXL V5.0 (фотореализм) | "sdxl" = SDXL base 1.0
REALVIS_REPO = "SG161222/RealVisXL_V5.0"
REALVIS_FILE = "RealVisXL_V5.0_fp16.safetensors"
SDXL_CONFIG = "stabilityai/stable-diffusion-xl-base-1.0"

print("⏳ Загрузка базовой модели...")
if BASE_MODEL == "realvis":
    try:
        ckpt = cached_hf_file("base/model.safetensors", REALVIS_REPO, REALVIS_FILE)
    except Exception:
        files = [f for f in list_repo_files(REALVIS_REPO, token=hf_token)
                 if f.lower().endswith(".safetensors")
                 and "vae" not in f.lower().replace("novae", "")   # исключаем VAE-файлы и вариант -Novae
                 and "inpaint" not in f.lower()]
        files.sort(key=lambda f: 0 if "fp16" in f.lower() else 1)
        ckpt = cached_hf_file("base/model.safetensors", REALVIS_REPO, files[0])
    try:
        pipe_base = StableDiffusionXLPipeline.from_single_file(ckpt, config=SDXL_CONFIG, torch_dtype=torch.float16, token=hf_token)
    except TypeError:
        pipe_base = StableDiffusionXLPipeline.from_single_file(ckpt, torch_dtype=torch.float16, token=hf_token)
else:
    pipe_base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", token=hf_token)

pipe_base.to("cuda")
for opt in (lambda: pipe_base.vae.enable_slicing(), lambda: pipe_base.vae.enable_tiling()):
    try: opt()
    except Exception: pass
torch.cuda.empty_cache()
print("✅ Базовая модель готова!")

# ---------- 2. NSFW-фильтр (принудительный) ----------
_safety_processor, _safety_checker = None, None

def _load_safety():
    global _safety_processor, _safety_checker
    if _safety_checker is not None:
        return
    from transformers import CLIPImageProcessor
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

    # 1) из кеша (Input / скачанный / собранный в этой сессии)
    sdir = cached_repo_dir("safety", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                           ["feature_extractor/*", "safety_checker/*"],
                           marker="safety_checker/config.json",
                           weights=("safety_checker/model.safetensors",
                                    "safety_checker/model.bin"),
                           ignore_patterns=["*.bin", "*.msgpack", "*.flax", "*.h5"])
    if sdir:
        try:
            _safety_processor = CLIPImageProcessor.from_pretrained(os.path.join(sdir, "feature_extractor"))
            _safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(sdir, "safety_checker"))
            _safety_checker.to("cpu").eval()
            print("✅ NSFW-фильтр включён (из кеша)")
            return
        except Exception:
            traceback.print_exc()
            _safety_processor, _safety_checker = None, None

    # 2) обычная загрузка с HF (как раньше)
    last = None
    for repo in ("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"):
        try:
            _safety_processor = CLIPImageProcessor.from_pretrained(repo, subfolder="feature_extractor", token=hf_token)
            _safety_checker = StableDiffusionSafetyChecker.from_pretrained(repo, subfolder="safety_checker", token=hf_token)
            _safety_checker.to("cpu").eval()
            print("✅ NSFW-фильтр включён (каждый результат проходит проверку)")
            return
        except Exception as e:
            last = e
    raise RuntimeError(f"Не удалось загрузить NSFW-фильтр: {last}")

def nsfw_filter_image(img):
    """Официальный NSFW-классификатор diffusers. Пометка -> чёрный кадр (стандарт diffusers)."""
    if _safety_checker is None:
        return img, False
    try:
        pil = img.convert("RGB")
        inputs = _safety_processor(images=[pil], return_tensors="pt")
        np_img = (np.array(pil).astype(np.float32) / 255.0)[None, ...]
        with torch.no_grad():
            _, has_nsfw = _safety_checker(images=np_img, clip_input=inputs.pixel_values.float())
        flagged = bool(has_nsfw[0]) if has_nsfw else False
        if flagged:
            return Image.new("RGB", pil.size, (0, 0, 0)), True
        return img, False
    except Exception:
        return Image.new("RGB", img.size, (0, 0, 0)), True   # сбой фильтра -> блокируем кадр

_load_safety()

# ---------- 3. InstantID (грузится сразу при старте) ----------
pipe_instant = None
face_app = None
INSTANTID_HF_REPO = "InstantX/InstantID"   # официальные веса (организация InstantX)
INSTANTID_CODE_DIR = os.path.join(WORK, "instantid_code")
INSTANTID_FILE = "pipeline_stable_diffusion_xl_instantid.py"
INSTANTID_REPO_DIR = os.path.join(WORK, "InstantID_repo")
IFR_ROOT = os.path.join(WORK, "insightface_models")

def apply_sampler(pipe, sampler_name):
    if sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

def draw_kps(image_pil, kps, color_list=((255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255))):
    """Точная копия официальной функции InstantID (на таких картинках обучался ControlNet)."""
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]], dtype=int)
    kps = np.asarray(kps, dtype=np.float32)[:, :2]
    img = np.array(image_pil)[:, :, ::-1].copy()
    for a, b in limbSeq:
        color = color_list[a]
        x1, y1 = kps[a]; x2, y2 = kps[b]
        mean_x, mean_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        length = int(np.hypot(x2 - x1, y2 - y1))
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        polygon = cv2.ellipse2Poly((mean_x, mean_y), (length // 2, stickwidth), int(angle), 0, 360, 1)
        canvas = img.copy()
        cv2.fillConvexPoly(canvas, polygon, color)
        img = cv2.bitwise_or(img, canvas)
    for j, (x, y) in enumerate(kps):
        cv2.circle(img, (int(x), int(y)), stickwidth, color_list[j], -1)
    return Image.fromarray(img)

def _load_instantid_class():
    """Класс пайплайна: ядро diffusers -> community-файл под версию diffusers -> репозиторий InstantID."""
    try:
        from diffusers import StableDiffusionXLInstantIDPipeline
        return StableDiffusionXLInstantIDPipeline
    except ImportError:
        pass
    import diffusers as _d
    os.makedirs(INSTANTID_CODE_DIR, exist_ok=True)
    dest = os.path.join(INSTANTID_CODE_DIR, INSTANTID_FILE)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 10000):
        last_err = None
        for url in (f"https://raw.githubusercontent.com/huggingface/diffusers/v{_d.__version__}/examples/community/{INSTANTID_FILE}",
                    f"https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/{INSTANTID_FILE}"):
            try:
                urllib.request.urlretrieve(url, dest)
                if os.path.getsize(dest) > 10000:
                    last_err = None
                    break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Не удалось скачать {INSTANTID_FILE}: {last_err}")
    if INSTANTID_CODE_DIR not in sys.path:
        sys.path.insert(0, INSTANTID_CODE_DIR)
    sys.modules.pop("pipeline_stable_diffusion_xl_instantid", None)
    try:
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        return StableDiffusionXLInstantIDPipeline
    except Exception:
        if not os.path.exists(os.path.join(INSTANTID_REPO_DIR, INSTANTID_FILE)):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/InstantID/InstantID.git", INSTANTID_REPO_DIR],
                           check=True, capture_output=True)
        if INSTANTID_REPO_DIR in sys.path:
            sys.path.remove(INSTANTID_REPO_DIR)
        sys.path.insert(0, INSTANTID_REPO_DIR)
        sys.modules.pop("pipeline_stable_diffusion_xl_instantid", None)
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        return StableDiffusionXLInstantIDPipeline

def _ensure_antelopev2(progress=None):
    """Модели InsightFace: .onnx обязаны лежать ровно в {root}/models/antelopev2/."""
    from insightface.app import FaceAnalysis
    model_dir = os.path.join(IFR_ROOT, "models", "antelopev2")
    os.makedirs(model_dir, exist_ok=True)

    # --- Kaggle-кеш: если onnx уже есть в кеше — просто копируем на место ---
    cdir = _cache_path("insightface/antelopev2")
    if cdir:
        for f in glob.glob(os.path.join(cdir, "*.onnx")):
            shutil.copy(f, os.path.join(model_dir, os.path.basename(f)))

    def _top():
        return sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
    onnx_files = _top()
    if not onnx_files:
        for src in glob.glob(os.path.join(model_dir, "**", "*.onnx"), recursive=True):
            shutil.copy(src, os.path.join(model_dir, os.path.basename(src)))
        onnx_files = _top()
    if not onnx_files:
        zip_path = os.path.join(IFR_ROOT, "antelopev2.zip")
        last_err = None
        for attempt in (1, 2):
            try:
                _p(progress, 0.15, f"Скачивание antelopev2 (попытка {attempt}/2)...")
                urllib.request.urlretrieve("https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip", zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for n in [n for n in zf.namelist() if n.lower().endswith(".onnx")]:
                        with zf.open(n) as s, open(os.path.join(model_dir, os.path.basename(n)), "wb") as d:
                            shutil.copyfileobj(s, d)
                os.remove(zip_path)
                break
            except Exception as e:
                last_err = e
                if os.path.exists(zip_path):
                    try: os.remove(zip_path)
                    except Exception: pass
        else:
            raise RuntimeError(f"Не удалось скачать antelopev2: {last_err}")
        onnx_files = _top()

    # --- если собираем кеш-датасет — сохраняем onnx для будущей заливки ---
    if _cache["upload"] and onnx_files:
        cdir = _cache_path("insightface/antelopev2")
        os.makedirs(cdir, exist_ok=True)
        for f in onnx_files:
            dst = os.path.join(cdir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)

    listing = ", ".join(f"{os.path.basename(f)} ({os.path.getsize(f)//(1024*1024)} МБ)" for f in onnx_files)
    print("📦 Модели InsightFace:", listing)
    try:
        return FaceAnalysis(name="antelopev2", root=IFR_ROOT, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise RuntimeError(f"InsightFace не инициализировался ({e}). Файлы: {listing}")

def ensure_face_app(progress=None):
    global face_app
    if face_app is None:
        face_app = _ensure_antelopev2(progress)
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return face_app

def _load_instantid_adapter(pipe, adapter_path):
    def _reset():
        try: pipe.set_ip_adapter()
        except Exception: pass
    for arg in (INSTANTID_HF_REPO, adapter_path, [adapter_path]):
        try:
            pipe.load_ip_adapter_instantid(arg)
            return
        except Exception:
            _reset()
    try:
        try: state = torch.load(adapter_path, map_location="cpu")
        except Exception: state = torch.load(adapter_path, map_location="cpu", weights_only=False)
        pipe.image_proj_model.load_state_dict(state["image_proj"])
        torch.nn.ModuleList(list(pipe.unet.attn_processors.values())).load_state_dict(state["ip_adapter"])
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Не удалось загрузить IP-Adapter InstantID (подробности в консоли).")

def ensure_instantid(progress=None):
    """Сборка InstantID на компонентах базовой модели (одна копия весов в памяти)."""
    global pipe_instant
    if pipe_instant is not None:
        return pipe_instant
    torch.cuda.empty_cache(); gc.collect()
    _p(progress, 0.05, "Получение кода InstantID...")
    PipelineClass = _load_instantid_class()
    ensure_face_app(progress)
    _p(progress, 0.3, "Загрузка ControlNet InstantID (~2.5 ГБ)...")
    # --- Kaggle-кеш: ControlNet из локальной папки, иначе как раньше с HF ---
    cn_dir = cached_repo_dir("instantid", INSTANTID_HF_REPO, ["ControlNetModel/*"],
                             marker="ControlNetModel/config.json",
                             weights=("ControlNetModel/diffusion_pytorch_model.safetensors",
                                      "ControlNetModel/diffusion_pytorch_model.bin"),
                             ignore_patterns=["*.bin", "*.msgpack", "*.flax", "*.h5"])
    if cn_dir:
        controlnet = ControlNetModel.from_pretrained(os.path.join(cn_dir, "ControlNetModel"),
                                                     torch_dtype=torch.float16)
    else:
        controlnet = ControlNetModel.from_pretrained(INSTANTID_HF_REPO, subfolder="ControlNetModel",
                                                     torch_dtype=torch.float16, token=hf_token)
    _p(progress, 0.5, "Сборка пайплайна (компоненты базовой модели переиспользуются)...")
    allowed = set(inspect.signature(PipelineClass.__init__).parameters) - {"self"}
    comps = {k: v for k, v in dict(pipe_base.components).items() if k in allowed}
    pipe_instant = PipelineClass(controlnet=controlnet, **comps)
    _p(progress, 0.7, "Загрузка IP-Adapter InstantID (~1.7 ГБ)...")
    # --- Kaggle-кеш: IP-Adapter как одиночный файл ---
    adapter_path = cached_hf_file("instantid/ip-adapter.bin", INSTANTID_HF_REPO, "ip-adapter.bin")
    _load_instantid_adapter(pipe_instant, adapter_path)
    # ФИКС: community-файл вызывает check_inputs с устаревшим порядком аргументов ->
    # ложная ошибка "controlnet_conditioning_scale must be float". Отключаем валидацию.
    try: PipelineClass.check_inputs = lambda self, *a, **k: None
    except Exception: pass
    try: pipe_instant.check_inputs = lambda *a, **k: None
    except Exception: pass
    pipe_instant.to("cuda")
    try: pipe_instant.image_proj_model.to("cuda", torch.float16)
    except Exception: pass
    torch.cuda.empty_cache()
    return pipe_instant

print("⏳ Подготовка InstantID (ControlNet + IP-Adapter + InsightFace)...")
try:
    ensure_face_app()
    ensure_instantid()
    print("✅ InstantID готов — режим персонажа доступен сразу!")
except Exception:
    traceback.print_exc()
    print("⚠️ InstantID не собрался при старте — попытка повторится при первом использовании режима.")

# ---------- 3.5. Сборка кеш-датасета Kaggle (один раз) ----------
# === ИЗМЕНЕНО: честная проверка после заливки (datasets files + ожидание),
# при ошибке печатается реальный вывод CLI, а не молчаливый успех ===
def _dataset_exists(dataset_id):
    ok, _ = _run_kaggle("datasets", "files", dataset_id, timeout=60)
    return ok

def _maybe_create_cache_dataset():
    if not _cache["upload"] or not _cache["dataset_id"]:
        return
    root = _cache["root"]
    if not root or not os.path.isdir(root):
        return

    def _any(*rels):
        return any(os.path.isfile(os.path.join(root, r)) for r in rels)

    missing = []
    if not _any("base/model.safetensors"):
        missing.append("базовая модель")
    if not _any("instantid/ControlNetModel/diffusion_pytorch_model.safetensors",
                "instantid/ControlNetModel/diffusion_pytorch_model.bin"):
        missing.append("ControlNet InstantID")
    if not _any("instantid/ip-adapter.bin"):
        missing.append("IP-Adapter InstantID")
    if not _any("safety/safety_checker/model.safetensors",
                "safety/safety_checker/model.bin"):
        missing.append("NSFW-фильтр")
    if len(glob.glob(os.path.join(root, "insightface/antelopev2/*.onnx"))) < 3:
        missing.append("InsightFace (antelopev2)")
    if missing:
        print(f"⚠️ Кеш неполный ({', '.join(missing)}) — датасет не создаём. "
              "Перезапустите ноутбук, чтобы попробовать снова.")
        return

    gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root) for f in fs) / 2**30
    print(f"⏳ Создаём кеш-датасет {_cache['dataset_id']} (~{gb:.1f} ГБ, займёт 10–25 минут)...")
    with open(os.path.join(root, "dataset-metadata.json"), "w") as f:
        json.dump({"title": KAGGLE_CACHE_SLUG, "id": _cache["dataset_id"],
                   "licenses": [{"name": "CC0-1.0"}]}, f, indent=2)

    ok, out = _run_kaggle("datasets", "create", "-p", root, "--dir-mode", "zip", timeout=7200)
  
    if not ok:
        print("⚠️ kaggle datasets create вернул ошибку. Последние строки вывода:")
        print((out or "")[-800:])
        if "already exist" not in (out or "").lower():
            print("   Генерации это не ломает — при следующем запуске попытка повторится.")
            return
        print("   (похоже, датасет с таким slug уже существует — проверяем...)")

    # честная финальная проверка: датасет должен отвечать на 'datasets files'
    found = _dataset_exists(_cache["dataset_id"])
    attempt = 0
    while not found and attempt < 5:
        attempt += 1
        print(f"   Датасет ещё не виден, ждём 30 с (попытка {attempt}/5)...")
        time.sleep(30)
        found = _dataset_exists(_cache["dataset_id"])

    if found:
        print(f"🎉 Кеш-датасет подтверждён: https://www.kaggle.com/datasets/{_cache['dataset_id']}")
        print("   Подключите его как Input (Input → Add Input → Your Work + Datasets) и перезапустите сессию.")
        print("   ⚠️ В поиске Add Input свежий датасет может появиться не сразу — иногда до часа.")
    else:
        print("❌ Датасет не подтверждается через kaggle datasets files.")
        print("   Возможно, ещё обрабатывается — проверьте через пару минут вручную:")
        print(f"   kaggle datasets files {_cache['dataset_id']}")
        print("   Либо посмотрите в профиле kaggle.com/<логин> → Datasets.")
        print("   На генерацию не влияет: при следующем запуске кеш попробует создаться снова.")

_maybe_create_cache_dataset()

def detect_face(pil_image):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    faces = face_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    if not faces:
        return None, None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face, img

def prepare_face_inputs(face, img):
    emb = torch.from_numpy(np.asarray(face["embedding"], dtype=np.float32)).unsqueeze(0).to("cuda", torch.float16)
    x1, y1, x2, y2 = face["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.6
    left, top = cx - side / 2, cy - side / 2
    crop = img.crop((int(left), int(top), int(left + side), int(top + side))).resize((512, 512), Image.LANCZOS)
    kps = np.array(face["kps"], dtype=np.float32)[:, :2]
    kps[:, 0] -= left; kps[:, 1] -= top
    kps *= (512 / side)
    return emb, draw_kps(crop, kps)

def view_selected_image(evt: gr.SelectData, gallery_items):
    if gallery_items and evt.index < len(gallery_items):
        return gallery_items[evt.index][0]
    return None

def _eta_progress(progress, total_steps, base_frac, span_frac):
    """Колбэк с ETA: среднее время шага × оставшиеся шаги."""
    t0 = time.time()
    total = int(total_steps)
    def cb(pipe, step, timestep, cb_kwargs):
        done = step + 1
        per_step = (time.time() - t0) / max(done, 1)
        eta = per_step * (total - done)
        _p(progress, base_frac + (done / total) * span_frac, f"Шаг {done}/{total} · осталось ~{eta:.0f} с")
        return cb_kwargs
    return cb

# ---------- 4. ГЕНЕРАЦИЯ ----------
def generate(prompt, negative_prompt, steps, cfg, seed, width, height, sampler, batch_size,
             use_face, face_image, id_scale, struct_scale, progress=gr.Progress()):
    actual_seed = torch.randint(0, 2**32 - 1, (1,)).item() if seed == -1 else int(seed)
    gallery_items = []
    nsfw_blocked = 0
    try:
        if use_face:
            # ============ РЕЖИМ ПЕРСОНАЖА (InstantID) ============
            if face_image is None:
                return [], "❌ Включён режим персонажа, но фото не загружено!"
            pipe = ensure_instantid(progress)
            _p(progress, 0.1, "Анализ лица на фото...")
            face, img = detect_face(face_image)
            if face is None:
                return [], "❌ Лицо не найдено! Нужно фото анфас, лицо крупно, хороший свет."
            emb, kps_img = prepare_face_inputs(face, img)
            apply_sampler(pipe, sampler)
            try: pipe.set_ip_adapter_scale(float(id_scale))
            except Exception: pass
            has_cb = "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
            for i in range(int(batch_size)):
                base = 0.2 + (i / int(batch_size)) * 0.6
                span = 0.6 / int(batch_size)
                call_kwargs = {}
                if has_cb:
                    call_kwargs["callback_on_step_end"] = _eta_progress(progress, steps, base, span)
                    call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                with torch.inference_mode():
                    raw = pipe(prompt=prompt, negative_prompt=negative_prompt,
                               image_embeds=emb,                    # идентичность с фото (IP-Adapter)
                               image=kps_img,                       # структура лица (ControlNet)
                               controlnet_conditioning_scale=float(struct_scale),
                               num_inference_steps=int(steps), guidance_scale=float(cfg),
                               width=int(width), height=int(height),
                               generator=torch.Generator("cuda").manual_seed(actual_seed + i),
                               **call_kwargs).images[0]
                _p(progress, base + span, "Проверка NSFW-фильтром...")
                out, flagged = nsfw_filter_image(raw)
                nsfw_blocked += int(flagged)
                cap = f"Персонаж {i+1} (Seed: {actual_seed + i})" + (" · 🚫 NSFW → заблокировано" if flagged else "")
                gallery_items.append((out, cap))
                torch.cuda.empty_cache(); gc.collect()
            msg = f"✅ Готово! Персонаж с фото (InstantID), {int(batch_size)} шт."
            if nsfw_blocked: msg += f" | 🚫 NSFW-фильтр заблокировал: {nsfw_blocked}"
            return gallery_items, msg
        else:
            # ============ ОБЫЧНАЯ ГЕНЕРАЦИЯ ============
            apply_sampler(pipe_base, sampler)
            try: pipe_instant.set_ip_adapter_scale(0.0)   # IP-Adapter не должен влиять на обычную генерацию
            except Exception: pass
            for i in range(int(batch_size)):
                base = 0.1 + (i / int(batch_size)) * 0.7
                span = 0.7 / int(batch_size)
                with torch.inference_mode():
                    raw = pipe_base(prompt=prompt, negative_prompt=negative_prompt,
                                    num_inference_steps=int(steps), guidance_scale=float(cfg),
                                    width=int(width), height=int(height),
                                    generator=torch.Generator("cuda").manual_seed(actual_seed + i),
                                    callback_on_step_end=_eta_progress(progress, steps, base, span),
                                    callback_on_step_end_tensor_inputs=['latents']).images[0]
                _p(progress, base + span, "Проверка NSFW-фильтром...")
                out, flagged = nsfw_filter_image(raw)
                nsfw_blocked += int(flagged)
                cap = f"Вариант {i+1} (Seed: {actual_seed + i})" + (" · 🚫 NSFW → заблокировано" if flagged else "")
                gallery_items.append((out, cap))
                torch.cuda.empty_cache(); gc.collect()
            msg = f"✅ Успешно! Сгенерировано {int(batch_size)} изображений."
            if nsfw_blocked: msg += f" | 🚫 NSFW-фильтр заблокировал: {nsfw_blocked}"
            return gallery_items, msg
    except Exception as e:
        traceback.print_exc()
        msg = f"Ошибка: {e}"
        if "out of memory" in str(e).lower():
            msg += " 💡 Уменьшите разрешение (например, 832×832) или Кол-во."
        return [], msg

# ---------- 5. ИНТЕРФЕЙС ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 AI Studio: фотореалистичная генерация + Персонаж с фото")
    gr.Markdown("🛡️ NSFW-фильтр включён принудительно: помеченные кадры заменяются чёрными.")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(label="Prompt",
                value="candid photograph of a young woman with wavy chestnut hair sitting by the window in a Parisian cafe, soft golden morning light, shallow depth of field, 85mm lens, photorealistic, detailed skin texture, film grain")
            neg_in = gr.Textbox(label="Negative Prompt",
                value="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, blurry, jpeg artifacts, worst quality, low quality, watermark, text, signature")
            with gr.Accordion("👤 Персонаж с фото (InstantID)", open=False):
                gr.Markdown("Модели уже загружены при старте — просто включите, загрузите портрет и пишите промпт.")
                use_face_cb = gr.Checkbox(label="Использовать персонажа с фото", value=False)
                face_img = gr.Image(label="Фото лица (лучше анфас, лицо крупно, без очков)", type="pil", height=300)
                with gr.Row():
                    id_scale_s = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.9, label="Сходство лица (ID)")
                    struct_scale_s = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.8, label="Точность структуры лица")
                gr.Markdown("💡 Опишите персонажа и сцену в промпте. ID 0.8–1.0, структура 0.8–0.9. "
                            "Если лицо «не то» — поднимите ID до 1.0–1.2.")
            with gr.Row():
                sampler_d = gr.Dropdown(choices=["DPM++ 2M Karras", "Euler a", "Euler"], value="DPM++ 2M Karras", label="Sampler")
                steps_s = gr.Slider(minimum=10, maximum=50, step=1, value=30, label="Шаги (Steps)")
            with gr.Row():
                width_s = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Ширина")
                height_s = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Высота")
            with gr.Row():
                cfg_s = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, value=5.0, label="CFG Scale (для RealVisXL хорошо 4–7)")
                batch_s = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Кол-во")
                seed_n = gr.Number(label="Seed (-1 = случайно)", value=-1)
            gen_btn = gr.Button("🎨 Сгенерировать изображение", variant="primary")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Результаты (кликните для увеличения)", columns=2, height=500,
                                 object_fit="contain", elem_classes="gallery-container")
            status = gr.Textbox(label="Статус", interactive=False)
            viewer = gr.Image(label="Просмотр полного размера", type="pil", height=512)

    gen_btn.click(fn=generate,
                  inputs=[prompt_in, neg_in, steps_s, cfg_s, seed_n, width_s, height_s, sampler_d, batch_s,
                          use_face_cb, face_img, id_scale_s, struct_scale_s],
                  outputs=[gallery, status])
    gallery.select(fn=view_selected_image, inputs=gallery, outputs=viewer)

# ---------- 6. ЗАПУСК: основной Gradio Share + резерв trycloudflare ----------
PORT = 7860
_cf_url = {"link": None}
_cf_event = threading.Event()

def _start_cloudflared(timeout=60):
    try:
        bin_path = os.path.join(WORK, "cloudflared")
        if not os.path.exists(bin_path):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", bin_path)
            os.chmod(bin_path, 0o755)
        proc = subprocess.Popen([bin_path, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        def _reader():
            for raw in proc.stdout:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", raw.decode(errors="ignore"))
                if m and _cf_url["link"] is None:
                    _cf_url["link"] = m.group(0)
                    _cf_event.set()
        threading.Thread(target=_reader, daemon=True).start()
        if _cf_event.wait(timeout):
            print("🌍 РЕЗЕРВНЫЙ URL (Cloudflare):", _cf_url["link"])
            return True
        print(f"⚠️ Cloudflare не выдал ссылку за {timeout} с")
        return False
    except Exception as e:
        print("⚠️ Cloudflare не запустился:", e)
        return False

print("🚀 Запуск интерфейса (основной — Gradio Share)...")
demo.launch(server_name="127.0.0.1", server_port=PORT, share=True, prevent_thread_lock=True)

share_url = getattr(demo, "share_url", None)
if share_url:
    try:
        urllib.request.urlopen(urllib.request.Request(share_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=25)
        print("✅ ОСНОВНОЙ URL (Gradio Share):", share_url)
    except Exception:
        print("⚠️ Gradio Share не отвечает (504?) — используйте резервный Cloudflare-URL ниже")
else:
    print("⚠️ Gradio Share не создался в этой сессии — используйте резервный Cloudflare-URL")

_start_cloudflared()
print("\n🟢 Сервер запущен. Держите ячейку запущенной, пока работаете с интерфейсом.")
try:
    while True:
        time.sleep(2)
except KeyboardInterrupt:
    print("⏹ Остановлено пользователем.")

if os.path.exists("sdph.py"):
    os.remove("sdph.py")
