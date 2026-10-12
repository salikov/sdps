#=========================
# 1 установка зависимостей
#=========================

import os, sys, subprocess, importlib

def _run_quiet(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print("❌ Команда не удалась:", " ".join(str(c) for c in cmd[:5]), "…")
        print(((r.stdout or "") + "\n" + (r.stderr or ""))[-3000:])
        raise RuntimeError("Установка прервана — см. вывод выше")
    return r

def _already_installed():
    try:
        import gfpgan, insightface, diffusers, transformers, accelerate, gradio  # noqa
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

if _already_installed():
    print("✅ Зависимости уже установлены — установка пропущена")
else:
    BSR = "/kaggle/working/BasicSR"

    print("⏳ Шаг 1/5: клон BasicSR + патч setup.py...")
    if not os.path.isdir(BSR):
        _run_quiet(["git", "clone", "https://github.com/XPixelGroup/BasicSR.git", BSR])
    setup_py = os.path.join(BSR, "setup.py")
    with open(setup_py, "r") as f:
        _code = f.read()
    if "version=get_version()," in _code:
        with open(setup_py, "w") as f:
            f.write(_code.replace("version=get_version(),", "version='1.4.2',"))
    print("✅ Шаг 1/5 готов: BasicSR склонирован и пропатчен")

    print("⏳ Шаг 2/5: зависимости BasicSR...")
    _run_quiet([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=BSR)
    print("✅ Шаг 2/5 готов")

    print("⏳ Шаг 3/5: установка BasicSR (без CUDA-компиляции)...")
    _run_quiet([sys.executable, "setup.py", "develop"], cwd=BSR)
    print("✅ Шаг 3/5 готов")

    print("⏳ Шаг 4/5: gfpgan, insightface, onnxruntime-gpu, gradio...")
    _run_quiet([sys.executable, "-m", "pip", "install",
                "gfpgan", "insightface", "onnxruntime-gpu", "opencv-python", "pillow", "gradio"])
    print("✅ Шаг 4/5 готов")

    print("⏳ Шаг 5/5: diffusers, transformers, accelerate...")
    _run_quiet([sys.executable, "-m", "pip", "install",
                "diffusers", "transformers", "accelerate", "huggingface_hub"])
    print("✅ Шаг 5/5 готов")
    print("🎉 Установка завершена")

# Страховка: если insightface капризничает из-за numpy 2.x — откат на 1.x
try:
    import insightface  # noqa
except Exception:
    _run_quiet([sys.executable, "-m", "pip", "install", "-q", "numpy<2"])
    importlib.invalidate_caches()

#=========================
# 2 загрузка моделей
# =========================

import warnings, logging, os, sys, gc, re, time, shutil, zipfile, traceback, subprocess, threading, urllib.request
import numpy as np, cv2, torch, gradio as gr
from PIL import Image, ImageOps
from diffusers import (StableDiffusionXLPipeline, DPMSolverMultistepScheduler,
                       EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)
from huggingface_hub import hf_hub_download, list_repo_files

warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append('/kaggle/working/BasicSR')   # gfpgan импортирует basicsr
import insightface
from gfpgan import GFPGANer

WORK = "/kaggle/working"
PORT = 7860
IFR_ROOT = os.path.join(WORK, "insightface_models")

custom_css = """
.gallery-container .grid-wrap { justify-content: flex-start !important; gap: 10px !important; }
.gallery-container .thumbnail-item { width: 120px !important; height: 120px !important; object-fit: contain !important; }
"""

# ---------- Вспомогательные ----------
def _p(prog, frac, desc):
    if prog is not None:
        try: prog(frac, desc=desc)
        except Exception: pass

def _get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest + ".part", "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(dest + ".part", dest)

hf_token = _get_secret("HF_TOKEN")   # опционален
assert torch.cuda.is_available(), "GPU не подключена!"

# ---------- 1. Базовая модель RealVisXL V5.0 ----------
REALVIS_REPO = "SG161222/RealVisXL_V5.0"
REALVIS_FILE = "RealVisXL_V5.0_fp16.safetensors"
SDXL_CONFIG = "stabilityai/stable-diffusion-xl-base-1.0"

print("⏳ Загрузка базовой модели RealVisXL V5.0 (~7 ГБ)...")
try:
    ckpt = hf_hub_download(REALVIS_REPO, REALVIS_FILE, token=hf_token)
except Exception:
    files = [f for f in list_repo_files(REALVIS_REPO, token=hf_token)
             if f.lower().endswith(".safetensors") and "vae" not in f.lower().replace("novae", "") and "inpaint" not in f.lower()]
    files.sort(key=lambda f: 0 if "fp16" in f.lower() else 1)
    ckpt = hf_hub_download(REALVIS_REPO, files[0], token=hf_token)

pipe_base = StableDiffusionXLPipeline.from_single_file(ckpt, config=SDXL_CONFIG, torch_dtype=torch.float16, token=hf_token)
pipe_base.to("cuda")
try: pipe_base.vae.enable_slicing()
except Exception: pass
try: pipe_base.vae.enable_tiling()
except Exception: pass
torch.cuda.empty_cache()
print("✅ Базовая модель готова!")

# ---------- 2. NSFW-ФИЛЬТР (только для генерации) ----------
_safety_processor, _safety_checker = None, None

def _load_safety():
    global _safety_processor, _safety_checker
    if _safety_checker is not None:
        return
    from transformers import CLIPImageProcessor
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    last = None
    for repo in ("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"):
        try:
            _safety_processor = CLIPImageProcessor.from_pretrained(repo, subfolder="feature_extractor", token=hf_token)
            _safety_checker = StableDiffusionSafetyChecker.from_pretrained(repo, subfolder="safety_checker", token=hf_token)
            _safety_checker.to("cpu").eval()
            print("✅ NSFW-фильтр включён (результаты генерации проходят проверку)")
            return
        except Exception as e:
            last = e
    raise RuntimeError(f"Не удалось загрузить NSFW-фильтр: {last}")

def nsfw_filter_image(img):
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
        return Image.new("RGB", img.size, (0, 0, 0)), True

_load_safety()

# ---------- 3. InsightFace (buffalo_l — нужен для inswapper) ----------
face_app = None

def ensure_face_app():
    global face_app
    if face_app is not None:
        return face_app
    from insightface.app import FaceAnalysis
    model_dir = os.path.join(IFR_ROOT, "models", "buffalo_l")
    det_model_path = os.path.join(model_dir, "det_10g.onnx")
    if not os.path.exists(det_model_path):
        print("⏳ Скачивание моделей распознавания лиц (buffalo_l, ~280 МБ)...")
        os.makedirs(model_dir, exist_ok=True)
        zip_path = os.path.join(IFR_ROOT, "buffalo_l.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        url = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith(".onnx"):
                    with zf.open(member) as source, open(os.path.join(model_dir, os.path.basename(member)), "wb") as target:
                        shutil.copyfileobj(source, target)
        os.remove(zip_path)
        print("✅ Модели лица (buffalo_l) готовы!")
    try:
        face_app = FaceAnalysis(name="buffalo_l", root=IFR_ROOT,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace работает на GPU")
    except Exception:
        face_app = FaceAnalysis(name="buffalo_l", root=IFR_ROOT, providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ InsightFace работает на CPU")
    return face_app

face_app = ensure_face_app()

# ---------- 4. inswapper_128 + GFPGAN ----------
GFPGAN_PATH = os.path.join(WORK, "GFPGANv1.4.pth")
INSP_PATH = os.path.join(WORK, "inswapper_128.onnx")

if not os.path.isfile(GFPGAN_PATH):
    print("⏳ Скачивание GFPGANv1.4.pth (~350 МБ)...")
    _download("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth", GFPGAN_PATH)
    print("✅ GFPGANv1.4.pth готов")

if not os.path.isfile(INSP_PATH) or os.path.getsize(INSP_PATH) < 100 << 20:
    print("⏳ Скачивание inswapper_128.onnx (~526 МБ)...")
    try:
        p = hf_hub_download(repo_id="Gourieff/ReActor", filename="models/inswapper_128.onnx",
                            repo_type="dataset", token=hf_token)
        shutil.copyfile(p, INSP_PATH)
    except Exception as e:
        print(f"⚠️ hf_hub_download не сработал: {str(e)[:120]} — фолбэк на прямой URL")
        _download("https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx", INSP_PATH)
    print("✅ inswapper_128.onnx готов")

assert os.path.getsize(INSP_PATH) > 100 << 20, "inswapper_128.onnx не скачался целиком — перезапустите ячейку"

print("⏳ Подключение inswapper...")
try:
    swapper = insightface.model_zoo.get_model(os.path.abspath(INSP_PATH), download=False,
                                              providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
except Exception:
    swapper = insightface.model_zoo.get_model(os.path.abspath(INSP_PATH), download=False)

print("⏳ Загрузка GFPGAN...")
face_enhancer = GFPGANer(
    model_path=os.path.abspath(GFPGAN_PATH),
    upscale=1, arch='clean', channel_multiplier=2, device='cuda'
)
print("🚀 Все модели загружены!")


# ============================================================
# 3. ИНТЕРФЕЙС
# ============================================================
import gc, re, time, os, zipfile, traceback, subprocess, threading
import numpy as np, cv2, torch, gradio as gr
from PIL import Image, ImageOps

# ---------- Общие хелперы ----------
def _clean_name(raw):
    return re.sub(r"[^\w\-]", "", (raw or "").strip()).strip("_-")

def _p(prog, frac, desc):
    if prog is not None:
        try: prog(frac, desc=desc)
        except Exception: pass

def view_selected_image(evt: gr.SelectData, gallery_items):
    if gallery_items and evt.index < len(gallery_items):
        item = gallery_items[evt.index]
        if isinstance(item, (tuple, list)):
            return item[0]
        return item
    return None

def suggest_char_name(path):
    if not path:
        return gr.update()
    base = os.path.splitext(os.path.basename(str(path)))[0]
    clean = _clean_name(base)
    return clean if clean else gr.update()

def apply_sampler(pipe, sampler_name):
    if sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

def _eta_progress(progress, total_steps, base_frac, span_frac):
    t0 = time.time()
    total = int(total_steps)
    def cb(pipe, step, timestep, cb_kwargs):
        done = step + 1
        per_step = (time.time() - t0) / max(done, 1)
        eta = per_step * (total - done)
        _p(progress, base_frac + (done / total) * span_frac, f"Шаг {done}/{total} · осталось ~{eta:.0f} с")
        return cb_kwargs
    return cb

# ---------- Ядро face-swap (общее для обеих вкладок) ----------
def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

def get_source_face(source_image):
    """source_image — PIL.Image или путь к файлу. Возвращает лицо-источник либо None."""
    if isinstance(source_image, Image.Image):
        pil = ImageOps.exif_transpose(source_image).convert("RGB")
    else:
        pil = ImageOps.exif_transpose(Image.open(source_image)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    faces = face_app.get(cv_img)
    if not faces:
        return None
    return _largest_face(faces)

def apply_swap(target_cv, source_face, use_gfp, all_faces=True):
    """Свап на BGR-изображении. Возвращает (result_cv | None, кол-во замен, примечание)."""
    faces = face_app.get(target_cv)
    if not faces:
        return None, 0, ""
    chosen = list(faces) if all_faces else [_largest_face(faces)]
    result = target_cv.copy()
    for f in chosen:
        result = swapper.get(result, f, source_face, paste_back=True)
    gfp_note = ""
    if use_gfp:
        try:
            _, _, result = face_enhancer.enhance(result, has_aligned=False, only_center_face=False, paste_back=True)
        except Exception as e:
            gfp_note = f" · ⚠️ GFPGAN: {str(e)[:60]}"
    return result, len(chosen), gfp_note

# ---------- ВКЛАДКА 1: генерация (+ опциональный face-swap) ----------
def generate(prompt, negative_prompt, steps, cfg, seed, width, height, sampler, batch_size,
             do_swap, ref_image, use_gfp, all_faces, progress=gr.Progress()):
    seed = -1 if seed is None else seed
    actual_seed = torch.randint(0, 2**32 - 1, (1,)).item() if seed == -1 else int(seed)
    gallery_items = []
    nsfw_blocked = 0
    try:
        source_face = None
        if do_swap:
            if ref_image is None:
                return [], "❌ Включён face-swap, но фото-референс не загружен!", None
            _p(progress, 0.02, "Анализ лица на референсе...")
            source_face = get_source_face(ref_image)
            if source_face is None:
                return [], "❌ На фото-референсе лицо не найдено!", None

        apply_sampler(pipe_base, sampler)

        stamp = time.strftime("%m%d_%H%M%S")
        session_dir = os.path.join("/kaggle/tmp", f"gen_{stamp}")
        os.makedirs(session_dir, exist_ok=True)
        out_files = []

        for i in range(int(batch_size)):
            span = 0.75 if do_swap else 0.85
            base = 0.05 + (i / int(batch_size)) * span
            span = span / int(batch_size)

            with torch.inference_mode():
                raw = pipe_base(prompt=prompt, negative_prompt=negative_prompt,
                                num_inference_steps=int(steps), guidance_scale=float(cfg),
                                width=int(width), height=int(height),
                                generator=torch.Generator("cuda").manual_seed(actual_seed + i),
                                callback_on_step_end=_eta_progress(progress, steps, base, span),
                                callback_on_step_end_tensor_inputs=['latents']).images[0]

            _p(progress, base + span, "Проверка NSFW-фильтром...")
            out, flagged = nsfw_filter_image(raw)
            note = ""
            if flagged:
                nsfw_blocked += 1
                note = " · 🚫 NSFW → заблокировано"
            elif do_swap:
                _p(progress, base + span, f"Кадр {i+1}/{int(batch_size)}: face-swap...")
                cv_img = cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
                result, n_swapped, gfp_note = apply_swap(cv_img, source_face, use_gfp, all_faces)
                if result is not None:
                    out = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    note = f" · 🎭 лиц заменено: {n_swapped}" + gfp_note
                else:
                    note = " · ⚠️ лицо на кадре не найдено — свап пропущен"

            cap = f"Вариант {i+1} (Seed: {actual_seed + i})" + note
            gallery_items.append((out, cap))

            fp = os.path.join(session_dir, f"gen_{actual_seed + i}_{i+1}.png")
            out.save(fp)
            out_files.append(fp)
            torch.cuda.empty_cache(); gc.collect()

        zip_path = os.path.join(WORK, f"gen_results_{stamp}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in out_files:
                zf.write(fp, arcname=os.path.basename(fp))

        msg = f"✅ Готово! Сгенерировано {int(batch_size)} шт. (seed: {actual_seed})"
        if do_swap: msg += " · face-swap применён"
        if nsfw_blocked: msg += f" | 🚫 NSFW-фильтр: {nsfw_blocked}"
        msg += f" · архив: {os.path.basename(zip_path)}"
        return gallery_items, msg, zip_path
    except Exception as e:
        traceback.print_exc()
        return gallery_items, f"Ошибка: {e}", None

# ---------- ВКЛАДКА 2: face-swap на загруженные фото ----------
def faceswap_batch(source_path, files, char_name_raw, use_gfp, all_faces, progress=gr.Progress()):
    gallery_items = []
    try:
        if not source_path:
            return [], "❌ Загрузите фото-источник (чьё лицо берём)", None
        if not files:
            return [], "❌ Загрузите целевые фото (куда вставляем лицо)", None

        _p(progress, 0.02, "Анализ лица-источника...")
        source_face = get_source_face(source_path)
        if source_face is None:
            return [], "❌ Лицо-источник не найдено!", None

        prefix = _clean_name(char_name_raw)
        paths = [getattr(f, "name", f) for f in files if getattr(f, "name", f)]
        stamp = time.strftime("%m%d_%H%M%S")
        session_dir = os.path.join("/kaggle/tmp", f"session_{stamp}")
        os.makedirs(session_dir, exist_ok=True)

        n_ok, n_no_face, n_err = 0, 0, 0
        out_files = []
        for i, p in enumerate(paths):
            frac = 0.05 + 0.9 * (i / max(len(paths), 1))
            base = os.path.splitext(os.path.basename(p))[0]
            try:
                target_pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            except Exception as e:
                gallery_items.append((Image.new("RGB", (512, 512), (32, 32, 32)), f"{base} · ❌ файл не открылся: {e}"))
                n_err += 1
                continue

            target_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
            _p(progress, frac, f"Фото {i+1}/{len(paths)}: свап...")
            try:
                result_cv, n_swapped, gfp_note = apply_swap(target_cv, source_face, use_gfp, all_faces)
            except Exception as e:
                gallery_items.append((target_pil, f"{base} · ❌ ошибка свапа: {str(e)[:80]}"))
                n_err += 1
                continue
            if result_cv is None:
                gallery_items.append((target_pil, f"{base} · ❌ лицо не найдено"))
                n_no_face += 1
                continue

            result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
            fname = f"{prefix}_{base}_swap.png" if prefix else f"{base}_swap.png"
            out_path = os.path.join(session_dir, fname)
            result_pil.save(out_path)
            out_files.append(out_path)
            gallery_items.append((result_pil, f"{base} · ✅ лиц заменено: {n_swapped}{gfp_note}"))
            n_ok += 1
            gc.collect()
            torch.cuda.empty_cache()

        zip_name = f"face_swap_{prefix}_{stamp}.zip" if prefix else f"face_swap_results_{stamp}.zip"
        zip_path = os.path.join(WORK, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in out_files:
                zf.write(fp, arcname=os.path.basename(fp))

        msg = f"✅ Обработано: {n_ok}/{len(paths)}"
        if prefix: msg += f" · персонаж: {prefix}"
        if n_no_face: msg += f" · без лица: {n_no_face}"
        if n_err: msg += f" · с ошибками: {n_err}"
        msg += f" · архив: {zip_name}"
        return gallery_items, msg, zip_path
    except Exception as e:
        traceback.print_exc()
        return gallery_items, f"❌ Ошибка: {e}", None

# ---------- Веб-интерфейс ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 AI Studio: RealVisXL V5.0 + Face-Swap (InsightFace + GFPGAN)")
    gr.Markdown("**Вкладка 1** — генерация (опционально с наложением лица с референса). **Вкладка 2** — face-swap на загруженные фото.")

    with gr.Tabs():
        # ================= ГЕНЕРАЦИЯ =================
        with gr.Tab("🎨 Генерация изображений"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_in = gr.Textbox(label="Prompt", value="candid medium shot photograph of a beautiful woman with styled wavy platinum blonde hair, natural makeup, clean smooth skin, sitting by the window in a Parisian cafe, wearing a dark blazer, soft cinematic diffused morning light, shallow depth of field, 85mm lens, film grain")
                    neg_in = gr.Textbox(label="Negative Prompt", value="bright lipstick, dark lipstick, heavy makeup, wrinkles, eye bags, deep shadows on face, old, elderly, young, teenager, brunette, deformed, distorted, bad anatomy")

                    with gr.Accordion("🎭 Лицо с референса (face-swap на результат)", open=False):
                        do_swap_cb = gr.Checkbox(label="Наложить лицо с фото-референса на сгенерированное", value=False)
                        ref_img = gr.Image(label="Фото-референс (чьё лицо)", type="pil", height=260)
                        gfp_gen_cb = gr.Checkbox(label="GFPGAN-восстановление чёткости после свапа", value=True)
                        all_faces_gen_cb = gr.Checkbox(label="Заменять все лица на кадре (иначе — только главное)", value=True)

                    with gr.Row():
                        sampler_d = gr.Dropdown(choices=["DPM++ 2M Karras", "Euler a", "Euler"], value="DPM++ 2M Karras", label="Sampler")
                        steps_s = gr.Slider(minimum=10, maximum=50, step=1, value=30, label="Шаги (Steps)")
                    with gr.Row():
                        width_s = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Ширина")
                        height_s = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Высота")
                    with gr.Row():
                        cfg_s = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, value=4.5, label="CFG Scale")
                        batch_s = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Кол-во")
                        seed_n = gr.Number(label="Seed (-1 = случайно)", value=-1)
                    gen_btn = gr.Button("🎨 Сгенерировать изображение", variant="primary")

                with gr.Column(scale=1):
                    gen_gallery = gr.Gallery(label="Результаты", columns=2, height=500, object_fit="contain", elem_classes="gallery-container")
                    gen_status = gr.Textbox(label="Статус", interactive=False)
                    gen_viewer = gr.Image(label="Просмотр полного размера", type="pil", height=512)
                    gen_zip = gr.File(label="💾 Скачать результаты (ZIP)")

        # ================= FACE-SWAP =================
        with gr.Tab("🎭 Face-swap на загруженные фото"):
            with gr.Row():
                with gr.Column(scale=1):
                    src_input = gr.Image(label="📷 Чьё лицо взять? (Source — одно фото)", type="filepath", height=300)
                    char_name = gr.Textbox(label="🏷 Имя персонажа (подставится из имени файла — можно переименовать)",
                                           placeholder="например: alice")
                    files_in = gr.File(label="🖼 Куда вставить лицо? (Target — можно пачку)",
                                       file_count="multiple", file_types=["image"])
                    with gr.Accordion("⚙️ Настройки", open=False):
                        gfp_swap_cb = gr.Checkbox(label="GFPGAN-восстановление чёткости лица", value=True)
                        all_faces_swap_cb = gr.Checkbox(label="Заменять все лица (иначе — только главное)", value=True)
                    swap_btn = gr.Button("🎭 Пересадить лицо", variant="primary")

                with gr.Column(scale=1):
                    swap_gallery = gr.Gallery(label="Результаты (клик — превью)", columns=2, height=500,
                                              object_fit="contain", elem_classes="gallery-container")
                    swap_viewer = gr.Image(label="Просмотр полного размера", type="pil", height=512)
                    swap_status = gr.Textbox(label="Статус", interactive=False)
                    zip_out = gr.File(label="💾 Скачать все результаты (ZIP)")

    # ---------- События ----------
    gen_btn.click(fn=generate,
                  inputs=[prompt_in, neg_in, steps_s, cfg_s, seed_n, width_s, height_s, sampler_d, batch_s,
                          do_swap_cb, ref_img, gfp_gen_cb, all_faces_gen_cb],
                  outputs=[gen_gallery, gen_status, gen_zip])
    gen_gallery.select(fn=view_selected_image, inputs=gen_gallery, outputs=gen_viewer)

    src_input.upload(fn=suggest_char_name, inputs=src_input, outputs=char_name)
    swap_btn.click(fn=faceswap_batch,
                   inputs=[src_input, files_in, char_name, gfp_swap_cb, all_faces_swap_cb],
                   outputs=[swap_gallery, swap_status, zip_out])
    swap_gallery.select(fn=view_selected_image, inputs=swap_gallery, outputs=swap_viewer)

# ---------- Запуск (Gradio Share + резервный cloudflared) ----------
def _cloudflared_tunnel(port):
    try:
        cf_bin = "/kaggle/tmp/cloudflared"
        os.makedirs("/kaggle/tmp", exist_ok=True)
        if not os.path.exists(cf_bin):
            _download("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf_bin)
            os.chmod(cf_bin, 0o755)
        proc = subprocess.Popen([cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if "trycloudflare.com" in line:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m:
                    print(f"🌍 Резервный туннель: {m.group(0)}")
                    return
    except Exception as e:
        print(f"⚠️ cloudflared не удался: {e}")

def _free_port(port):
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=20)
        time.sleep(1.0)
    except Exception:
        pass

print("🚀 Запуск интерфейса...")
_free_port(PORT)
_launched = False
for _pt in (PORT, PORT + 1):
    try:
        demo.launch(server_name="127.0.0.1", server_port=_pt, share=True)
        _launched = True
        break
    except Exception as e:
        print(f"⚠️ Порт {_pt} не поднялся: {str(e)[:120]}")
if not _launched:
    print("⚠️ Резервный вариант: локальный сервер + cloudflared...")
    threading.Thread(target=demo.launch,
                     kwargs=dict(server_name="127.0.0.1", server_port=PORT + 2, share=False),
                     daemon=True).start()
    time.sleep(6)
    _cloudflared_tunnel(PORT + 2)

while True:
    time.sleep(3600)

if os.path.exists("photo.py"):
    os.remove("photo.py")
