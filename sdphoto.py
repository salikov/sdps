# ============================================================
# ФИКС СОВМЕСТИМОСТИ diffusers ↔ torchao
# (обязателен ДО импорта diffusers!)
# В свежих образах Kaggle предустановленный diffusers при импорте пытается
# сделать `from torchao.quantization import FqnToConfig`, а предустановленный
# torchao слишком старый — импорт diffusers падает. torchao нам не нужен,
# поэтому удаляем его, и diffusers импортируется штатно.
# ============================================================
import importlib, importlib.util, subprocess, sys

def _torchao_conflicts():
    if importlib.util.find_spec("torchao") is None:
        return False                      # torchao нет — конфликта нет
    try:
        from torchao.quantization import FqnToConfig  # noqa: F401
        return False                      # torchao свежий — всё в порядке
    except Exception:
        return True                       # torchao старый — diffusers упадёт

if _torchao_conflicts():
    print("🔧 Конфликт diffusers ↔ torchao: удаляю устаревший torchao...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                   capture_output=True, text=True)
    for m in [k for k in list(sys.modules) if k == "torchao" or k.startswith("torchao.")]:
        sys.modules.pop(m, None)
    importlib.invalidate_caches()
    print("✅ torchao удалён")
else:
    print("✅ torchao не мешает — фикс не требуется")

# Если diffusers уже успел импортироваться в сессии с ошибкой — сбрасываем кэш
stale = [k for k in list(sys.modules) if k == "diffusers" or k.startswith("diffusers.")]
if stale:
    for m in stale:
        sys.modules.pop(m, None)
    importlib.invalidate_caches()

# ================== ОБЫЧНЫЕ ИМПОРТЫ ==================
import warnings, logging, os, gc, traceback, time
import torch, gradio as gr
import numpy as np
from PIL import Image
from diffusers import (StableDiffusionXLPipeline,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)
from huggingface_hub import hf_hub_download, list_repo_files

warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

custom_css = """
.gallery-container .grid-wrap { justify-content: flex-start !important; gap: 10px !important; }
.gallery-container .thumbnail-item { width: 120px !important; height: 120px !important; object-fit: contain !important; }
"""

WORK = "/kaggle/working"

def _get_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None

hf_token = _get_token()
assert torch.cuda.is_available(), "GPU не подключена!"

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

def _p(prog, frac, desc):
    if prog is not None:
        try: prog(frac, desc=desc)
        except Exception: pass

# ---------- 1. Базовая модель ----------
REALVIS_REPO = "SG161222/RealVisXL_V5.0"
REALVIS_FILE = "RealVisXL_V5.0_fp16.safetensors"
SDXL_CONFIG = "stabilityai/stable-diffusion-xl-base-1.0"

print("⏳ Загрузка базовой модели...")
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

# ---------- 2. NSFW-ФИЛЬТР (ПРИНУДИТЕЛЬНЫЙ) ----------
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

# ---------- 3. Сэмплеры и хелперы ----------
def apply_sampler(pipe, sampler_name):
    if sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

def view_selected_image(evt: gr.SelectData, gallery_items):
    """Извлекаем саму PIL-картинку из кортежа галереи (элемент с индексом 0)"""
    if gallery_items and evt.index < len(gallery_items):
        item = gallery_items[evt.index]
        if isinstance(item, (tuple, list)):
            return item[0]
        return item
    return None

def _eta_progress(progress, total_steps, base_frac, span_frac):
    """Пошаговый прогресс-бар с ETA внутри Gradio"""
    t0 = time.time()
    total = int(total_steps)
    def cb(pipe, step, timestep, cb_kwargs):
        done = step + 1
        per_step = (time.time() - t0) / max(done, 1)
        eta = per_step * (total - done)
        _p(progress, base_frac + (done / total) * span_frac, f"Шаг {done}/{total} · осталось ~{eta:.0f} с")
        return cb_kwargs
    return cb

# ---------- 4. ГЕНЕРАЦИЯ (С ПОШАГОВЫМ ПРОГРЕССОМ) ----------
def generate(prompt, negative_prompt, steps, cfg, seed, width, height, sampler, batch_size, progress=gr.Progress()):
    actual_seed = torch.randint(0, 2**32 - 1, (1,)).item() if seed == -1 else int(seed)
    gallery_items = []
    nsfw_blocked = 0
    try:
        apply_sampler(pipe_base, sampler)

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
        if nsfw_blocked: msg += f" | 🚫 NSFW-фильтр заблокировал изображений: {nsfw_blocked}"
        return gallery_items, msg

    except Exception as e:
        traceback.print_exc()
        return [], f"Ошибка: {e}"

# ---------- 5. ВЕБ-ИНТЕРФЕЙС ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 AI Studio: RealVisXL V5.0")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(label="Prompt", value="candid medium shot photograph of a beautiful woman with styled wavy platinum blonde hair, natural makeup, clean smooth skin, sitting by the window in a Parisian cafe, wearing a dark blazer, soft cinematic diffused morning light, shallow depth of field, 85mm lens, film grain")
            neg_in = gr.Textbox(label="Negative Prompt", value="bright lipstick, dark lipstick, heavy makeup, wrinkles, eye bags, deep shadows on face, old, elderly, young, teenager, brunette, deformed, distorted, bad anatomy")

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
            gallery = gr.Gallery(label="Результаты", columns=2, height=500, object_fit="contain", elem_classes="gallery-container")
            status = gr.Textbox(label="Статус", interactive=False)
            viewer = gr.Image(label="Просмотр полного размера", type="pil", height=512)

    # Связываем элементы интерфейса
    gen_btn.click(fn=generate, inputs=[prompt_in, neg_in, steps_s, cfg_s, seed_n, width_s, height_s, sampler_d, batch_s], outputs=[gallery, status])

    # Просмотр выбранного изображения из галереи в полном размере
    gallery.select(fn=view_selected_image, inputs=gallery, outputs=viewer)

PORT = 7860
demo.launch(server_name="127.0.0.1", server_port=PORT, share=True)

if os.path.exists("sdph.py"):
    os.remove("sdph.py")
