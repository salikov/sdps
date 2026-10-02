# !pip install -q -U transformers diffusers accelerate gradio
# !pip install -q Cython
# !pip install -q insightface onnxruntime --no-build-isolation || (pip install -q "numpy<2" && pip install -q insightface onnxruntime --no-build-isolation)

import warnings, logging, os, gc, re, sys, math, glob, inspect, shutil, zipfile, traceback, subprocess, threading, time
import torch, gradio as gr, urllib.request
import numpy as np, cv2
from PIL import Image, ImageOps
from diffusers import (StableDiffusionXLPipeline, ControlNetModel,
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
IFR_ROOT = os.path.join(WORK, "insightface_models")
INSTANTID_HF_REPO = "InstantX/InstantID"
INSTANTID_CODE_DIR = os.path.join(WORK, "instantid_code")
INSTANTID_FILE = "pipeline_stable_diffusion_xl_instantid.py"

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

# ---------- 1.5. NSFW-ФИЛЬТР (ПРИНУДИТЕЛЬНЫЙ) ----------
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

# ---------- 2. Инициализация InsightFace ----------
face_app = None
def ensure_face_app(progress=None):
    global face_app
    if face_app is None:
        from insightface.app import FaceAnalysis
        
        # Строим правильные целевые пути
        model_dir = os.path.join(IFR_ROOT, "models", "antelopev2")
        det_model_path = os.path.join(model_dir, "scrfd_10g_bnkps.onnx")
        
        # Если файла детекции нет, качаем архив по вашей рабочей ссылке
        if not os.path.exists(det_model_path):
            _p(progress, 0.1, "Скачивание и подготовка моделей лица (antelopev2)...")
            os.makedirs(model_dir, exist_ok=True)
            zip_path = os.path.join(IFR_ROOT, "antelopev2.zip")
            
            # Очищаем битый файл, если он остался от прошлой попытки
            if os.path.exists(zip_path):
                os.remove(zip_path)
            
            # Скачиваем архив через urllib по проверенной ссылке
            print("⏳ Скачивание моделей лица (350 МБ) с официального зеркала...")
            url = "https://github.com/deepinsight/insightface/releases/download/model-zoo/antelopev2.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            
            # Распаковываем файлы через встроенный zipfile Python
            print("📦 Распаковка моделей лица...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".onnx"):
                        filename = os.path.basename(member)
                        with zf.open(member) as source, open(os.path.join(model_dir, filename), "wb") as target:
                            shutil.copyfileobj(source, target)
            
            # Удаляем временный архив
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print("✅ Все 5 ONNX моделей успешно подготовлены!")
            
        face_app = FaceAnalysis(name="antelopev2", root=IFR_ROOT, providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return face_app

# ---------- 3. Загрузка InstantID + IP-Adapter Пайплайна ----------
pipe_instant = None

def ensure_instantid(progress=None):
    global pipe_instant
    if pipe_instant is not None:
        return pipe_instant
    torch.cuda.empty_cache(); gc.collect()
    
    ensure_face_app(progress)
    
    _p(progress, 0.2, "Загрузка ControlNet InstantID...")
    controlnet = ControlNetModel.from_pretrained(INSTANTID_HF_REPO, subfolder="ControlNetModel", torch_dtype=torch.float16, token=hf_token)
    
    _p(progress, 0.4, "Сборка динамического пайплайна через Hugging Face Community...")
    # ФИКС: Используем встроенный механизм загрузки кастомных пайплайнов Diffusers с Hugging Face
    from diffusers import DiffusionPipeline
    
    # Извлекаем все готовые компоненты из нашей уже загруженной базовой модели RealVisXL
    comps = dict(pipe_base.components)
    comps["controlnet"] = controlnet
    
    # Динамически собираем InstantID пайплайн в обход локальных импортов пакета
    pipe_instant = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", # Нужен для внутренней конфигурации схемы
        custom_pipeline="pipeline_stable_diffusion_xl_instantid",
        torch_dtype=torch.float16,
        token=hf_token,
        **comps
    )
    
    _p(progress, 0.6, "Загрузка весов InstantID...")
    adapter_path = hf_hub_download(INSTANTID_HF_REPO, "ip-adapter.bin", token=hf_token)
    pipe_instant.load_ip_adapter_instantid(adapter_path)
    
    # Отключаем встроенные проверки входных аргументов, которые могут конфликтовать в community-файле
    try: pipe_instant.check_inputs = lambda *a, **k: None
    except Exception: pass
    
    pipe_instant.to("cuda")
    torch.cuda.empty_cache()
    return pipe_instant

def draw_kps(image_pil, kps, color_list=((255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255))):
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

def detect_face(pil_image):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    faces = face_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    if not faces:
        return None, None
    # Фикс индексов: Выбираем самое крупное лицо по площади рамки детекции
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
    return emb, draw_kps(crop, kps), crop

def apply_sampler(pipe, sampler_name):
    if sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

def view_selected_image(evt: gr.SelectData, gallery_items):
    """ФИКС: Извлекаем строго саму PIL-картинку из кортежа галереи (элемент с индексом 0)"""
    if gallery_items and evt.index < len(gallery_items):
        item = gallery_items[evt.index]
        if isinstance(item, (tuple, list)):
            return item[0]  # Возвращаем только изображение, отсекая подпись
        return item
    return None

def _eta_progress(progress, total_steps, base_frac, span_frac):
    """Возвращаем пошаговый прогресс-бар для отображения процентов внутри Gradio"""
    t0 = time.time()
    total = int(total_steps)
    def cb(pipe, step, timestep, cb_kwargs):
        done = step + 1
        per_step = (time.time() - t0) / max(done, 1)
        eta = per_step * (total - done)
        _p(progress, base_frac + (done / total) * span_frac, f"Шаг {done}/{total} · осталось ~{eta:.0f} с")
        return cb_kwargs
    return cb

# ---------- 4. ГЕНЕРАЦИЯ (БЕЗОПАСНАЯ, С ПОШАГОВЫМ ПРОГРЕССОМ) ----------
def generate(prompt, negative_prompt, steps, cfg, seed, width, height, sampler, batch_size,
             use_face, face_image, id_scale, struct_scale, progress=gr.Progress()):
    actual_seed = torch.randint(0, 2**32 - 1, (1,)).item() if seed == -1 else int(seed)
    gallery_items = []
    nsfw_blocked = 0
    try:
        if use_face:
            if face_image is None: return [], "❌ Фото не загружено!"
            pipe = ensure_instantid(progress)
            _p(progress, 0.1, "Анализ структуры лица...")
            face, img = detect_face(face_image)
            if face is None: return [], "❌ Лицо не найдено!"
            
            emb, kps_img, _ = prepare_face_inputs(face, img)
            apply_sampler(pipe, sampler)
            
            try: pipe.set_ip_adapter_scale(float(id_scale))
            except Exception: pass
                
            has_cb = "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
            for i in range(int(batch_size)):
                base = 0.2 + (i / int(batch_size)) * 0.6
                span = 0.6 / int(batch_size)
                
                call_kwargs = {}
                if has_cb:
                    # Включаем пошаговый прогресс-бар для InstantID
                    call_kwargs["callback_on_step_end"] = _eta_progress(progress, steps, base, span)
                    call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                
                with torch.inference_mode():
                    raw = pipe(prompt=prompt, negative_prompt=negative_prompt,
                               image_embeds=emb, image=kps_img,                       
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
            if nsfw_blocked: msg += f" | 🚫 NSFW-фильтр заблокировал изображений: {nsfw_blocked}"
            return gallery_items, msg
        else:
            # ============ ОБЫЧНАЯ ГЕНЕРАЦИЯ БЕЗ ФОТО ============
            apply_sampler(pipe_base, sampler)
            if pipe_instant is not None:
                try: pipe_instant.set_ip_adapter_scale(0.0)
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
            if nsfw_blocked: msg += f" | 🚫 NSFW-фильтр заблокировал изображений: {nsfw_blocked}"
            return gallery_items, msg
            
    except Exception as e:
        traceback.print_exc()
        return [], f"Ошибка: {e}"

# ---------- 5. ОБНОВЛЕННЫЙ ВЕБ-ИНТЕРФЕЙС ----------
ensure_face_app()
ensure_instantid()

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 AI Studio: InstantID + RealVisXL V5.0")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(label="Prompt", value="candid medium shot photograph of a beautiful woman with styled wavy platinum blonde hair, natural makeup, clean smooth skin, sitting by the window in a Parisian cafe, wearing a dark blazer, soft cinematic diffused morning light, shallow depth of field, 85mm lens, film grain")
            neg_in = gr.Textbox(label="Negative Prompt", value="bright lipstick, dark lipstick, heavy makeup, wrinkles, eye bags, deep shadows on face, old, elderly, young, teenager, brunette, deformed, distorted, bad anatomy")
            
            # ФИКС: По умолчанию настройки свернуты (open=False)
            with gr.Accordion("👤 Настройки персонажа с фото", open=False):
                # ФИКС: Чекбокс по умолчанию снят (value=False)
                use_face_cb = gr.Checkbox(label="Использовать персонажа с фото", value=False)
                face_img = gr.Image(label="Фото лица", type="pil", height=300)
                with gr.Row():
                    id_scale_s = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.80, label="Сходство лица (ID)")
                    struct_scale_s = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.45, label="Точность структуры лица")
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
    gen_btn.click(fn=generate, inputs=[prompt_in, neg_in, steps_s, cfg_s, seed_n, width_s, height_s, sampler_d, batch_s, use_face_cb, face_img, id_scale_s, struct_scale_s], outputs=[gallery, status])
    
    # ФИКС: Передаем в inputs саму галерею, чтобы функция view_selected_image имела к ней доступ
    gallery.select(fn=view_selected_image, inputs=gallery, outputs=viewer)

PORT = 7860
demo.launch(server_name="127.0.0.1", server_port=PORT, share=True)

if os.path.exists("sdph.py"):
    os.remove("sdph.py")
