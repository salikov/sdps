import os, sys, subprocess

# ================== ЧАСТЬ 1. УСТАНОВКА ==================
def _run_quiet(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print("❌ Команда не удалась:", " ".join(str(c) for c in cmd[:5]), "…")
        print(((r.stdout or "") + "\n" + (r.stderr or ""))[-3000:])
        raise RuntimeError("Установка прервана — см. вывод выше")
    return r

def _already_installed():
    """Полная цепочка импортируется и onnxruntime с CUDA-провайдером?"""
    try:
        import gfpgan, insightface  # noqa: F401
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

if _already_installed():
    print("✅ Зависимости уже установлены — установка пропущена")
else:
    BSR = "/kaggle/working/BasicSR"

    print("⏳ Шаг 1/4: клон BasicSR + патч setup.py...")
    if not os.path.isdir(BSR):
        _run_quiet(["git", "clone", "https://github.com/XPixelGroup/BasicSR.git", BSR])
    setup_py = os.path.join(BSR, "setup.py")
    with open(setup_py, "r") as f:
        _code = f.read()
    if "version=get_version()," in _code:
        with open(setup_py, "w") as f:
            f.write(_code.replace("version=get_version(),", "version='1.4.2',"))
    print("✅ Шаг 1/4 готов: BasicSR склонирован и пропатчен")

    print("⏳ Шаг 2/4: зависимости BasicSR (requirements.txt)...")
    _run_quiet([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=BSR)
    print("✅ Шаг 2/4 готов: зависимости установлены")

    print("⏳ Шаг 3/4: установка BasicSR (setup.py develop, без CUDA-компиляции)...")
    _run_quiet([sys.executable, "setup.py", "develop"], cwd=BSR)
    print("✅ Шаг 3/4 готов: BasicSR установлен")

    print("⏳ Шаг 4/4: gfpgan, insightface, onnxruntime-gpu, gradio...")
    _run_quiet([sys.executable, "-m", "pip", "install",
                "gfpgan", "insightface", "onnxruntime-gpu", "opencv-python", "pillow", "gradio"])
    print("✅ Шаг 4/4 готов: пакеты установлены")
    print("🎉 Установка завершена")

# ================== ЧАСТЬ 2. ПРИЛОЖЕНИЕ ==================
import re, gc, time, shutil, zipfile, traceback, threading, urllib.request
import numpy as np, cv2, torch, gradio as gr
from PIL import Image, ImageOps

sys.path.append('/kaggle/working/BasicSR')   # gfpgan импортирует basicsr из локальной установки
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer

custom_css = """
.gallery-container .grid-wrap { justify-content: flex-start !important; gap: 10px !important; }
.gallery-container .thumbnail-item { width: 120px !important; height: 120px !important; object-fit: contain !important; }
"""

WORK = "/kaggle/working"
PORT = 7860

# ---------- HF-токен (тихо; нужен только для скачивания inswapper) ----------
def _get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

hf_token = _get_secret("HF_TOKEN")   # опционален: зеркало публичное

def _p(prog, frac, desc):
    if prog is not None:
        try: prog(frac, desc=desc)
        except Exception: pass

def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest + ".part", "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(dest + ".part", dest)

def _clean_name(raw):
    """Имя персонажа -> безопасный префикс для имён файлов (буквы/цифры/-/_)."""
    return re.sub(r"[^\w\-]", "", (raw or "").strip()).strip("_-")

# ---------- Веса моделей ----------
GFPGAN_PATH = os.path.join(WORK, "GFPGANv1.4.pth")
INSP_PATH = os.path.join(WORK, "inswapper_128.onnx")

if not os.path.isfile(GFPGAN_PATH):
    print("⏳ Скачивание GFPGANv1.4.pth (~350 МБ)...")
    _download("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth", GFPGAN_PATH)
    print("✅ GFPGANv1.4.pth готов")

if not os.path.isfile(INSP_PATH) or os.path.getsize(INSP_PATH) < 100 << 20:
    print("⏳ Скачивание inswapper_128.onnx (~526 МБ) с Hugging Face (Gourieff/ReActor)...")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="Gourieff/ReActor", filename="models/inswapper_128.onnx",
                            repo_type="dataset", token=hf_token)
        shutil.copyfile(p, INSP_PATH)
    except Exception as e:
        print(f"⚠️ hf_hub_download не сработал: {str(e)[:120]} — фолбэк на прямой URL")
        _download("https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
                  INSP_PATH)
    print("✅ inswapper_128.onnx готов")

assert os.path.getsize(INSP_PATH) > 100 << 20, "inswapper_128.onnx не скачался целиком — перезапустите ячейку"

# ---------- Инициализация моделей на GPU ----------
print("Инициализация моделей на GPU...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(os.path.abspath(INSP_PATH), download=False)
face_enhancer = GFPGANer(
    model_path=os.path.abspath(GFPGAN_PATH),
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    device='cuda'
)
print("Все модели (включая GFPGAN) успешно загружены!")

# ---------- Просмотр из галереи ----------
def view_selected_image(evt: gr.SelectData, gallery_items):
    if gallery_items and evt.index < len(gallery_items):
        item = gallery_items[evt.index]
        if isinstance(item, (tuple, list)):
            return item[0]
        return item
    return None

# ---------- Авто-имя персонажа из имени файла источника ----------
def suggest_char_name(path):
    """Срабатывает при ЗАГРУЗКЕ файла источника: имя файла -> имя персонажа.
    При редактировании фото/очистке поле не трогаем — переименование не потеряется."""
    if not path:
        return gr.update()
    base = os.path.splitext(os.path.basename(str(path)))[0]
    clean = _clean_name(base)
    return clean if clean else gr.update()

# ---------- Пакетная обработка ----------
def faceswap_batch(source_path, files, char_name_raw, use_gfp, progress=gr.Progress()):
    gallery_items = []
    try:
        if not source_path:
            return [], "❌ Загрузите фото-источник (чьё лицо берём)", None
        if not files:
            return [], "❌ Загрузите целевые фото (куда вставляем лицо)", None

        _p(progress, 0.02, "Анализ лица-источника...")
        source_pil = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
        source_cv = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
        source_faces = app.get(source_cv)
        if not source_faces:
            return [], "❌ Лицо-источник не найдено!", None
        source_face = source_faces[0]   # детектор возвращает лица по убыванию площади
        if len(source_faces) > 1:
            print(f"ℹ️ На источнике найдено лиц: {len(source_faces)} — используется крупнейшее")

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
                gallery_items.append((Image.new("RGB", (512, 512), (32, 32, 32)),
                                      f"{base} · ❌ файл не открылся: {e}"))
                n_err += 1
                continue

            target_cv = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
            _p(progress, frac, f"Фото {i+1}/{len(paths)}: детекция лиц...")
            target_faces = app.get(target_cv)
            if not target_faces:
                gallery_items.append((target_pil, f"{base} · ❌ лицо не найдено"))
                n_no_face += 1
                continue

            try:
                # Шаг 1: замена лица (получаем размытые 128x128)
                _p(progress, frac + 0.02, f"Фото {i+1}/{len(paths)}: свап ({len(target_faces)} лиц)...")
                result_cv = target_cv.copy()
                for target_face in target_faces:
                    result_cv = swapper.get(result_cv, target_face, source_face, paste_back=True)

                # Шаг 2: восстановление чёткости (GFPGAN сам находит заменённые лица)
                gfp_note = ""
                if use_gfp:
                    _p(progress, frac + 0.04, f"Фото {i+1}/{len(paths)}: GFPGAN-восстановление...")
                    try:
                        _, _, result_cv = face_enhancer.enhance(
                            result_cv, has_aligned=False, only_center_face=False, paste_back=True)
                    except Exception as e:
                        gfp_note = f" · ⚠️ GFPGAN не сработал ({str(e)[:60]})"

                result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
                fname = f"{prefix}_{base}_swap.png" if prefix else f"{base}_swap.png"
                out_path = os.path.join(session_dir, fname)
                result_pil.save(out_path)
                out_files.append(out_path)
                gallery_items.append((result_pil, f"{base} · ✅ лиц заменено: {len(target_faces)}{gfp_note}"))
                n_ok += 1
            except Exception as e:
                gallery_items.append((target_pil, f"{base} · ❌ ошибка свапа: {str(e)[:80]}"))
                n_err += 1
            gc.collect()
            torch.cuda.empty_cache()

        zip_name = f"face_swap_{prefix}_{stamp}.zip" if prefix else f"face_swap_results_{stamp}.zip"
        zip_path = os.path.join(WORK, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in out_files:
                zf.write(fp, arcname=os.path.basename(fp))

        msg = f"✅ Обработано: {n_ok}/{len(paths)}"
        if prefix:
            msg += f" · персонаж: {prefix}"
        if n_no_face: msg += f" · без лица: {n_no_face}"
        if n_err:     msg += f" · с ошибками: {n_err}"
        msg += f" · архив: {zip_name}"
        return gallery_items, msg, zip_path
    except Exception as e:
        traceback.print_exc()
        return gallery_items, f"❌ Ошибка: {e}", None

# ---------- Веб-интерфейс ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎭 Face-Swap HD — InsightFace + GFPGAN v1.4 (GPU T4)")
    gr.Markdown("Лицо с одного референса пересаживается на все загруженные фото; чёткость восстанавливает GFPGAN.")
    with gr.Row():
        with gr.Column(scale=1):
            src_input = gr.Image(label="📷 Чьё лицо взять? (Source — одно фото)",
                                 type="filepath", height=300)
            char_name = gr.Textbox(
                label="🏷 Имя персонажа (подставится из имени файла источника — можно переименовать)",
                placeholder="например: alice")
            files_in = gr.File(label="🖼 Куда вставить лицо? (Target — можно пачку)",
                               file_count="multiple", file_types=["image"])
            with gr.Accordion("⚙️ Настройки", open=False):
                use_gfp_cb = gr.Checkbox(label="GFPGAN-восстановление чёткости лица", value=True)
            submit_btn = gr.Button("🎭 Пересадить лицо", variant="primary")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Результаты (клик — превью)", columns=2, height=500,
                                 object_fit="contain", elem_classes="gallery-container")
            viewer = gr.Image(label="Просмотр полного размера", type="pil", height=512)
            status = gr.Textbox(label="Статус", interactive=False)
            zip_out = gr.File(label="💾 Скачать все результаты (ZIP)")

    src_input.upload(fn=suggest_char_name, inputs=src_input, outputs=char_name)
    submit_btn.click(fn=faceswap_batch,
                     inputs=[src_input, files_in, char_name, use_gfp_cb],
                     outputs=[gallery, status, zip_out])
    gallery.select(fn=view_selected_image, inputs=gallery, outputs=viewer)

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
    print("⚠️ Включаю резервный вариант (локальный сервер + cloudflared)...")
    threading.Thread(target=demo.launch,
                     kwargs=dict(server_name="127.0.0.1", server_port=PORT + 2, share=False),
                     daemon=True).start()
    time.sleep(6)
    _cloudflared_tunnel(PORT + 2)

if os.path.exists("fasw.py"):
    os.remove("fasw.py")

while True:
    time.sleep(3600)
