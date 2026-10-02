# ================================================================
#  0. Установка: библиотеки + JS-runtime (deno) для yt-dlp
# ================================================================

# deno ставим напрямую с GitHub — официальный install.sh теперь
# задаёт интерактивный вопрос про PATH и виснет в Kaggle (нет stdin).
# Бинарник кладём в /usr/local/bin — этот путь уже есть в PATH.
import os
os.system("curl -fsSL -o /tmp/deno.zip https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip")


import zipfile
with zipfile.ZipFile("/tmp/deno.zip") as z:
    z.extractall("/usr/local/bin")
os.chmod("/usr/local/bin/deno", 0o755)

os.system("deno --version")
print("УСПЕШНО: Библиотеки и deno установлены!")

# ================================================================
#  1. Импорты и настройки
# ================================================================
import gradio as gr
import torch
import os
import re
import glob
import time
import yt_dlp
import subprocess
import math
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
MAX_PARAGRAPH_CHARS = 900  # максимальная длина одного абзаца (символов)

# ================================================================
#  2. Токен Hugging Face из Kaggle Secrets (если задан)
# ================================================================
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = None  # не Kaggle или секрет не создан — работаем без токена

if hf_token:
    print("HF_TOKEN найден — модель скачается быстро и без лимитов.")
else:
    print("HF_TOKEN не найден — качаем без токена (медленнее, возможны лимиты).")

# ================================================================
#  3. Whisper Large v3 Turbo + кеш в личном Kaggle-датасете
# ================================================================
# Приоритет источников модели:
#   1) /kaggle/input/whisper-cache — датасет подключён как Input: мгновенно;
#   2) /kaggle/working/whisper_model — уже скачана в этой сессии;
#   3) личный датасет через Kaggle API (нужен секрет KAGGLE_API);
#   4) Hugging Face — старый путь, если секрета нет.
import json
import shutil
import importlib.util
from huggingface_hub import snapshot_download

HF_MODEL_ID = "openai/whisper-large-v3-turbo"
DATASET_SLUG = "whisper-cache"                    # имя кеш-датасета
DATASET_DIR = f"/kaggle/input/{DATASET_SLUG}"     # путь, если подключён как Input
LOCAL_MODEL_DIR = os.path.join(WORK_DIR, "whisper_model")


def _run_kaggle(*args, timeout=1800):
    """Kaggle CLI через подпроцесс. Возвращает (успех, текст вывода)."""
    try:
        r = subprocess.run(["kaggle"] + list(args),
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return False, "kaggle CLI не найден"
    except subprocess.TimeoutExpired:
        return False, "таймаут команды kaggle"


def _get_kaggle_username():
    """Секрет KAGGLE_API (содержимое kaggle.json) -> username или None."""
    if importlib.util.find_spec("kaggle") is None:
        return None  # kaggle CLI не установлен (не Kaggle-среда)
    try:
        from kaggle_secrets import UserSecretsClient
        raw = UserSecretsClient().get_secret("KAGGLE_API")
    except Exception:
        return None  # секрета нет или он не прикреплён к ноутбуку
    if not raw or not raw.strip():
        return None
    try:
        creds = json.loads(raw.strip())
        username, key = creds["username"], creds["key"]
    except Exception:
        print("⚠️ Секрет KAGGLE_API заполнен, но это не содержимое kaggle.json — игнорируем.")
        return None
    # кладём kaggle.json туда, где его ждёт CLI
    kdir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kdir, exist_ok=True)
    kjson = os.path.join(kdir, "kaggle.json")
    with open(kjson, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kjson, 0o600)
    return username


def resolve_model_source():
    """
    Возвращает путь к папке с файлами модели или None
    (None = грузить с Hugging Face по имени, как раньше).
    """
    # 1) датасет подключён как Input
    if os.path.isfile(os.path.join(DATASET_DIR, "config.json")):
        print(f"✅ Модель берём из подключённого Input: {DATASET_DIR} (без скачивания)")
        return DATASET_DIR

    # 2) уже скачана в этой сессии (например, перезапуск ячейки)
    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        print(f"✅ Модель уже в локальной папке: {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR

    # 3) есть ли API-ключ Kaggle
    username = _get_kaggle_username()
    if not username:
        print("KAGGLE_API не задан (или не прикреплён) — модель качается с Hugging Face, как раньше.")
        return None

    dataset_id = f"{username}/{DATASET_SLUG}"

    # 3а) датасет уже существует — качаем модель из него
    ok, _ = _run_kaggle("datasets", "files", dataset_id, timeout=60)
    if ok:
        print(f"📦 Датасет {dataset_id} найден — скачиваем модель из него (мимо Hugging Face)...")
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        ok2, out2 = _run_kaggle("datasets", "download", dataset_id,
                                "-p", LOCAL_MODEL_DIR, "--unzip")
        if ok2 and os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
            print("✅ Модель скачана из Kaggle-датасета.")
            return LOCAL_MODEL_DIR
        print(f"⚠️ Скачать датасет не вышло: {out2[:200]} — качаем с Hugging Face.")
        shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)
        return None

    # 3б) датасета нет: качаем модель с HF и создаём кеш-датасет
    print("Кеш-датасета ещё нет — скачиваем модель с Hugging Face (один раз)...")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    snapshot_download(repo_id=HF_MODEL_ID, local_dir=LOCAL_MODEL_DIR, token=hf_token)
    shutil.rmtree(os.path.join(LOCAL_MODEL_DIR, ".cache"), ignore_errors=True)

    with open(os.path.join(LOCAL_MODEL_DIR, "dataset-metadata.json"), "w") as f:
        json.dump({"title": DATASET_SLUG, "id": dataset_id,
                   "licenses": [{"name": "CC0-1.0"}]}, f, indent=2)

    ok3, out3 = _run_kaggle("datasets", "create", "-p", LOCAL_MODEL_DIR, timeout=3600)
    if ok3:
        print(f"🎉 Создан кеш-датасет: https://www.kaggle.com/datasets/{dataset_id}")
        print("   Чтобы модель загружалась мгновенно, один раз подключите его:")
        print(f"   Add Input → Your Work + Datasets → {DATASET_SLUG} → Restart Session.")
        print("   А без подключения следующие сессии просто скачают модель с Kaggle —")
        print("   это быстрее HF и не зависит от токена и лимитов Hugging Face.")
    elif "already exist" in (out3 or "").lower():
        print("Датасет-кеш уже существует (видимо, создан только что) — используем локальную копию.")
    else:
        print(f"⚠️ Не удалось создать датасет-кеш: {out3[:300]}")
        print("   Не страшно: модель уже скачана локально, работаем как обычно.")

    return LOCAL_MODEL_DIR


model_dir = resolve_model_source()

print("Загрузка Whisper Large v3 Turbo...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model_dir if model_dir else HF_MODEL_ID,
    device=device,
    chunk_length_s=30,
    batch_size=8,  # при OutOfMemory уменьшите до 4
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=hf_token,
)
print("Whisper загружен!")

LANG_MAP = {
    "Автоопределение": None,
    "Русский": "russian",
    "English": "english",
    "Українська": "ukrainian",
    "Deutsch": "german",
    "Español": "spanish",
    "Français": "french",
}

DEFAULT_AI_PROMPT = """Ты — профессиональный редактор и аналитик. Ниже — транскрипция видео.

Сделай подробное структурированное саммари на русском языке в формате Markdown, строго по плану:

### 📌 О чём видео
(1–2 предложения: главная тема)

### 📝 Ключевые идеи и факты
(подробный список основных мыслей, аргументов и хода рассуждений. Пиши развёрнуто, не сокращай)

### 💡 Имена, цифры, примеры
(всё конкретное: имена, числа, инструменты, названия, примеры. Если ничего нет — так и напиши)

### 🎯 Выводы
(главный итог, к чему пришёл спикер)

Если в тексте есть таймкоды вида [MM:SS] — проставляй их у важных мыслей, чтобы можно было перейти к моменту в видео."""

# ================================================================
#  4. Вспомогательные функции
# ================================================================
def sec_to_time(seconds):
    """Секунды -> MM:SS или H:MM:SS"""
    if seconds is None:
        return "00:00"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def download_youtube(url, progress=None):
    """Скачивание с защитой от обрывов связи. Возвращает (путь, название)."""
    # Чистим остатки прошлых загрузок, чтобы не подхватить чужой файл
    for f in glob.glob(f"{WORK_DIR}/yt_audio.*"):
        try: os.remove(f)
        except OSError: pass

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": f"{WORK_DIR}/yt_audio.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "noplaylist": True,
        # --- устойчивость к обрывам ---
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,                # докачка частично скачанного файла
        "http_chunk_size": 10_000_000,     # кусками по 10 МБ — лечит троттлинг YouTube
        "socket_timeout": 30,
        "concurrent_fragment_downloads": 4,
    }

    last_err = None
    for attempt in range(1, 4):
        try:
            if progress:
                progress(0.05 + 0.05 * (attempt - 1),
                         desc=f"Скачиваем аудио (попытка {attempt}/3)...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "YouTube видео")
            return f"{WORK_DIR}/yt_audio.wav", title
        except yt_dlp.utils.DownloadError as e:
            last_err = e
            time.sleep(3)
    raise gr.Error(f"YouTube не отдал файл за 3 попытки. Последняя ошибка: {last_err}")

def get_audio_duration(path):
    """Длительность аудио в секундах через ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None

def _asr(audio_path, lang):
    """Вызов Whisper с языком или без. generate_kwargs=None в этой
    версии transformers падает, поэтому передаём его только когда язык задан."""
    kwargs = {"return_timestamps": True}
    if lang:
        kwargs["generate_kwargs"] = {"language": lang}
    return asr_pipeline(audio_path, **kwargs)

def transcribe_with_progress(audio_path, lang, progress, piece_sec=600):
    """
    Длинное аудио транскрибируем кусками по piece_sec секунд,
    обновляя прогресс после каждого куска. Короткое (< 20 мин) — одним вызовом.
    Возвращает сегменты с абсолютными таймкодами.
    """
    duration = get_audio_duration(audio_path)

    if not duration or duration <= 1200:
        progress(0.25, desc="Транскрибируем (это самое долгое)...")
        result = _asr(audio_path, lang)
        return result.get("chunks") or []

    n_pieces = math.ceil(duration / piece_sec)
    all_segments = []
    for i in range(n_pieces):
        start = i * piece_sec
        progress(0.25 + 0.55 * (i / n_pieces),
                 desc=f"Транскрибируем: кусок {i+1} из {n_pieces} (с {int(start // 60)}-й минуты)...")
        piece_path = os.path.join(WORK_DIR, f"_asr_piece_{i}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(piece_sec),
             "-i", audio_path, "-ar", "16000", "-ac", "1", piece_path],
            capture_output=True,
        )
        result = _asr(piece_path, lang)
        
        # сдвигаем таймкоды куска к абсолютному времени видео
        for seg in result.get("chunks") or []:
            ts = seg.get("timestamp") or (None, None)
            s = (ts[0] if ts[0] is not None else 0.0) + start
            e = (ts[1] + start) if ts[1] is not None else None
            all_segments.append({"timestamp": (s, e), "text": seg.get("text", "")})
        try:
            os.remove(piece_path)
        except OSError:
            pass
    return all_segments

def build_paragraphs(segments, pause=1.2, max_chars=MAX_PARAGRAPH_CHARS):
    """
    Склеиваем короткие сегменты Whisper в абзацы.
    Новый абзац — если пауза в речи длиннее `pause` секунд,
    либо абзац дорос до max_chars и закончился на конец предложения.
    Возвращает список (время_начала, текст).
    """
    paragraphs = []
    cur_text, cur_start, prev_end = "", None, None
    for seg in segments:
        ts = seg.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if start is None:
            start = prev_end if prev_end is not None else 0.0
        if cur_text:
            gap = start - (prev_end if prev_end is not None else start)
            ends_sentence = cur_text.rstrip()[-1:] in ".!?…"
            if gap > pause or (len(cur_text) > max_chars and ends_sentence):
                paragraphs.append((cur_start, cur_text))
                cur_text, cur_start = "", None
        if not cur_text:
            cur_start = start
        cur_text = f"{cur_text} {text}".strip()
        prev_end = end if end is not None else start
    if cur_text:
        paragraphs.append((cur_start, cur_text))
    return paragraphs


def format_transcript(paragraphs, with_timestamps, with_paragraphs):
    parts = []
    for start, text in paragraphs:
        prefix = f"[{sec_to_time(start)}] " if with_timestamps else ""
        parts.append(prefix + text)
    sep = "\n\n" if with_paragraphs else " "
    return sep.join(parts).strip()

# ================================================================
#  4.5. Файлы из Input (датасеты Kaggle) и из /kaggle/working
# ================================================================
INPUT_AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg",
}

def list_input_files():
    """
    Ищем аудио/видео файлы:
      * в /kaggle/input — датасеты, подключённые к ноутбуку (рекурсивно);
      * в WORK_DIR — например, yt_audio.wav, скачанный с YouTube ранее.
    Возвращает список пар (метка, полный путь) для Dropdown.
    """
    found = []

    inp = "/kaggle/input"
    if os.path.isdir(inp):
        for root, _, files in os.walk(inp):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in INPUT_AUDIO_EXTS:
                    p = os.path.join(root, fn)
                    found.append(("📘 " + os.path.relpath(p, inp), p))

    if os.path.isdir(WORK_DIR):
        for fn in sorted(os.listdir(WORK_DIR)):
            if fn.startswith("_asr_piece_"):
                continue  # технические куски от прошлой транскрибации
            if os.path.splitext(fn)[1].lower() in INPUT_AUDIO_EXTS:
                p = os.path.join(WORK_DIR, fn)
                found.append(("💾 " + fn + " (working)", p))

    return found


# ================================================================
#  5. Основной пайплайн
# ================================================================
def process_media(input_type, file_path, youtube_url, input_file_choice,
                  lang_choice, with_timestamps, with_paragraphs, pause_sec,
                  make_prompt, instruction, progress=gr.Progress()):

    # --- получаем аудио ---
    title = None
    if input_type == "YouTube":
        if not (youtube_url and youtube_url.strip()):
            raise gr.Error("Укажите ссылку на YouTube или выберите другой источник.")
        audio_path, title = download_youtube(youtube_url.strip(), progress)
    elif input_type == "Из загрузок (Input)":
        if not input_file_choice:
            raise gr.Error("Выберите файл в списке (или нажмите «Обновить список»).")
        if not os.path.exists(input_file_choice):
            raise gr.Error("Файл не найден — возможно, сессия перезапускалась. Нажмите «Обновить список».")
        audio_path = input_file_choice
        title = os.path.splitext(os.path.basename(input_file_choice))[0]
    else:
        audio_path = file_path
        if file_path:
            title = os.path.splitext(os.path.basename(file_path))[0]

    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("Загрузите файл или укажите ссылку.")

    # --- транскрибация ---
    lang = LANG_MAP.get(lang_choice)
    try:
        segments = transcribe_with_progress(audio_path, lang, progress)
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("Не хватило видеопамяти. Перезапустите ноутбук и уменьшите batch_size в коде (например, до 4).")

    if not segments:
        raise gr.Error("Whisper не распознал речь (возможно, в аудио только музыка или тишина).")

    # --- форматирование ---
    progress(0.85, desc="Форматируем текст...")
    paragraphs = build_paragraphs(segments, pause=pause_sec)
    transcript = format_transcript(paragraphs, with_timestamps, with_paragraphs)

    # --- готовый промпт для внешней большой модели ---
    prompt_text = ""
    if make_prompt:
        instr = (instruction or "").strip() or DEFAULT_AI_PROMPT
        head = f"Транскрипция видео: «{title}»\n\n" if title else ""
        prompt_text = f"{instr}\n\n---\n\n{head}{transcript}"

    # --- сохраняем в файл ---
    safe = re.sub(r"[^\w\-\s]", "", title or "transcript").strip()[:60] or "transcript"
    txt_path = os.path.join(WORK_DIR, f"{safe}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    progress(1.0, desc="Готово!")
    return transcript, prompt_text, txt_path

# ================================================================
#  6. Интерфейс Gradio
# ================================================================
with gr.Blocks() as app:
    gr.Markdown("# 🎬 Видео/Аудио → Текст (таймкоды и абзацы — опционально)")
    gr.Markdown("Транскрибация идёт локально через Whisper. Саммари делаете в большой модели: "
                "скопируйте готовый промпт из второй вкладки и вставьте в ChatGPT / Claude / Gemini.")

    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(
                ["Файл", "YouTube", "Из загрузок (Input)"],
                label="Откуда берём аудио?", value="YouTube")
            file_input = gr.Audio(label="Загрузите аудио/видео файл", type="filepath", visible=False)
            url_input = gr.Textbox(label="Ссылка на YouTube", value="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            input_file_dd = gr.Dropdown(
                choices=list_input_files(),
                label="Выбрать из загруженных",
                info="Файлы из /kaggle/input (подключённые датасеты) и ранее скачанные в /kaggle/working",
                visible=False,
            )
            lang_selector = gr.Dropdown(list(LANG_MAP.keys()), value="Автоопределение",
                                        label="Язык видео (явное указание повышает точность)")

            with gr.Accordion("⚙️ Опции текста", open=True):
                cb_timestamps = gr.Checkbox(value=False, label="⏱ Таймкоды [MM:SS] в начале абзацев")
                cb_paragraphs = gr.Checkbox(value=True, label="¶ Разбивать на абзацы (по паузам в речи)")
                pause_slider = gr.Slider(0.4, 3.0, value=1.2, step=0.1,
                                         label="Пауза для нового абзаца, сек",
                                         info="Насколько длинную тишину считать границей абзаца")

            with gr.Accordion("🤖 Промпт для внешней ИИ (ChatGPT / Claude / Gemini)", open=False):
                cb_prompt = gr.Checkbox(value=True, label="Собрать готовый промпт (инструкция + транскрипция)")
                prompt_instruction = gr.Textbox(
                    label="Инструкция для ИИ (можно отредактировать перед запуском)",
                    lines=10, value=DEFAULT_AI_PROMPT)

            process_btn = gr.Button("🚀 Получить текст", variant="primary")

        with gr.Column():
            with gr.Tabs():
                with gr.Tab("📜 Транскрипция"):
                    transcript_out = gr.Textbox(label="Текст", lines=18,
                                                interactive=False, show_copy_button=True)
                with gr.Tab("🤖 Промпт для большой модели"):
                    prompt_out = gr.Textbox(label="Скопируйте целиком и вставьте в веб-версию ИИ",
                                            lines=18, interactive=False, show_copy_button=True)
            file_out = gr.File(label="Скачать транскрипцию (.txt)")

    def toggle_inputs(choice):
        show_file = choice == "Файл"
        show_url = choice == "YouTube"
        show_dd = choice == "Из загрузок (Input)"
        return (gr.update(visible=show_file), gr.update(visible=show_url),
                gr.update(visible=show_dd))
    input_type.change(toggle_inputs, inputs=[input_type],
                      outputs=[file_input, url_input, input_file_dd])

    process_btn.click(
        fn=process_media,
        inputs=[input_type, file_input, url_input, input_file_dd, lang_selector,
                cb_timestamps, cb_paragraphs, pause_slider,
                cb_prompt, prompt_instruction],
        outputs=[transcript_out, prompt_out, file_out],
    )

app.launch(share=True, debug=True)

if os.path.exists("vtt.py"):
    os.remove("vtt.py")
