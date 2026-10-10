#!pip install -q -U yt-dlp
#!pip install -q transformers accelerate gradio
# =======================

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
import gc              # очистка памяти между файлами пакета
import time
import yt_dlp
import subprocess
import math
import zipfile
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
MAX_PARAGRAPH_CHARS = 900  # максимальная длина одного абзаца (символов)

# Папка, куда складываются тексты пакетной обработки
TRANSCRIPTS_DIR = os.path.join(WORK_DIR, "transcripts")
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ошибка «не хватило видеопамяти» (перехватываем её, не прерывая очередь)
OOMError = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

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
#  3. Whisper Large v3 Turbo — просто качаем с Hugging Face
# ================================================================
# Кеш-датасет whisper-cache больше не используется: монтирование Input
# при старте сессии медленнее, чем скачать модель с HF напрямую.
# Повторные запуски этой ячейки в той же сессии не качают модель заново:
# huggingface_hub сам кеширует файлы в ~/.cache/huggingface.
HF_MODEL_ID = "openai/whisper-large-v3-turbo"

print("Загрузка Whisper Large v3 Turbo с Hugging Face...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=HF_MODEL_ID,
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
    Возвращает список пар (метка, полный путь) для мультивыбора.
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
#  5. Пакетный пайплайн: очередь из нескольких источников
# ================================================================
def collect_tasks(input_type, file_paths, youtube_urls, input_choices):
    """Собираем очередь задач: список пар (тип, источник)."""
    tasks = []
    if input_type == "YouTube":
        for line in (youtube_urls or "").splitlines():
            u = line.strip()
            if u:
                tasks.append(("youtube", u))
    elif input_type == "Из загрузок (Input)":
        if isinstance(input_choices, str):
            input_choices = [input_choices]
        for path in (input_choices or []):
            tasks.append(("path", path))
    else:  # "Файлы"
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for path in (file_paths or []):
            tasks.append(("path", path))
    return tasks


def unique_txt_path(title):
    """Путь для нового .txt с безопасным уникальным именем (без перезаписи)."""
    safe = re.sub(r"[^\w\-\s]", "", title or "transcript").strip()[:60].strip() or "transcript"
    path = os.path.join(TRANSCRIPTS_DIR, f"{safe}.txt")
    i = 2
    while os.path.exists(path):
        path = os.path.join(TRANSCRIPTS_DIR, f"{safe}_{i}.txt")
        i += 1
    return path


def load_result(txt_path):
    """Загружаем сохранённые транскрипцию и промпт по пути .txt-файла."""
    transcript, prompt = "", ""
    if txt_path and os.path.isfile(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            transcript = f.read()
        prompt_path = os.path.splitext(txt_path)[0] + "_prompt.txt"
        if os.path.isfile(prompt_path):
            with open(prompt_path, encoding="utf-8") as f:
                prompt = f.read()
    return transcript, prompt


def build_zip():
    """Архив из всех .txt папки транскрипций (сами тексты + промпты)."""
    files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")))
    if not files:
        return None
    zip_path = os.path.join(WORK_DIR, "transcripts.zip")
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except OSError:
            pass
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.basename(f))
    return zip_path


def process_batch(input_type, file_paths, youtube_urls, input_choices,
                  lang_choice, with_timestamps, with_paragraphs, pause_sec,
                  make_prompt, instruction, progress=gr.Progress()):
    """
    Обрабатывает ОЧЕРЕДЬ файлов/ссылок по одному:
    скачивание → транскрибация → форматирование → сохранение в отдельный .txt.
    Ошибка на одном файле не останавливает остальные.
    """
    tasks = collect_tasks(input_type, file_paths, youtube_urls, input_choices)
    if not tasks:
        raise gr.Error("Не выбрано ни одного файла и не указано ни одной ссылки.")

    lang = LANG_MAP.get(lang_choice)
    n = len(tasks)
    rows = []      # строки таблицы результатов
    choices = []   # варианты для просмотра в дропдауне
    ok_count = 0

    for idx, (kind, src) in enumerate(tasks):
        # локальный прогресс файла -> общий прогресс очереди
        def p(frac, desc="", _idx=idx, _n=n):
            frac = min(max(frac, 0.0), 1.0)
            progress((_idx + frac) / _n, desc=f"Файл {_idx + 1}/{_n}: {desc}")

        # предварительное имя — чтобы ошибки тоже попали в таблицу с понятной подписью
        title = (f"YouTube #{idx + 1}" if kind == "youtube"
                 else os.path.splitext(os.path.basename(src))[0])

        try:
            # --- 1. получаем аудио ---
            if kind == "youtube":
                p(0.02, f"скачиваем: {src[:60]}…")
                audio_path, title = download_youtube(src, p)
            else:
                if not src or not os.path.exists(src):
                    raise gr.Error(f"файл не найден: {src}")
                audio_path = src

            # --- 2. транскрибация ---
            p(0.2, f"«{title[:50]}»: транскрибация (самое долгое)…")
            segments = transcribe_with_progress(audio_path, lang, p)
            if not segments:
                raise gr.Error("Whisper не распознал речь (возможно, только музыка или тишина).")

            # --- 3. форматирование ---
            p(0.9, f"«{title[:50]}»: форматируем текст…")
            paragraphs = build_paragraphs(segments, pause=pause_sec)
            transcript = format_transcript(paragraphs, with_timestamps, with_paragraphs)

            # --- 4. промпт для внешней большой модели ---
            prompt_text = ""
            if make_prompt:
                instr = (instruction or "").strip() or DEFAULT_AI_PROMPT
                head = f"Транскрипция видео: «{title}»\n\n" if title else ""
                prompt_text = f"{instr}\n\n---\n\n{head}{transcript}"

            # --- 5. сохраняем в ОТДЕЛЬНЫЙ файл ---
            txt_path = unique_txt_path(title)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            if prompt_text:
                prompt_path = os.path.splitext(txt_path)[0] + "_prompt.txt"
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)

            rows.append([str(idx + 1), title, "✅ готово", os.path.basename(txt_path)])
            choices.append((title, txt_path))
            ok_count += 1

            # --- 6. чистим память перед следующим файлом очереди ---
            del segments, paragraphs
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        except OOMError:
            if device == "cuda":
                torch.cuda.empty_cache()
            rows.append([str(idx + 1), title,
                         "❌ не хватило видеопамяти — уменьшите batch_size в коде", ""])
        except Exception as e:
            msg = getattr(e, "message", None) or str(e) or type(e).__name__
            rows.append([str(idx + 1), title, f"❌ {msg[:300]}", ""])

    err_count = n - ok_count
    status = f"**Готово: {ok_count} из {n}.**" + \
             (f" Не обработано: {err_count} — детали в таблице." if err_count else "")
    progress(1.0, desc="Готово!")

    # --- вывод: таблица, просмотр, файлы, архив ---
    first_txt = choices[0][1] if choices else None
    transcript_text, prompt_text = load_result(first_txt)

    transcripts_all = sorted(
        f for f in glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt"))
        if not f.endswith("_prompt.txt"))
    dd_choices = [(os.path.splitext(os.path.basename(f))[0], f) for f in transcripts_all]

    all_txt = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")))
    zip_path = build_zip() if choices else None

    return (rows, status,
            gr.update(choices=dd_choices, value=first_txt),
            transcript_text, prompt_text,
            all_txt, zip_path)


# ================================================================
#  6. Интерфейс Gradio
# ================================================================
with gr.Blocks() as app:
    gr.Markdown("# 🎬 Видео/Аудио → Текст — пакетная обработка")
    gr.Markdown(
        "Транскрибация идёт локально через Whisper. Можно закинуть сразу несколько файлов, "
        "несколько ссылок на YouTube или выбрать несколько файлов из Input — они обработаются "
        "по очереди, каждый текст сохранится в отдельный .txt, всё скачивается одним архивом. "
        "Саммари делаете в большой модели: скопируйте готовый промпт из второй вкладки.")

    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(
                ["Файлы", "YouTube", "Из загрузок (Input)"],
                label="Откуда берём аудио?", value="YouTube")
            file_input = gr.File(
                label="Загрузите файлы (аудио/видео, можно несколько)",
                file_count="multiple",
                file_types=sorted(INPUT_AUDIO_EXTS),
                visible=False)
            url_input = gr.Textbox(
                label="Ссылки на YouTube — по одной в строке",
                info="Сколько непустых строк — столько видео в очереди",
                lines=4,
                value="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            input_files_group = gr.CheckboxGroup(
                choices=list_input_files(),
                label="Выбрать из загруженных (можно несколько)",
                info="Файлы из /kaggle/input (подключённые датасеты) и ранее скачанные в /kaggle/working",
                visible=False)
            refresh_btn = gr.Button("🔄 Обновить список файлов", visible=False)

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

            process_btn = gr.Button("🚀 Обработать очередь", variant="primary")

        with gr.Column():
            results_table = gr.Dataframe(
                headers=["№", "Название", "Статус", "Файл"],
                datatype=["str", "str", "str", "str"],
                interactive=False, wrap=True,
                label="Результаты обработки")
            status_out = gr.Markdown("Очередь пуста — выберите файлы или вставьте ссылки.")

            with gr.Tabs():
                with gr.Tab("📜 Транскрипция"):
                    result_dd = gr.Dropdown(
                        choices=[], label="📂 Готовые транскрипции — выберите для просмотра")
                    transcript_out = gr.Textbox(label="Текст", lines=18,
                                                interactive=False, show_copy_button=True)
                with gr.Tab("🤖 Промпт для большой модели"):
                    prompt_out = gr.Textbox(label="Скопируйте целиком и вставьте в веб-версию ИИ",
                                            lines=18, interactive=False, show_copy_button=True)

            files_out = gr.File(
                label="Скачать по отдельности (.txt — транскрипции и промпты)",
                file_count="multiple")
            zip_btn = gr.Button("📦 Скачать архивом", variant="secondary")
            zip_out = gr.File(label="Архив всех транскрипций (.zip)")

    def toggle_inputs(choice):
        show_file = choice == "Файлы"
        show_url = choice == "YouTube"
        show_dd = choice == "Из загрузок (Input)"
        return (gr.update(visible=show_file), gr.update(visible=show_url),
                gr.update(visible=show_dd), gr.update(visible=show_dd))
    input_type.change(toggle_inputs, inputs=[input_type],
                      outputs=[file_input, url_input, input_files_group, refresh_btn])

    def refresh_files():
        return gr.update(choices=list_input_files())
    refresh_btn.click(refresh_files, outputs=[input_files_group])

    result_dd.change(load_result, inputs=[result_dd],
                     outputs=[transcript_out, prompt_out])

    process_btn.click(
        fn=process_batch,
        inputs=[input_type, file_input, url_input, input_files_group, lang_selector,
                cb_timestamps, cb_paragraphs, pause_slider,
                cb_prompt, prompt_instruction],
        outputs=[results_table, status_out, result_dd,
                 transcript_out, prompt_out, files_out, zip_out],
    )

    def on_zip_click():
        zip_path = build_zip()
        if not zip_path:
            raise gr.Error("Нет сохранённых транскрипций — сначала обработайте файлы.")
        return zip_path
    zip_btn.click(on_zip_click, outputs=[zip_out])

app.launch(share=True, debug=True)

if os.path.exists("vtt.py"):
    os.remove("vtt.py")
