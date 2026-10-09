#!pip install -q -U yt-dlp
#!pip install -q transformers accelerate gradio
#!pip install -q pyannote.audio
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

# ⬇ НОВОЕ: pyannote.audio — определение спикеров (диаризация).
# Если после установки ноутбук заругается на версии библиотек —
# Restart Session и запустить ячейку снова.
_ret = os.system("pip install -q pyannote.audio")
if _ret == 0:
    print("✅ pyannote.audio установлен — режим «Спикеры» доступен")
else:
    print("⚠️ pyannote.audio не установился: разметка спикеров будет недоступна, остальное работает.")

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

# ⬇ НОВОЕ: pyannote может не импортироваться (если установка не удалась) —
# тогда отключаем режим спикеров, но не роняем весь ноутбук.
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    DIARIZATION_AVAILABLE = True
except Exception:
    DIARIZATION_AVAILABLE = False
    print("⚠️ pyannote.audio не импортируется — разметка спикеров недоступна.")

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
    if os.path.isfile(os.path.join(DATASET_DIR, "config.json")):
        print(f"✅ Модель берём из подключённого Input: {DATASET_DIR} (без скачивания)")
        return DATASET_DIR

    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        print(f"✅ Модель уже в локальной папке: {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR

    username = _get_kaggle_username()
    if not username:
        print("KAGGLE_API не задан (или не прикреплён) — модель качается с Hugging Face, как раньше.")
        return None

    dataset_id = f"{username}/{DATASET_SLUG}"

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
    elif "already exist" in (out3 or "").lower():
        print("Датасет-кеш уже существует (видимо, создан только что) — используем локальную копию.")
    else:
        print(f"⚠️ Не удалось создать датасет-кеш: {out3[:300]}")

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

Если в тексте есть таймкоды вида [MM:SS] — проставляй их у важных мыслей, чтобы можно было перейти к моменту в видео.
Если транскрипция размечена по спикерам («Спикер 1», «Спикер 2», …) — в идеях и выводах указывай, кто именно говорил."""

# ================================================================
#  3.5. НОВОЕ: Диаризация спикеров (pyannote.audio)
# ================================================================
# Модели pyannote «gated»: чтобы они скачались, нужно ОДИН раз
# залогиниться на Hugging Face (под аккаунтом вашего HF_TOKEN)
# и нажать «Agree» на ОБЕИХ страницах:
#   * https://huggingface.co/pyannote/speaker-diarization-3.1
#   * https://huggingface.co/pyannote/segmentation-3.0
# Пайплайн грузится лениво — только если включили галочку «Спикеры».

_diar_pipeline = None


def get_diarization_pipeline():
    global _diar_pipeline
    if _diar_pipeline is not None:
        return _diar_pipeline
    if not DIARIZATION_AVAILABLE:
        raise gr.Error("pyannote.audio не установлен (см. секцию 0). Перезапустите ноутбук.")
    if not hf_token:
        raise gr.Error("Для определения спикеров нужен HF_TOKEN в Kaggle Secrets "
                       "(Add-ons → Secrets → HF_TOKEN).")
    print("Загрузка пайплайна диаризации pyannote/speaker-diarization-3.1...")
    try:
        pipe = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    except Exception as e:
        raise gr.Error("Не удалось скачать модель pyannote. Проверьте, что на HF-аккаунте "
                       "приняты условия ОБЕИХ моделей: pyannote/speaker-diarization-3.1 и "
                       f"pyannote/segmentation-3.0. Подробности: {e}")
    if device == "cuda":
        pipe.to(torch.device("cuda"))
    _diar_pipeline = pipe
    print("Пайплайн диаризации загружен!")
    return _diar_pipeline


def parse_num_speakers(choice):
    """«Авто» или пусто -> None (pyannote определит сам), иначе число."""
    try:
        n = int(str(choice))
        return n if n >= 1 else None
    except (TypeError, ValueError):
        return None


def run_diarization(audio_path, num_speakers=None):
    """Диаризация всего файла. Возвращает список (start, end, метка_спикера)."""
    pipe = get_diarization_pipeline()
    # pyannote надёжнее всего читает wav; mp4/m4a/… сначала конвертируем
    tmp_wav = None
    if os.path.splitext(audio_path)[1].lower() != ".wav":
        tmp_wav = os.path.join(WORK_DIR, "_diar_tmp.wav")
        r = subprocess.run(["ffmpeg", "-y", "-i", audio_path,
                            "-ar", "16000", "-ac", "1", tmp_wav], capture_output=True)
        if r.returncode != 0 or not os.path.exists(tmp_wav):
            raise gr.Error("ffmpeg не смог подготовить аудио для диаризации.")
        audio_for_diar = tmp_wav
    else:
        audio_for_diar = audio_path
    try:
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        diar = pipe(audio_for_diar, **kwargs)
        return [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
    finally:
        if tmp_wav:
            try: os.remove(tmp_wav)
            except OSError: pass


def assign_speakers(segments, turns):
    """
    Приклеиваем метки спикеров к сегментам Whisper: сегмент получает метку того,
    кто говорил дольше всего внутри его таймкода.
    Возвращает (segments, число_спикеров). Если спикер один — метки не ставим.
    """
    if not turns:
        return segments, 1

    # SPEAKER_00 -> «Спикер 1» (по порядку первого появления в записи)
    order, norm = {}, []
    for s, e, spk in turns:
        if spk not in order:
            order[spk] = f"Спикер {len(order) + 1}"
        norm.append((s, e, order[spk]))
    norm.sort(key=lambda t: t[0])

    if len(order) < 2:
        return segments, len(order)

    def speaker_for(start, end):
        best, best_ov = None, 0.0
        for ts, te, spk in norm:
            if ts >= end:
                break  # реплики отсортированы по началу — дальше только позже
            ov = min(end, te) - max(start, ts)
            if ov > best_ov:
                best, best_ov = spk, ov
        return best

    for seg in segments:
        ts = seg.get("timestamp") or (None, None)
        s = ts[0] if ts[0] is not None else 0.0
        e = ts[1] if ts[1] is not None else s + 5.0
        seg["speaker"] = speaker_for(s, e)
    return segments, len(order)

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
    for f in glob.glob(f"{WORK_DIR}/yt_audio.*"):
        try: os.remove(f)
        except OSError: pass

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": f"{WORK_DIR}/yt_audio.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "http_chunk_size": 10_000_000,
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
    """Вызов Whisper с языком или без."""
    kwargs = {"return_timestamps": True}
    if lang:
        kwargs["generate_kwargs"] = {"language": lang}
    return asr_pipeline(audio_path, **kwargs)


def transcribe_with_progress(audio_path, lang, progress, piece_sec=600):
    """
    Длинное аудио транскрибируем кусками по piece_sec секунд,
    короткое (< 20 мин) — одним вызовом.
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
    Новый абзац — если пауза длиннее `pause` секунд, СМЕНИЛСЯ СПИКЕР,
    либо абзац дорос до max_chars и закончился на конец предложения.
    Возвращает список (время_начала, спикер|None, текст).
    """
    paragraphs = []
    cur_text, cur_start, prev_end, cur_spk = "", None, None, None
    for seg in segments:
        ts = seg.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if start is None:
            start = prev_end if prev_end is not None else 0.0
        spk = seg.get("speaker")  # НОВОЕ
        if cur_text:
            gap = start - (prev_end if prev_end is not None else start)
            ends_sentence = cur_text.rstrip()[-1:] in ".!?…"
            speaker_changed = (spk is not None and cur_spk is not None and spk != cur_spk)
            if gap > pause or (len(cur_text) > max_chars and ends_sentence) or speaker_changed:
                paragraphs.append((cur_start, cur_spk, cur_text))
                cur_text, cur_start, cur_spk = "", None, None
        if not cur_text:
            cur_start = start
            cur_spk = spk
        cur_text = f"{cur_text} {text}".strip()
        prev_end = end if end is not None else start
    if cur_text:
        paragraphs.append((cur_start, cur_spk, cur_text))
    return paragraphs


def format_transcript(paragraphs, with_timestamps, with_paragraphs):
    """Абзацы -> итоговый текст: [MM:SS] Спикер N: текст"""
    parts = []
    for start, spk, text in paragraphs:
        prefix = f"[{sec_to_time(start)}] " if with_timestamps else ""
        if spk:
            prefix += f"{spk}: "
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
    Ищем аудио/видео файлы в /kaggle/input (рекурсивно)
    и в WORK_DIR. Возвращает список пар (метка, полный путь).
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
            if fn.startswith("_asr_piece_") or fn.startswith("_diar_tmp"):
                continue
            if os.path.splitext(fn)[1].lower() in INPUT_AUDIO_EXTS:
                p = os.path.join(WORK_DIR, fn)
                found.append(("💾 " + fn + " (working)", p))

    return found

# ================================================================
#  5. Основной пайплайн
# ================================================================
class _SubProgress:
    """Прогресс одного файла внутри пакета: маппит 0..1 в свой диапазон общего прогресса."""
    def __init__(self, parent, lo, hi):
        self._p, self._lo, self._hi = parent, lo, hi

    def __call__(self, frac, desc=None):
        try:
            self._p(self._lo + (self._hi - self._lo) * float(frac), desc=desc)
        except Exception:
            pass  # проблемы прогресса не должны ронять обработку


def safe_filename(title):
    safe = re.sub(r"[^\w\-\s]", "", title or "").strip()[:60]
    return safe or "transcript"


def unique_path(path):
    """Не перезаписывать существующие файлы: добавляем _2, _3, ..."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def _transcribe_one(audio_path, title, lang_choice, with_timestamps, with_paragraphs,
                    pause_sec, diar_enabled, diar_n_choice, make_prompt, instruction,
                    progress):
    """
    Весь путь для одного аудио: Whisper -> (диаризация) -> абзацы -> .txt-файл.
    Возвращает (текст, промпт, путь_к_txt, число_спикеров).
    `progress` — любой callable(fraction, desc=...): gr.Progress или _SubProgress.
    """
    lang = LANG_MAP.get(lang_choice)

    try:
        segments = transcribe_with_progress(audio_path, lang, progress)
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("Не хватило видеопамяти. Уменьшите batch_size в коде (например, до 4).")

    if not segments:
        raise gr.Error("Whisper не распознал речь (возможно, в аудио только музыка или тишина).")

    # --- диаризация ---
    n_speakers = 1
    if diar_enabled:
        progress(0.75, desc="Определяем спикеров (pyannote)...")
        try:
            turns = run_diarization(audio_path, parse_num_speakers(diar_n_choice))
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Диаризация не удалась: {e}. Попробуйте без разметки спикеров.")
        segments, n_speakers = assign_speakers(segments, turns)

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
    txt_path = unique_path(os.path.join(WORK_DIR, f"{safe_filename(title)}.txt"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return transcript, prompt_text, txt_path, n_speakers


def process_media(input_type, file_path, youtube_url, input_file_choice,
                  lang_choice, with_timestamps, with_paragraphs, pause_sec,
                  diar_enabled, diar_n_choice, make_prompt, instruction,
                  progress=gr.Progress()):

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

    transcript, prompt_text, txt_path, _ = _transcribe_one(
        audio_path, title, lang_choice, with_timestamps, with_paragraphs,
        pause_sec, diar_enabled, diar_n_choice, make_prompt, instruction, progress)

    progress(1.0, desc="Готово!")
    return transcript, prompt_text, txt_path


def process_batch(batch_urls, batch_files, batch_input_selected,
                  lang_choice, with_timestamps, with_paragraphs, pause_sec,
                  diar_enabled, diar_n_choice, prompts_in_zip,
                  progress=gr.Progress()):
    """
    НОВОЕ: пакетная обработка (генератор — лог обновляется после каждого файла).
    Источники: ссылки YouTube + загруженные файлы + файлы из Input/working.
    Каждый текст — отдельный .txt; в конце собираем ZIP.
    """
    sources = []  # (тип, значение, метка для лога)
    for line in (batch_urls or "").splitlines():
        line = line.strip()
        if line:
            sources.append(("url", line, line))
    for p in (batch_files or []):
        p = str(p)
        if os.path.exists(p):
            sources.append(("file", p, os.path.basename(p)))
    for p in (batch_input_selected or []):
        p = str(p)
        if os.path.exists(p):
            sources.append(("file", p, os.path.basename(p)))
        else:
            sources.append(("missing", p, os.path.basename(p)))

    if not sources:
        raise gr.Error("Нечего обрабатывать: добавьте ссылки, файлы или выберите из списка.")

    log = [f"Всего источников: {len(sources)}. Поехали!"]
    yield "\n".join(log), None

    # если включена диаризация — грузим пайплайн один раз, заранее
    if diar_enabled:
        try:
            get_diarization_pipeline()
        except gr.Error as e:
            diar_enabled = False
            log.append(f"⚠️ Диаризация отключена для всего пакета: {getattr(e, 'message', None) or str(e)}")
            yield "\n".join(log), None

    txt_paths, prompt_paths, ok_count = [], [], 0
    total = len(sources)

    for i, (kind, value, label) in enumerate(sources):
        sub = _SubProgress(progress, i / total, (i + 0.95) / total)
        log.append("")
        log.append(f"▶️ [{i + 1}/{total}] {label}")
        yield "\n".join(log), None

        if kind == "missing":
            log.append(f"❌ Файл не найден (сессия перезапускалась?): {value}")
            yield "\n".join(log), None
            continue

        try:
            # 1) добываем аудио
            if kind == "url":
                audio_path, title = download_youtube(value, sub)
            else:
                audio_path, title = value, os.path.splitext(os.path.basename(value))[0]

            # 2) транскрибируем (+диаризация, если включена)
            transcript, prompt_text, txt_path, n_speakers = _transcribe_one(
                audio_path, title, lang_choice, with_timestamps, with_paragraphs,
                pause_sec, diar_enabled, diar_n_choice, prompts_in_zip, None, sub)
            txt_paths.append(txt_path)

            # 3) промпт отдельным файлом (если просили)
            if prompts_in_zip and prompt_text:
                p_path = unique_path(os.path.join(WORK_DIR, f"{safe_filename(title)}__prompt.txt"))
                with open(p_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                prompt_paths.append(p_path)

            spk = f", спикеров: {n_speakers}" if diar_enabled else ""
            log.append(f"✅ «{title}» — {len(transcript)} симв.{spk} → {os.path.basename(txt_path)}")
            ok_count += 1
        except torch.cuda.OutOfMemoryError:
            log.append("❌ Не хватило видеопамяти — файл пропущен (уменьшите batch_size в коде).")
        except gr.Error as e:
            log.append(f"❌ {getattr(e, 'message', None) or str(e)}")
        except Exception as e:
            log.append(f"❌ Непредвиденная ошибка ({type(e).__name__}): {e}")
        yield "\n".join(log), None

    # --- архив ---
    zip_path = None
    if txt_paths:
        stamp = time.strftime("%Y%m%d_%H%M")
        zip_path = os.path.join(WORK_DIR, f"transcripts_{stamp}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in txt_paths:
                zf.write(p, arcname=os.path.basename(p))
            for p in prompt_paths:
                zf.write(p, arcname=os.path.basename(p))
            zf.writestr("_batch_log.txt", "\n".join(log))
        log.append("")
        log.append(f"🗜 Готово: {ok_count} из {total}. Архив: {os.path.basename(zip_path)} — "
                   "жмите кнопку скачивания ниже.")
    else:
        log.append("")
        log.append("⚠️ Ни один файл не обработан — архив не собираем.")

    progress(1.0, desc="Пакет завершён!")
    yield "\n".join(log), zip_path

# ================================================================
#  6. Интерфейс Gradio
# ================================================================
NUM_SPEAKERS_CHOICES = ["Авто"] + [str(i) for i in range(1, 11)]

with gr.Blocks() as app:
    gr.Markdown("# 🎬 Видео/Аудио → Текст (таймкоды, абзацы, спикеры, пакет)")
    gr.Markdown("Транскрибация идёт локально через Whisper. Саммари делаете в большой модели: "
                "скопируйте готовый промпт из второй вкладки и вставьте в ChatGPT / Claude / Gemini.")

    with gr.Tabs():
        # ==================== Вкладка 1: один файл ====================
        with gr.Tab("🎬 Один файл"):
            with gr.Row():
                with gr.Column():
                    input_type = gr.Radio(
                        ["Файл", "YouTube", "Из загрузок (Input)"],
                        label="Откуда берём аудио?", value="YouTube")
                    file_input = gr.Audio(label="Загрузите аудио/видео файл", type="filepath", visible=False)
                    url_input = gr.Textbox(label="Ссылка на YouTube",
                                           value="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                    input_file_dd = gr.Dropdown(
                        choices=list_input_files(),
                        label="Выбрать из загруженных",
                        info="Файлы из /kaggle/input (подключённые датасеты) и ранее скачанные в /kaggle/working",
                        visible=False,
                    )
                    btn_refresh = gr.Button("🔄 Обновить список файлов")
                    lang_selector = gr.Dropdown(list(LANG_MAP.keys()), value="Автоопределение",
                                                label="Язык видео (явное указание повышает точность)")

                    with gr.Accordion("⚙️ Опции текста", open=True):
                        cb_timestamps = gr.Checkbox(value=False, label="⏱ Таймкоды [MM:SS] в начале абзацев")
                        cb_paragraphs = gr.Checkbox(value=True, label="¶ Разбивать на абзацы (по паузам в речи)")
                        pause_slider = gr.Slider(0.4, 3.0, value=1.2, step=0.1,
                                                 label="Пауза для нового абзаца, сек",
                                                 info="Насколько длинную тишину считать границей абзаца")

                    with gr.Accordion("👥 Спикеры (кто говорит)", open=False):
                        cb_diar = gr.Checkbox(
                            value=False,
                            label="Размечать реплики: «Спикер 1: …», «Спикер 2: …»",
                            info="Нужен HF_TOKEN и одноразовое согласие с условиями моделей pyannote (см. комментарий в коде, секция 3.5)")
                        diar_n = gr.Dropdown(NUM_SPEAKERS_CHOICES, value="Авто",
                                             label="Сколько спикеров в видео",
                                             info="«Авто» — определить автоматически; если знаете точное число — укажите, будет точнее")

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

        # ==================== Вкладка 2: пакет ====================
        with gr.Tab("📦 Пакетная обработка"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Укажите любые комбинации источников — обработаются по очереди. "
                                "Каждая транскрипция сохранится отдельным .txt, в конце — общий ZIP.")
                    batch_urls = gr.Textbox(
                        label="🔗 Ссылки на YouTube (по одной в строке)", lines=4,
                        placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...")
                    batch_files = gr.File(label="💾 Или загрузите несколько файлов",
                                          file_count="multiple")
                    batch_input = gr.CheckboxGroup(
                        choices=list_input_files(),
                        label="📘 Или выберите из Input (датасеты) и /kaggle/working")
                    btn_refresh_batch = gr.Button("🔄 Обновить списки файлов")
                    batch_lang = gr.Dropdown(list(LANG_MAP.keys()), value="Автоопределение",
                                             label="Язык видео")

                    with gr.Accordion("⚙️ Опции текста", open=True):
                        cb_timestamps_b = gr.Checkbox(value=False, label="⏱ Таймкоды [MM:SS] в начале абзацев")
                        cb_paragraphs_b = gr.Checkbox(value=True, label="¶ Разбивать на абзацы")
                        pause_slider_b = gr.Slider(0.4, 3.0, value=1.2, step=0.1,
                                                   label="Пауза для нового абзаца, сек")

                    with gr.Accordion("👥 Спикеры", open=False):
                        cb_diar_b = gr.Checkbox(value=False, label="Размечать реплики спикеров")
                        diar_n_b = gr.Dropdown(NUM_SPEAKERS_CHOICES, value="Авто",
                                               label="Сколько спикеров в видео")

                    cb_prompt_zip = gr.Checkbox(
                        value=False,
                        label="🤖 Добавить в архив готовые промпты (по файлу на видео)")

                    run_batch_btn = gr.Button("🚀 Запустить пакет", variant="primary")

                with gr.Column():
                    batch_log = gr.Textbox(label="Лог обработки", lines=20, interactive=False)
                    batch_zip = gr.File(label="📥 Скачать всё архивом (.zip)")

    def toggle_inputs(choice):
        show_file = choice == "Файл"
        show_url = choice == "YouTube"
        show_dd = choice == "Из загрузок (Input)"
        return (gr.update(visible=show_file), gr.update(visible=show_url),
                gr.update(visible=show_dd))
    input_type.change(toggle_inputs, inputs=[input_type],
                      outputs=[file_input, url_input, input_file_dd])

    def refresh_lists():
        choices = list_input_files()
        return gr.update(choices=choices), gr.update(choices=choices)
    btn_refresh.click(refresh_lists, outputs=[input_file_dd, batch_input])
    btn_refresh_batch.click(refresh_lists, outputs=[input_file_dd, batch_input])

    process_btn.click(
        fn=process_media,
        inputs=[input_type, file_input, url_input, input_file_dd, lang_selector,
                cb_timestamps, cb_paragraphs, pause_slider,
                cb_diar, diar_n,
                cb_prompt, prompt_instruction],
        outputs=[transcript_out, prompt_out, file_out],
    )

    run_batch_btn.click(
        fn=process_batch,
        inputs=[batch_urls, batch_files, batch_input, batch_lang,
                cb_timestamps_b, cb_paragraphs_b, pause_slider_b,
                cb_diar_b, diar_n_b, cb_prompt_zip],
        outputs=[batch_log, batch_zip],
    )

app.launch(share=True, debug=True)

if os.path.exists("vtt.py"):
    os.remove("vtt.py")
