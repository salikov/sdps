import warnings
import logging
import os
import gc
import re
import torch
import gradio as gr
import urllib.request
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from transformers import AutoModelForImageSegmentation, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torchvision import transforms
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
import tempfile
import glob

# 1. Чистая консоль
warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

custom_css = """
.gallery-container .grid-wrap { justify-content: flex-start !important; gap: 10px !important; }
.gallery-container .thumbnail-item { width: 120px !important; height: 120px !important; object-fit: contain !important; }
"""

# 2. Безопасное получение HF_TOKEN
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    print("⚠️ HF_TOKEN не найден. Скачивание без авторизации.")
else:
    if hf_token:
        print("✅ HF_TOKEN найден.")

# 3. Загрузка BiRefNet
print("⏳ Загрузка BiRefNet...")
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True, token=hf_token)
birefnet.to("cuda")
birefnet.eval()

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Загрузка SDXL
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
print(f"⏳ Загрузка SDXL...")

pipe_t2i = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16", token=hf_token)
pipe_t2i.enable_model_cpu_offload()

# 5. Загрузка LLM
print("⏳ Загрузка Qwen2.5-3B-Instruct (4-bit)...")
llm_id = "Qwen/Qwen2.5-3B-Instruct"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, token=hf_token)
llm_model = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=bnb_config, device_map="auto", token=hf_token)
print("✅ Все модели загружены!")

CUTOUT_DIR = "/kaggle/working/cutouts"
os.makedirs(CUTOUT_DIR, exist_ok=True)

# --- ФУНКЦИИ ВКЛАДКИ 1 (ТОВАРНЫЕ ФОТО) ---
def batch_cutout(input_gallery, threshold, blur_radius, bg_color, progress=gr.Progress()):
    if not input_gallery: return [], "Сначала загрузите фото!"
    for f in glob.glob(os.path.join(CUTOUT_DIR, "*")): os.remove(f)
    total = len(input_gallery)
    progress(0.1, desc=f"Начало пакетной обработки ({total} шт)...")
    gallery_items = []
    for i, file_data in enumerate(input_gallery):
        progress(0.1 + (i / total) * 0.8, desc=f"Вырезание {i+1} из {total}...")
        file_path = file_data['path'] if isinstance(file_data, dict) else file_data[0]
        pil_img = Image.open(file_path).convert("RGB")
        w, h = pil_img.size
        if w > h: left = (w - h) // 2; pil_img = pil_img.crop((left, 0, left + h, h))
        else: top = (h - w) // 2; pil_img = pil_img.crop((0, top, w, top + w))
        pil_img = pil_img.resize((1024, 1024), Image.LANCZOS)
        
        input_tensor = transform_image(pil_img).unsqueeze(0).to("cuda").to(birefnet.dtype)
        with torch.no_grad(): scaled_preds = birefnet(input_tensor)[-1].sigmoid().squeeze().cpu().float()
        raw_mask = transforms.ToPILImage()(scaled_preds)
        mask_array = np.array(raw_mask); mask_array[mask_array < threshold] = 0
        mask = Image.fromarray(mask_array.astype(np.uint8))
        if blur_radius > 0: mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        cutout_rgba = pil_img.convert("RGBA"); cutout_rgba.putalpha(mask)
        cutout_rgba.save(os.path.join(CUTOUT_DIR, f"cutout_{i}.png"))
        
        if bg_color == "Прозрачный (PNG)": preview_bg = cutout_rgba
        else:
            colors = {"Белый": (255, 255, 255), "Черный": (0, 0, 0), "Зеленый (Хромакей)": (0, 255, 0)}
            c = colors.get(bg_color, (255, 255, 255)); preview_bg = Image.new("RGB", pil_img.size, c)
            preview_bg.paste(pil_img, (0, 0), mask)
        gallery_items.append((preview_bg, f"Предмет {i+1}"))
        del input_tensor, scaled_preds, raw_mask, mask_array, mask
        torch.cuda.empty_cache(); gc.collect()
    return gallery_items, f"✅ Успешно вырезано {total} предметов!"

def batch_compose(prompt, negative_prompt, steps, cfg, custom_bg_files, bg_only, progress=gr.Progress()):
    def update_ui_progress(pipe, step, timestep, callback_kwargs):
        progress(0.2 + (step / int(steps)) * 0.6, desc=f"Генерация (Шаг {step}/{int(steps)})")
        return callback_kwargs

    if bg_only:
        progress(0.2, desc="Генерация одного фона...")
        with torch.inference_mode():
            bg_img = pipe_t2i(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=int(steps), guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(42), callback_on_step_end=update_ui_progress, callback_on_step_end_tensor_inputs=['latents']).images[0]
        return [(bg_img, "Сгенерированный фон")], "Успешно! Сгенерирован 1 фон."

    cutouts = glob.glob(os.path.join(CUTOUT_DIR, "*.png"))
    if not cutouts: return [], "Сначала вырежьте предметы на Шаге 1!"
    use_custom = True if custom_bg_files else False
    total = len(cutouts); gallery_items = []
    for i, cutout_path in enumerate(cutouts):
        progress(0.1 + (i / total) * 0.8, desc=f"Обработка {i+1} из {total}...")
        if use_custom: bg_img = Image.open(custom_bg_files[i % len(custom_bg_files)]).convert("RGBA").resize((1024, 1024), Image.LANCZOS)
        else:
            with torch.inference_mode():
                bg_img = pipe_t2i(prompt=prompt, negative_prompt=negative_prompt + ", humans, animals, text, watermarks", num_inference_steps=int(steps), guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(42 + i), callback_on_step_end=update_ui_progress, callback_on_step_end_tensor_inputs=['latents']).images[0].convert("RGBA")
        subject = Image.open(cutout_path).convert("RGBA"); alpha = subject.split()[3]
        shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0)); shadow.paste((0, 0, 0, 150), (0, 0), alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(20)); shadow_offset = Image.new("RGBA", subject.size, (0, 0, 0, 0)); shadow_offset.paste(shadow, (0, 20))
        final_img = bg_img.copy(); final_img.alpha_composite(shadow_offset); final_img.alpha_composite(subject)
        gallery_items.append((final_img.convert("RGB"), f"Результат {i+1}"))
        del bg_img, subject, alpha, shadow, shadow_offset, final_img
        torch.cuda.empty_cache(); gc.collect()
    return gallery_items, f"Успешно! Обработано {total} фото."

def view_selected_image(evt: gr.SelectData, gallery_items):
    if gallery_items and evt.index < len(gallery_items): return gallery_items[evt.index][0]
    return None

# --- ФУНКЦИИ ВКЛАДКИ 2 (ГЕНЕРАЦИЯ) ---
def change_sampler(sampler_name):
    if sampler_name == "DPM++ 2M Karras":
        pipe_t2i.scheduler = DPMSolverMultistepScheduler.from_config(pipe_t2i.scheduler.config, use_karras_sigmas=True)
    elif sampler_name == "Euler a":
        pipe_t2i.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_t2i.scheduler.config)
    elif sampler_name == "Euler":
        pipe_t2i.scheduler = EulerDiscreteScheduler.from_config(pipe_t2i.scheduler.config)

def generate_creative(prompt, negative_prompt, steps, cfg, seed, width, height, sampler, batch_size, progress=gr.Progress()):
    def update_ui_progress(pipe, step, timestep, callback_kwargs):
        progress(0.2 + (step / int(steps)) * 0.6, desc=f"Генерация (Шаг {step}/{int(steps)})")
        return callback_kwargs

    change_sampler(sampler)
    
    actual_seed = torch.randint(0, 2**32 - 1, (1,)).item() if seed == -1 else int(seed)
    gallery_items = []
    
    try:
        for i in range(batch_size):
            progress(0.1 + (i / batch_size) * 0.8, desc=f"Генерация {i+1} из {batch_size}...")
            with torch.inference_mode():
                result = pipe_t2i(
                    prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    num_inference_steps=int(steps), 
                    guidance_scale=cfg, 
                    width=width, 
                    height=height,
                    generator=torch.Generator("cuda").manual_seed(actual_seed + i), 
                    callback_on_step_end=update_ui_progress, 
                    callback_on_step_end_tensor_inputs=['latents']
                ).images[0]
            gallery_items.append((result, f"Вариант {i+1} (Seed: {actual_seed + i})"))
            torch.cuda.empty_cache(); gc.collect()
            
        return gallery_items, f"Успешно! Сгенерировано {batch_size} изображений."
    except Exception as e:
        return [], f"Ошибка: {e}"

# --- ФУНКЦИИ ВКЛАДКИ 3 (ГЕНЕРАТОР ПРОМПТОВ) ---
def generate_sd_prompt(user_idea, progress=gr.Progress()):
    if not user_idea: return "", "", "Опишите вашу идею!"
    progress(0.2, desc="Очистка памяти и генерация промпта ИИ...")
    torch.cuda.empty_cache(); gc.collect()
    
    system_prompt = """You are an expert prompt engineer for Stable Diffusion XL. 
Convert the user's simple idea into a highly detailed, descriptive English prompt. 
Include subject, environment, lighting, camera angles, style, and quality modifiers (e.g., 8k, masterpiece, highly detailed).
Also provide a good negative prompt.
Respond STRICTLY in this format:
Positive: <prompt>
Negative: <prompt>"""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_idea}]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text, return_tensors="pt").to("cuda")
    
    try:
        with torch.inference_mode():
            outputs = llm_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
        response = llm_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        pos_prompt = ""; neg_prompt = ""
        match_pos = re.search(r'Positive:\s*(.*?)(?:Negative:|$)', response, re.DOTALL | re.IGNORECASE)
        match_neg = re.search(r'Negative:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        
        if match_pos:
            pos_prompt = re.sub(r'<[^>]+>', '', match_pos.group(1)).strip()
        if match_neg:
            neg_prompt = re.sub(r'<[^>]+>', '', match_neg.group(1)).strip()
        if not pos_prompt: pos_prompt = response 
            
        return pos_prompt, neg_prompt, "✅ Промпт сгенерирован! Нажмите кнопку ниже, чтобы вставить его."
    except Exception as e:
        return "", "", f"Ошибка при генерации промпта: {str(e)}"

# --- ИНТЕРФЕЙС GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 AI Studio: Товарные фото, Генерация и Промпты")
    
    with gr.Tabs():
        with gr.Tab("📸 Товарные фото"):
            with gr.Accordion("✂️ Шаг 1: Загрузка и вырезание предметов", open=False):
                input_gallery = gr.Gallery(label="Загрузите фото товаров", columns=8, height=200, object_fit="contain")
                with gr.Row():
                    threshold_slider = gr.Slider(minimum=0, maximum=255, step=5, value=128, label="Порог краев")
                    blur_slider = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Размытие краев")
                    bg_color_dropdown = gr.Dropdown(choices=["Белый", "Черный", "Зеленый (Хромакей)", "Прозрачный (PNG)"], value="Белый", label="Цвет предпросмотра")
                cutout_btn = gr.Button("✂️ Вырезать все предметы", variant="secondary")
                gallery_cutouts = gr.Gallery(label="Превью вырезанных предметов", columns=8, height=300, object_fit="contain", elem_classes="gallery-container")
                status_text_1 = gr.Textbox(label="Статус", interactive=False)
                viewer_1 = gr.Image(label="Просмотр полного размера", type="pil", height=512)
            
            gr.Markdown("### 🚀 Шаг 2: Генерация")
            with gr.Row():
                with gr.Column():
                    prompt_tab1 = gr.Textbox(label="Описание фона", value="A commercial product shot placed on a rustic dark wooden table, cinematic depth of field, highly detailed, realistic textures, empty background")
                    negative_prompt_tab1 = gr.Textbox(label="Negative Prompt", value="low quality, worst quality, deformed, distorted, blurry, ugly, extra objects")
                    with gr.Row():
                        steps_slider = gr.Slider(minimum=10, maximum=50, step=1, value=30, label="Шаги")
                        cfg_slider = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, value=8.0, label="CFG Scale")
                    with gr.Row():
                        custom_bgs = gr.File(label="Загрузить свои фоны", file_count="multiple", file_types=["image"])
                        bg_only_checkbox = gr.Checkbox(label="Сгенерировать только 1 фон", value=False)
                with gr.Column():
                    gr.Markdown("#### ⚡ Быстрые пресеты")
                    with gr.Row(): btn_wood = gr.Button("🪵 Дерево"); btn_marble = gr.Button("⚪ Мрамор"); btn_studio = gr.Button("💡 Студия")
                    with gr.Row(): btn_nature = gr.Button("🌿 Природа"); btn_concrete = gr.Button("🏭 Бетон"); btn_silk = gr.Button("🧵 Шелк")
                    with gr.Row(): btn_velvet = gr.Button("🍷 Бархат"); btn_sand = gr.Button("🏖️ Песок"); btn_water = gr.Button("💧 Вода")
                    with gr.Row(): btn_neon = gr.Button("🌃 Неон"); btn_pastel = gr.Button("🎈 Пастель"); btn_gold = gr.Button("👑 Золото")
                    with gr.Row(): btn_wet_stone = gr.Button("🪨 Мокрый камень"); btn_podium = gr.Button("📦 Подиум"); btn_wood_sun = gr.Button("🌅 Дерево и Солнце")
                    with gr.Row(): btn_light_sand = gr.Button("🏜️ Светлый песок"); btn_black_marble = gr.Button("⬛ Черный мрамор"); btn_liquid_silk = gr.Button("💚 Жидкий шелк")
                    with gr.Row(): btn_cyberpunk = gr.Button("🌃 Киберпанк"); btn_loft = gr.Button("🏗️ Лофт"); btn_3d_geo = gr.Button("📐 3D-геометрия")
                    with gr.Row(): btn_clouds = gr.Button("☁️ Облака")
                    
            generate_btn = gr.Button("🚀 Сгенерировать", variant="primary")
            gallery_final = gr.Gallery(label="Финальные результаты", columns=8, height=400, object_fit="contain", elem_classes="gallery-container")
            status_text_2 = gr.Textbox(label="Статус", interactive=False)
            viewer_2 = gr.Image(label="Просмотр полного размера", type="pil", height=512)

            btn_wood.click(fn=lambda: "A commercial product shot placed on a rustic dark wooden table, empty background", outputs=prompt_tab1)
            btn_marble.click(fn=lambda: "A commercial product shot placed on a white marble surface, empty background", outputs=prompt_tab1)
            btn_nature.click(fn=lambda: "A commercial product shot placed on a mossy rock in a peaceful forest, empty background", outputs=prompt_tab1)
            btn_studio.click(fn=lambda: "A commercial product shot in a clean professional studio environment, empty background", outputs=prompt_tab1)
            btn_concrete.click(fn=lambda: "A commercial product shot placed on a raw dark concrete surface, empty background", outputs=prompt_tab1)
            btn_silk.click(fn=lambda: "A commercial product shot placed on flowing silk fabric, empty background", outputs=prompt_tab1)
            btn_velvet.click(fn=lambda: "A commercial product shot placed on rich dark velvet fabric, empty background", outputs=prompt_tab1)
            btn_sand.click(fn=lambda: "A commercial product shot placed on golden beach sand, empty background", outputs=prompt_tab1)
            btn_water.click(fn=lambda: "A commercial product shot splashing with clear water drops, empty background", outputs=prompt_tab1)
            btn_neon.click(fn=lambda: "A commercial product shot illuminated by vibrant neon lights, empty background", outputs=prompt_tab1)
            btn_pastel.click(fn=lambda: "A commercial product shot on a soft pastel pink background, empty background", outputs=prompt_tab1)
            btn_gold.click(fn=lambda: "A commercial product shot placed on a reflective gold surface, empty background", outputs=prompt_tab1)
            btn_wet_stone.click(fn=lambda: "A commercial product shot placed on a wet dark flat basalt stone, water splashes and ripples on the background, blurred tropical monstera leaves in the background, warm morning sunlight piercing through leaves, sharp focus, depth of field, photorealistic, 8k", outputs=prompt_tab1)
            btn_podium.click(fn=lambda: "A commercial product shot placed on a smooth dark grey concrete podium, minimalist modern studio background, soft dramatic studio lighting, soft shadows, sharp focus, volumetric light, depth of field, bokeh, 8k resolution, professional commercial product photography", outputs=prompt_tab1)
            btn_wood_sun.click(fn=lambda: "A commercial product shot standing on a rustic light oak wooden table, soft focus botanical garden background, beautiful bokeh, gentle golden hour morning sunbeams, hyperrealistic texture, professional product photography, 8k", outputs=prompt_tab1)
            btn_light_sand.click(fn=lambda: "A commercial product shot placed on fine smooth beige sand, blurred beach dunes and soft ocean waves in the far background, bright sunny day, cinematic lighting, high-end cosmetics shot, 8k, photorealistic", outputs=prompt_tab1)
            btn_black_marble.click(fn=lambda: "A commercial product shot standing on a polished black marble surface with elegant gold veins, dark luxury background with golden light leaks, blurred abstract geometry, premium commercial shot, rich textures, volumetric lighting, 8k resolution", outputs=prompt_tab1)
            btn_liquid_silk.click(fn=lambda: "A commercial product shot placed on flowing luxury emerald silk fabric folds, soft elegant studio lighting, glowing atmosphere, high fashion background, deeply out of focus backdrop, award-winning product photography, 8k", outputs=prompt_tab1)
            btn_cyberpunk.click(fn=lambda: "A commercial product shot placed on a wet asphalt surface reflecting neon lights, futuristic cyberpunk city street background at night, deep out of focus, neon blue and magenta colors, misty air, high-end commercial sneaker photography, 8k", outputs=prompt_tab1)
            btn_loft.click(fn=lambda: "A commercial product shot standing on a textured metal plate, blurred industrial loft background, exposed brick wall, harsh dramatic side lighting, deep shadows, cinematic tech product shot, 8k resolution", outputs=prompt_tab1)
            btn_3d_geo.click(fn=lambda: "A commercial product shot placed on a matte geometric pedestal, surrounded by abstract 3D shapes, pastel pink and clay blue color palette, soft global illumination, clean studio render aesthetic, trendy commercial product photography, 8k", outputs=prompt_tab1)
            btn_clouds.click(fn=lambda: "A commercial product shot floating in mid-air surrounded by soft fluffy white and pink clouds, magical ethereal atmosphere, soft pastel gradient background, dreamlike lighting, hyper-detailed, fantasy product shot, 8k", outputs=prompt_tab1)

            cutout_btn.click(fn=batch_cutout, inputs=[input_gallery, threshold_slider, blur_slider, bg_color_dropdown], outputs=[gallery_cutouts, status_text_1])
            generate_btn.click(fn=batch_compose, inputs=[prompt_tab1, negative_prompt_tab1, steps_slider, cfg_slider, custom_bgs, bg_only_checkbox], outputs=[gallery_final, status_text_2])
            gallery_cutouts.select(fn=view_selected_image, inputs=gallery_cutouts, outputs=viewer_1)
            gallery_final.select(fn=view_selected_image, inputs=gallery_final, outputs=viewer_2)

        with gr.Tab("🎨 Генерация изображений"):
            gr.Markdown("### Генерация картинок по текстовому описанию")
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_tab2 = gr.Textbox(label="Prompt (Опишите что нарисовать)", value="A majestic lion wearing a royal crown, sitting in a mystical glowing forest, cinematic lighting, 8k, photorealistic")
                    negative_prompt_tab2 = gr.Textbox(label="Negative Prompt", value="low quality, blurry, deformed, ugly, text, watermarks")
                    
                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(choices=["DPM++ 2M Karras", "Euler a", "Euler"], value="DPM++ 2M Karras", label="Метод сэмплинга (Sampler)")
                        steps_gen = gr.Slider(minimum=10, maximum=50, step=1, value=30, label="Шаги (Steps)")
                        
                    with gr.Row():
                        width_gen = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Ширина (Width)")
                        height_gen = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Высота (Height)")
                        
                    with gr.Row():
                        cfg_gen = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, value=7.0, label="CFG Scale")
                        batch_gen = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Batch Size (Кол-во)")
                        seed_gen = gr.Number(label="Seed (-1 = случайно)", value=-1)
                        
                    generate_gen_btn = gr.Button("🎨 Сгенерировать изображение", variant="primary")
                    
                with gr.Column(scale=1):
                    gallery_gen = gr.Gallery(label="Результаты (Кликните для увеличения)", columns=2, height=500, object_fit="contain", elem_classes="gallery-container")
                    status_gen = gr.Textbox(label="Статус", interactive=False)
                    viewer_gen = gr.Image(label="Просмотр полного размера", type="pil", height=512)

            generate_gen_btn.click(fn=generate_creative, inputs=[prompt_tab2, negative_prompt_tab2, steps_gen, cfg_gen, seed_gen, width_gen, height_gen, sampler_dropdown, batch_gen], outputs=[gallery_gen, status_gen])
            gallery_gen.select(fn=view_selected_image, inputs=gallery_gen, outputs=viewer_gen)

        with gr.Tab("🧠 Промпт-Генератор"):
            gr.Markdown("### Опишите идею простыми словами (можно по-русски), а ИИ сделает идеальный промпт для SDXL")
            with gr.Row():
                with gr.Column():
                    user_idea = gr.Textbox(label="Ваша идея", value="красивая девушка киберпанк с неоновыми волосами пьет кофе в кафе", lines=3)
                    gen_prompt_btn = gr.Button("🧠 Сгенерировать промпт", variant="primary")
                    
                    llm_status = gr.Textbox(label="Статус", interactive=False)
                    
                    pos_out = gr.Textbox(label="Positive Prompt", lines=4)
                    neg_out = gr.Textbox(label="Negative Prompt", lines=2)
                    
                    gr.Markdown("#### Вставить сгенерированный промпт в:")
                    with gr.Row():
                        btn_to_tab1 = gr.Button("📸 Вставить в Товарные фото")
                        btn_to_tab2 = gr.Button("🎨 Вставить в Генерацию")

                with gr.Column():
                    pass 

            gen_prompt_btn.click(fn=generate_sd_prompt, inputs=user_idea, outputs=[pos_out, neg_out, llm_status])
            btn_to_tab1.click(fn=lambda pos, neg: (pos, neg), inputs=[pos_out, neg_out], outputs=[prompt_tab1, negative_prompt_tab1])
            btn_to_tab2.click(fn=lambda pos, neg: (pos, neg), inputs=[pos_out, neg_out], outputs=[prompt_tab2, negative_prompt_tab2])

demo.launch(share=True, debug=True)
import os
if os.path.exists("app.py"):
    os.remove("app.py")
