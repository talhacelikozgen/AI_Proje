import torch
import os
import datetime
import json
import gc  # Bellek temizliği için şart
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
from urllib.parse import quote

# --- 1. SİSTEM VE GPU YAPILANDIRMASI ---
os.environ["SYCL_DEVICE_FILTER"] = "gpu"
os.environ["UR_L0_DEBUG"] = "0"
os.environ["SYCL_CACHE_PERSISTENT"] = "1"

# Klasör Yolları
BASE_DIR = r"C:\AI_Proje"
OUTPUT_DIR = r"E:\Dragon_AI_Depo\Outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_user_dir(user: str):
    safe_user = os.path.basename(user.strip() or "default").replace(" ", "_")
    user_dir = os.path.join(OUTPUT_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir, safe_user

# --- 2. MODEL YÜKLEME VE OPTİMİZASYON ---
print("Dragon AI v3: SDXL Modeli Hazırlanıyor...")
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
txt2img_pipe = None
img2img_pipe = None


def prepare_pipe(pipe):
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("xpu")
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    return pipe

try:
    if torch.xpu.is_available():
        print(f"UR Katmanı Doğrulandı: {torch.xpu.get_device_name(0)} aktif.")
        
        txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        txt2img_pipe = prepare_pipe(txt2img_pipe)

        img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        img2img_pipe = prepare_pipe(img2img_pipe)
        
        print("Büyük Tasarım Modelleri (SDXL) XPU üzerine başarıyla yerleşti!")
    else:
        print("HATA: GPU bulundu ama XPU katmanı aktif değil.")
except Exception as e:
    print(f"Model Yükleme Hatası: {e}")

# --- 3. API SUNUCUSU ---
app = FastAPI(title="Dragon AI v3 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/outputs/{user}/{file_path:path}")
async def serve_output(user: str, file_path: str):
    user_dir, _ = get_user_dir(user)
    safe_path = os.path.abspath(os.path.join(user_dir, file_path))
    if not safe_path.startswith(os.path.abspath(user_dir)):
        raise HTTPException(status_code=403, detail="Erişim engellendi.")
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")
    return FileResponse(
        safe_path,
        headers={"Cache-Control": "public, max-age=31536000"}
    )

@app.get("/")
async def read_index():

    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html bulunamadı!"}

# --- 4. ÜRETİM VE TEMİZLİK MANTIĞI ---
class GenRequest(BaseModel):
    prompt: str
    user: str


class DeleteRequest(BaseModel):
    filename: str
    user: str


def save_output(image, prompt_text, user: str):
    user_dir, safe_user = get_user_dir(user)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"dragon-{timestamp}.png"
    save_path = os.path.join(user_dir, filename)
    image.save(save_path)

    prompt_json_path = os.path.join(user_dir, "prompt.json")
    prompts_dict = {}
    if os.path.exists(prompt_json_path):
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)

    prompts_dict[filename] = prompt_text
    with open(prompt_json_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, ensure_ascii=False, indent=2)

    return filename, safe_user


@app.post("/generate")
async def generate(request: GenRequest):
    if txt2img_pipe is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi.")
    
    try:
        torch.xpu.empty_cache()
        gc.collect()

        design_prompt = f"{request.prompt}, game concept art, digital illustration, clean lines, high quality, stylized art, artstation trending"
        negative_prompt = "photorealistic, realistic, photography, blurry, messy, distorted, grainy, low quality"

        with torch.no_grad():
            output = txt2img_pipe(
                prompt=design_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=8.0,
                width=1024,
                height=1024,
            )
            image = output.images[0]

        filename, safe_user = save_output(image, request.prompt, request.user)

        del output
        del image
        gc.collect()
        torch.xpu.empty_cache()

        return {"image_url": f"/outputs/{quote(safe_user)}/{filename}"}
    except Exception as e:
        torch.xpu.empty_cache()
        print(f"Üretim Hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-from-image")
async def generate_from_image(
    prompt: str = Form(...),
    user: str = Form(...),
    strength: float = Form(0.5),
    image: UploadFile = File(...),
):
    if img2img_pipe is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi.")

    if image.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="Geçersiz dosya türü. Lütfen bir görsel gönderin.")

    try:
        torch.xpu.empty_cache()
        gc.collect()

        image_bytes = await image.read()
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_image = init_image.resize((1024, 1024), resample=Image.LANCZOS)

        design_prompt = f"{prompt}, game concept art, digital illustration, clean lines, high quality, stylized art, artstation trending"
        negative_prompt = "photorealistic, realistic, photography, blurry, messy, distorted, grainy, low quality"

        with torch.no_grad():
            output = img2img_pipe(
                prompt=design_prompt,
                image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
            )
            image_out = output.images[0]

        filename, safe_user = save_output(image_out, prompt, user)

        del output
        del image_out
        del init_image
        gc.collect()
        torch.xpu.empty_cache()

        return {"image_url": f"/outputs/{quote(safe_user)}/{filename}"}
    except Exception as e:
        torch.xpu.empty_cache()
        print(f"Görsel Dönüşüm Hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete_image(request: DeleteRequest):
    user_dir, _ = get_user_dir(request.user)
    image_path = os.path.join(user_dir, request.filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Görsel bulunamadı.")

    try:
        os.remove(image_path)
        prompt_json_path = os.path.join(user_dir, "prompt.json")
        if os.path.exists(prompt_json_path):
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                prompts_dict = json.load(f)
            prompts_dict.pop(request.filename, None)
            with open(prompt_json_path, 'w', encoding='utf-8') as f:
                json.dump(prompts_dict, f, ensure_ascii=False, indent=2)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user}")
async def history(user: str, limit: int = 5, offset: int = 0):
    user_dir, safe_user = get_user_dir(user)
    prompt_json_path = os.path.join(user_dir, "prompt.json")
    prompts_dict = {}
    if os.path.exists(prompt_json_path):
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)

    images = [f for f in os.listdir(user_dir) if f.lower().endswith('.png')]
    images.sort(reverse=True)

    total = len(images)
    paged_images = images[offset:offset + limit]

    history = []
    for filename in paged_images:
        history.append({
            "filename": filename,
            "prompt": prompts_dict.get(filename, ""),
            "image_url": f"/outputs/{quote(safe_user)}/{filename}"
        })

    return {"total": total, "items": history}


if __name__ == "__main__":
    # Uvicorn için çalışan en stabil ayarlar
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")