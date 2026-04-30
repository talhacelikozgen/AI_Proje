import torch
import os
import datetime
import json
import gc  # Bellek temizliği için şart
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io

# --- 1. SİSTEM VE GPU YAPILANDIRMASI ---
os.environ["SYCL_DEVICE_FILTER"] = "gpu"
os.environ["UR_L0_DEBUG"] = "0"
os.environ["SYCL_CACHE_PERSISTENT"] = "1"

# Klasör Yolları
BASE_DIR = r"C:\AI_Proje"
OUTPUT_DIR = r"E:\Dragon_AI_Depo\Outputs"
USER_DIR = os.path.join(OUTPUT_DIR, "Talha Celik")

if not os.path.exists(USER_DIR):
    os.makedirs(USER_DIR)

# --- 2. MODEL YÜKLEME VE OPTİMİZASYON ---
print("Dragon AI v3: SDXL Modeli Hazırlanıyor...")
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = None

try:
    if torch.xpu.is_available():
        print(f"UR Katmanı Doğrulandı: {torch.xpu.get_device_name(0)} aktif.")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        )
        
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("xpu")
        
        # --- VRAM İLAÇLARI ---
        pipe.enable_attention_slicing() # Belleği parçalayarak işler
        pipe.enable_model_cpu_offload() # Gerekmeyen model parçalarını RAM'e atar (VRAM korur)
        
        print("Büyük Tasarım Modeli (SDXL) XPU üzerine başarıyla yerleşti!")
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

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
async def read_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html bulunamadı!"}

# --- 4. ÜRETİM VE TEMİZLİK MANTIĞI ---
class GenRequest(BaseModel):
    prompt: str
    user: str

@app.post("/generate")
async def generate(request: GenRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi.")
    
    try:
        # Her üretimden önce ön temizlik
        torch.xpu.empty_cache()
        gc.collect()

        design_prompt = f"{request.prompt}, game concept art, digital illustration, clean lines, high quality, stylized art, artstation trending"
        negative_prompt = "photorealistic, realistic, photography, blurry, messy, distorted, grainy, low quality"

        # Görsel üretimi
        # 'with torch.no_grad()' hafıza birikmesini engeller
        with torch.no_grad():
            output = pipe(
                prompt=design_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25, # VRAM için adımı 30'dan 25'e çektik, kalite fark etmez
                guidance_scale=8.0,
                width=1024,
                height=1024
            )
            image = output.images[0]
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"dragon-{timestamp}.png"
        save_path = os.path.join(USER_DIR, filename)
        image.save(save_path)
        
        # --- KRİTİK TAHLİYE OPERASYONU ---
        # Üretilen ağır nesneleri sil ve GPU'yu boşalt
        del output
        del image
        gc.collect()
        torch.xpu.empty_cache() # VRAM'i Windows'a geri iade et
        # --------------------------------

        # JSON Kayıt İşlemleri
        prompt_json_path = os.path.join(USER_DIR, "prompt.json")
        prompts_dict = {}
        if os.path.exists(prompt_json_path):
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                prompts_dict = json.load(f)
        
        prompts_dict[filename] = request.prompt
        with open(prompt_json_path, 'w', encoding='utf-8') as f:
            json.dump(prompts_dict, f, ensure_ascii=False, indent=2)
        
        return {"image_url": f"https://talhacell.taila77dbf.ts.net/outputs/Talha%20Celik/{filename}"}

    except Exception as e:
        torch.xpu.empty_cache()
        print(f"Üretim Hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{username}")
async def get_history(username: str):
    if os.path.exists(USER_DIR):
        files = [f for f in os.listdir(USER_DIR) if f.lower().endswith(".png")]
        return sorted(files, reverse=True)
    return []

if __name__ == "__main__":
    # Uvicorn için çalışan en stabil ayarlar
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")