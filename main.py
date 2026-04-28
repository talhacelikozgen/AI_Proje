import base64
import torch
import os
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from pydantic import BaseModel
import uvicorn

os.environ["HF_TOKEN"] = "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"

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

# --- 2. BÜYÜK MODEL YÜKLEME (SDXL - TASARIM ODAKLI) ---
print("Dragon AI v3: SDXL Modeli Hazırlanıyor...")
# Tasarım ve konsept sanatı için en temiz temel model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = None

try:
    # Intel GPU (XPU) kontrolü
    if torch.xpu.is_available():
        print(f"UR Katmanı Doğrulandı: {torch.xpu.get_device_name(0)} aktif.")
        
        # SDXL Pipeline yükleme (12GB VRAM için FP16 ve Safetensors kullanımı)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        )
        
        # Tasarım odaklı (çizim gibi) sonuçlar için Scheduler ayarı
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # Modeli GPU'ya (XPU) taşı
        pipe = pipe.to("xpu")
        
        # 12GB VRAM'in tıkanmaması için bellek yönetimi
        pipe.enable_attention_slicing()
        
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

# --- 4. STATİK DOSYALAR ---
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
async def read_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html bulunamadı!"}

@app.get("/payload")
async def read_payload():
    payload_path = os.path.join(BASE_DIR, "index_payload.html")
    if os.path.exists(payload_path):
        with open(payload_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return {"payload": encoded}
    raise HTTPException(status_code=404, detail="Payload bulunamadı!")

# --- 5. API UÇ NOKTALARI ---
class GenRequest(BaseModel):
    prompt: str
    user: str

@app.post("/generate")
async def generate(request: GenRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi.")
    
    try:
        # TASARIM ODAKLI PROMPT (Otomatik oyun konsepti stili)
        design_prompt = f"{request.prompt}, game concept art, digital illustration, clean lines, high quality, stylized art, artstation trending"
        negative_prompt = "photorealistic, realistic, photography, blurry, messy, distorted, grainy, low quality"

        # Görsel üretimi (SDXL 1024x1024)
        image = pipe(
            prompt=design_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30, # Adım sayısını kalite için 30 yaptık
            guidance_scale=8.0,
            width=1024,
            height=1024
        ).images[0]
        
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"dragon-{timestamp}.png"
        save_path = os.path.join(USER_DIR, filename)
        
        image.save(save_path)
        
        return {"image_url": f"https://talhacell.taila77dbf.ts.net/outputs/Talha%20Celik/{filename}"}
    except Exception as e:
        print(f"Üretim Hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{username}")
async def get_history(username: str):
    if os.path.exists(USER_DIR):
        files = [f for f in os.listdir(USER_DIR) if f.lower().endswith(".png")]
        return sorted(files, reverse=True)
    return []

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)