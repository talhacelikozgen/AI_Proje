import torch
import os
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline
from pydantic import BaseModel
import uvicorn

# --- 1. SİSTEM VE GPU YAPILANDIRMASI ---
os.environ["SYCL_DEVICE_FILTER"] = "gpu"
os.environ["UR_L0_DEBUG"] = "0"
os.environ["SYCL_CACHE_PERSISTENT"] = "1"

# Klasör Yolları
BASE_DIR = r"C:\AI_Proje"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
USER_DIR = os.path.join(OUTPUT_DIR, "Talha Celik")

if not os.path.exists(USER_DIR):
    os.makedirs(USER_DIR)

# --- 2. MODEL YÜKLEME (DOĞRU SIRALAMA) ---
print("Dragon AI: Intel XPU Hazırlanıyor...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = None

try:
    if torch.xpu.is_available():
        print(f"UR Katmanı Doğrulandı: {torch.xpu.get_device_name(0)} aktif.")
        # Modeli önce RAM'e al, sonra XPU'ya taşı
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("xpu")
        print("Model Intel XPU (GPU) üzerine başarıyla yerleşti!")
    else:
        print("HATA: GPU bulundu ama Unified Runtime (UR) bağlantısı kurulamadı.")
except Exception as e:
    print(f"Model Yükleme Hatası: {e}")

# --- 3. API SUNUCUSU (FASTAPI) ---
app = FastAPI(title="Dragon AI v2 Backend")

# Web arayüzünün (GitHub/Domain) bağlanabilmesi için izinler
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm dış kaynaklardan gelen isteklere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. STATİK DOSYA VE ANA SAYFA YÖNETİMİ ---

# Görselleri /outputs yoluyla dışarı aç
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
async def read_index():
    """Tarayıcıya doğrudan index.html dosyasını gönderir."""
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": f"index.html dosyası {BASE_DIR} adresinde bulunamadı!"}

# --- 5. API UÇ NOKTALARI ---

class GenRequest(BaseModel):
    prompt: str
    user: str

@app.post("/generate")
async def generate(request: GenRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi veya GPU hatası mevcut.")
    
    try:
        # Görsel üretimi (Intel GPU üzerinde)
        image = pipe(request.prompt).images[0]
        
        # İsimlendirme: dragon-HHMMSS.png
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"dragon-{timestamp}.png"
        save_path = os.path.join(USER_DIR, filename)
        
        image.save(save_path)
        
        # main.py içindeki return satırı şöyle olmalı:
        return {"image_url": f"https://talhacell.taila77dbf.ts.net/outputs/Talha%20Celik/{filename}"}
    except Exception as e:
        print(f"Üretim Hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{username}")
async def get_history(username: str):
    """Kullanıcının geçmiş görsellerini listeler."""
    if os.path.exists(USER_DIR):
        # Sadece resim dosyalarını al ve tarihe göre (isme göre) sırala
        files = [f for f in os.listdir(USER_DIR) if f.lower().endswith(".png")]
        return sorted(files, reverse=True)
    return []

# --- 6. SUNUCUYU BAŞLAT ---
if __name__ == "__main__":
    # Host 0.0.0.0: Hem yerel ağdan hem Tailscale'den erişim sağlar
    uvicorn.run(app, host="0.0.0.0", port=8000)