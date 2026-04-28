import torch

if torch.xpu.is_available():
    print(f"Başarılı! Cihaz: {torch.xpu.get_device_name(0)}")
else:
    print("XPU bulunamadı. Sürücüleri veya kurulumu kontrol edin.")