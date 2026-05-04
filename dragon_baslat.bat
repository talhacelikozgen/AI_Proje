@echo off
title DRAGON AI v2 - SUNUCU YONETIMI
color 0b

echo [1/3] Intel oneAPI Ortami Hazirlaniyor...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

echo.
echo [2/3] Unified Runtime (UR) Yapilandirmasi Tamamlaniyor...
:: Yeni nesil UR takip degiskenleri
set SYCL_UR_TRACE=0
set UR_L0_DEBUG=0
:: GPU odakli calisma ve onbellek ayari
set SYCL_DEVICE_FILTER=gpu
set SYCL_CACHE_PERSISTENT=1

echo.
echo [3/3] Dragon AI Backend Baslatiliyor...
echo Yerel Adres: http://127.0.0.1:8000
echo Tailscale Funnel: https://talhacell.taila77dbf.ts.net/
echo.

:: Proje dizinine git ve sunucuyu calistir
cd /d C:\AI_Proje
python main.py

pause