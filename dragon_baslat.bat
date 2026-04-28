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
echo Sunucu Adresi: http://100.106.94.121:8000
echo.

:: Proje dizinine git ve sunucuyu calistir
cd /d C:\AI_Proje
python main.py

pause