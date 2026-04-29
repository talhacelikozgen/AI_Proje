@echo off
title Dragon AI & Cloud - Tailscale Funnel
echo [1/3] Mevcut Tailscale ayarlari sifirlaniyor...
tailscale serve reset

echo [2/3] Dragon AI hatti kuruluyor: https://talhacell.taila77dbf.ts.net/
:: Bu komut ana dizini 8000 portuna (Dragon AI) bağlar
tailscale funnel --https=443 --bg 8000

echo [3/3] Nextcloud hatti kuruluyor: https://talhacell.taila77dbf.ts.net/cloud
:: Bu komut /cloud yolunu 8081 portuna (Nextcloud) bağlar
tailscale funnel --https=443 --set-path /cloud --bg 8081

echo.
echo ========================================================
echo   ISLEM TAMAM! 
echo.
echo   Dragon AI: https://talhacell.taila77dbf.ts.net/
echo   Nextcloud: https://talhacell.taila77dbf.ts.net/cloud
echo ========================================================
echo.
echo Durumu kontrol etmek icin bu pencereyi acik tutabilirsin.
tailscale funnel status
pause