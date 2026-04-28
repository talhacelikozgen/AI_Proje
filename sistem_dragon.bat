@echo off
title DRAGON AI SYSTEM CONTROL
color 0B
echo ==========================================
echo       DRAGON AI SISTEM BASLATILIYOR
echo ==========================================

echo [STEP 1] Tailscale Funnel Tuneli Aciliyor...
start "Dragon_Tunnel" cmd /c "tailscale_tunnel.bat"

timeout /t 5

echo [STEP 2] Intel XPU Backend Yukleniyor...
:: Burada senin ana baslatma dosyanin adini kullaniyoruz
start "Dragon_Backend" cmd /c "dragon_baslat.bat"

echo ==========================================
echo Tum sistemler aktif! 
echo Dashboard: dragonhelmet.com
echo API: https://talhacell.taila77dbf.ts.net
echo ==========================================
pause