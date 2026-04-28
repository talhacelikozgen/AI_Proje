@echo off
title Dragon AI - Tailscale Funnel
echo [1/2] Mevcut Tailscale ayarlari sifirlaniyor...
tailscale serve reset
echo [2/2] Funnel hatti kuruluyor: https://talhacell.taila77dbf.ts.net
echo Lutfen bekleyin...
tailscale funnel --https=443 http://127.0.0.1:8000
pause