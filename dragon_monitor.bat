@echo off
title DRAGON MONITOR
:start
cls
echo [KONTROL EDILIYOR...] %time%
echo.

echo --- DNS DURUMU ---
nslookup www.dragonhelmet.com | findstr "talhacell" && echo DNS: OK

echo.
echo --- PORT DURUMU ---
netstat -an | find "8000" && echo Dragon AI: OK
netstat -an | find "8081" && echo Nextcloud: OK

echo.
echo 10 saniye bekleniyor...
ping -n 11 127.0.0.1 >nul
goto start