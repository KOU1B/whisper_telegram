#!/bin/bash

# Этот скрипт устанавливает системные зависимости, необходимые для проекта.

echo "Обновление списка пакетов..."
sudo apt-get update

echo "Установка ffmpeg, python3-pip и python3-venv..."
sudo apt-get install -y ffmpeg python3-pip python3-venv

echo "Установка завершена."
echo "Не забудьте создать виртуальное окружение: python3 -m venv venv"
echo "И активировать его: source venv/bin/activate"
echo "А затем установить Python-зависимости: pip install -r requirements.txt"
