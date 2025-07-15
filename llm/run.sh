#!/usr/bin/env bash

set -euo pipefail

# Путь к prompt-файлу
PROMPT_FILE="prompt.txt"
# Временный файл
TEMP_FILE="temp.txt"

# Проверяем, что prompt.txt существует
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Ошибка: файл $PROMPT_FILE не найден."
  exit 1
fi

# Проходим по всем .txt файлам в текущей директории
for file in *.txt; do
  # Пропускаем prompt.txt
  if [[ "$file" == "$PROMPT_FILE" ]]; then
    continue
  fi

  # Формируем имя выходного файла: <basename>_ready.txt
  base="${file%.txt}"
  output_file="${base}_ready.txt"

  echo "Обрабатываем '$file' → '$output_file'..."

  # Собираем temp.txt
  cat "$PROMPT_FILE" > "$TEMP_FILE"
  cat "$file" >> "$TEMP_FILE"

  # Запускаем модель
  python run.py \
    -p "$TEMP_FILE" \
    --device cuda \
    -m yandex/YandexGPT-5-Lite-8B-instruct \
    -n 8196 \
    -o "$output_file"

  echo "Готово: $output_file"
done
