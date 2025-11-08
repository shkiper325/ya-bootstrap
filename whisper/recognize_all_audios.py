#!/usr/bin/env python3
"""
Скрипт для автоматической транскрибации аудиофайлов с помощью Whisper.
Рекурсивно обходит указанную папку и создает .txt файлы с транскрипцией.
"""

import os
import argparse
import whisper


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Транскрибация аудиофайлов (.m4a, .opus) в текст'
    )
    parser.add_argument(
        'directory',
        help='Путь к папке с аудиофайлами'
    )
    parser.add_argument(
        '--device',
        required=True,
        choices=['cpu', 'gpu'],
        help='Устройство для вычислений (cpu или gpu)'
    )
    parser.add_argument(
        '--model',
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Модель Whisper (по умолчанию: large)'
    )
    return parser.parse_args()


def should_process(filename):
    """Проверка, нужно ли обрабатывать файл."""
    return filename.endswith('.m4a') or filename.endswith('.opus')


def transcribe_directory(directory, model, device):
    """
    Рекурсивная транскрибация всех аудиофайлов в директории.

    Args:
        directory: Путь к директории
        model: Загруженная модель Whisper
        device: Устройство (cpu/gpu)
    """
    # fp16 работает только на GPU
    use_fp16 = (device == 'gpu')

    for root, _, files in os.walk(directory):
        for filename in files:
            if not should_process(filename):
                continue

            audio_path = os.path.join(root, filename)
            txt_path = os.path.splitext(audio_path)[0] + '.txt'

            # Пропускаем уже обработанные файлы
            if os.path.exists(txt_path):
                print(f"✓ Пропускаем (уже есть): {audio_path}")
                continue

            # Транскрибируем
            print(f"⏳ Транскрибируем: {audio_path}")
            result = model.transcribe(audio_path, language='ru', fp16=use_fp16)

            # Сохраняем результат
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])

            print(f"✓ Сохранено: {txt_path}\n")


def main():
    """Основная функция скрипта."""
    args = parse_args()

    # Проверка существования директории
    if not os.path.isdir(args.directory):
        print(f"Ошибка: {args.directory} не является директорией")
        return 1

    # Загрузка модели
    print(f"Загрузка модели Whisper ({args.model}) на {args.device.upper()}...")
    model = whisper.load_model(args.model, device=args.device)

    # Транскрибация
    transcribe_directory(args.directory, model, args.device)

    print("✓ Готово!")
    return 0


if __name__ == '__main__':
    exit(main())
