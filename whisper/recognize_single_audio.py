#!/usr/bin/env python3
import sys
import os
import argparse
import whisper

def parse_args():
    parser = argparse.ArgumentParser(
        description="Транскрипция аудиофайла с помощью Whisper"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Путь к аудиофайлу"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Путь для сохранения результата транскрипции"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Устройство для выполнения модели (по умолчанию: cpu)"
    )
    parser.add_argument(
        "--model",
        default="large",
        help="Имя модели Whisper (по умолчанию: large)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Проверяем, что input_file существует
    if not os.path.isfile(args.input_file):
        print(f"Ошибка: файл \"{args.input_file}\" не найден.", file=sys.stderr)
        sys.exit(1)

    # Загружаем модель
    print(f"Загружаем модель \"{args.model}\" на устройстве \"{args.device}\"...")
    model = whisper.load_model(args.model, device=args.device)

    # Транскрибируем
    print(f"Транскрибируем файл \"{args.input_file}\"...")
    result = model.transcribe(args.input_file, language="ru")
    transcription = result.get("text", "")

    # Сохраняем результат
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"Транскрипция успешно сохранена в файле: {args.output_file}")

if __name__ == "__main__":
    main()
