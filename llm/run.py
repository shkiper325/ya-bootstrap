#!/usr/bin/env python3
import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Генерация текста с помощью LLM"
    )
    parser.add_argument(
        "--prompt_file", "-p",
        required=True,
        help="Путь к файлу с входным промптом (обязательный параметр)"
    )
    parser.add_argument(
        "--output_file", "-o",
        required=True,
        help="Путь к файлу для записи результата (обязательный параметр)"
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Устройство выполнения (по умолчанию cpu)"
    )
    parser.add_argument(
        "--model_name", "-m",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Имя модели в HuggingFace Hub"
    )
    parser.add_argument(
        "--max_new_tokens", "-n",
        type=int,
        default=8192,
        help="Максимальное число сгенерированных токенов"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Чтение промпта
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл промпта не найден: {args.prompt_file}", file=sys.stderr)
        sys.exit(1)

    # Подготовка устройства и загрузка модели
    dtype = torch.float16
    model_kwargs = {"torch_dtype": dtype}
    if args.device == "cuda":
        # Accelerate автоматически распределит модель по доступным GPU
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    # Создаём pipeline без явного указания `device`
    # — Accelerate уже разложил модель по GPU/CPU
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Генерация
    result = generator(
        input_text,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=1
    )
    generated_text = result[0]['generated_text']

    # Запись в файл
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Генерация завершена, результат записан в {args.output_file}")

if __name__ == "__main__":
    main()
