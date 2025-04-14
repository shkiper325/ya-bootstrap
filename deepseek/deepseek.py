import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys

if len(sys.argv) < 3:
    print('Слишком мало аргументов')
    quit(1)

# Определение устройства: используем GPU, если доступно, иначе CPU.
DEVICE = "cpu"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MAX_NEW_TOKENS = 8192

# Чтение файлов
with open(sys.argv[1], "r", encoding="utf-8") as f:
    input_text = f.read()

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Загрузка модели с вычислениями в fp16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # вычисления в fp16
    device_map=DEVICE
)

# Создание пайплайна для генерации текста
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Генерация текста (настройте параметры при необходимости)
result = generator(input_text, max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=1)

# Извлечение сгенерированного текста
generated_text = result[0]['generated_text']

# Запись результата в файл result.txt
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write(generated_text)
