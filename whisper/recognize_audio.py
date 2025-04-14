#!/usr/bin/env python3
import sys
import os
import whisper

MODE = "large"
DEVICE = "cpu"

def main():
    if len(sys.argv) < 2:
        print("Использование: python script.py путь_к_аудиофайлу")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    model = whisper.load_model(MODE, device=DEVICE)
    
    result = model.transcribe(audio_path, language="ru")
    
    transcription = result.get("text", "")
    
    output_file = os.path.splitext(audio_path)[0] + "_" + MODE + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"Транскрипция успешно сохранена в файле: {output_file}")

if __name__ == "__main__":
    main()
