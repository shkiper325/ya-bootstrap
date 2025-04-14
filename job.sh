#!/bin/bash

while kill -0 1570 2>/dev/null; do
    sleep 1
done

cd whisper
message "audio1"
python recognize_audio.py 1.ogg
message "audio2"
python recognize_audio.py 2.m4a
message "audio3"
python recognize_audio.py 3.m4a

cd ~/deepseek
message "text1"
python deepseek.py "2025-03-25.txt" "2025-03-25_out.txt"
message "text2"
python deepseek.py "2025-03-04.txt" "2025-03-04_out.txt"
message "done"
