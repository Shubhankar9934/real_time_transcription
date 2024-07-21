import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from datetime import datetime, timedelta, timezone
from transformers import MarianMTModel, MarianTokenizer
from queue import Queue
from time import sleep

def setup_microphone(mic_name):
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
        return None, None
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                return sr.Microphone(sample_rate=16000, device_index=index), name
    return None, None

def load_models(model_size, non_english):
    if model_size not in ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']:
        raise ValueError(f"Invalid model size '{model_size}', please choose one of: tiny, base, small, medium, large, large-v2, large-v3")
    audio_model = WhisperModel(model_size)

    model_name = 'Helsinki-NLP/opus-mt-es-en'
    translation_model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    return audio_model, translation_model, tokenizer

def setup_recorder(recorder, energy_threshold):
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

def record_callback(_, audio: sr.AudioData, data_queue: Queue) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

def translate_text(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**tokens)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

def transcribe_audio(data_queue, audio_model, translation_model, tokenizer, phrase_timeout):
    transcription = ['']
    phrase_time = None
    while True:
        try:
            now = datetime.now(timezone.utc)
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = data_queue.get()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                segments, _ = audio_model.transcribe(audio_np)
                segment_texts = [segment.text.strip() for segment in segments]
                text = ' '.join(segment_texts)

                translated_text = translate_text(text, translation_model, tokenizer)
                print(f"Translated text: {translated_text}")

                if phrase_complete:
                    transcription.append(translated_text)
                else:
                    transcription[-1] += ' ' + translated_text

                print("\033[H\033[J", end="")
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)
