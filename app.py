import os
import re
import subprocess
import tempfile

from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import socket

app = Flask(__name__)

recognizer = sr.Recognizer()

addr, port = "127.0.0.1", 1337

def execute(text):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((addr, port))
        s.sendall(text.encode())

        data = s.recv(4096).decode()

    print(data)
    return data

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def recognize_audio(webm_path):
    wav_path = webm_path + ".wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", webm_path,
            "-ar", "16000",
            "-ac", "1",
            wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(
                audio,
                language="en-GB"
            )
        except sr.UnknownValueError:
            return ""

        return clean_text(text)

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():

    audio_file = request.files.get("audio")

    if audio_file is None:
        return jsonify({"error": "No audio received"}), 400

    webm_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False
        ) as tmp:

            audio_file.save(tmp.name)
            webm_path = tmp.name

        text = recognize_audio(webm_path)

        print("Recognised:", text)

        wake_word = (
            "silas" in text
            or "sylas" in text
            or "silus" in text
            or "solace" in text
            or "cyrus" in text
            or "sinus" in text
        )

        repsonse = execute(text)

        return jsonify({
            "text": text,
            "wake_word": wake_word,
            "response": repsonse
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if webm_path and os.path.exists(webm_path):
            os.remove(webm_path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )