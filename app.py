import os
import re
import subprocess
import tempfile
import time
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import socket
import threading

app = Flask(__name__)
recognizer = sr.Recognizer()
addr, port = "127.0.0.1", 1337
socket_lock = threading.Lock()

def execute(text, timeout=15):
    if "hello" in text:
        return "Sir"
    else:
        try:
            with socket_lock:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    s.connect((addr, port))
                    s.sendall(text.encode())
                    data = s.recv(4096).decode()
            print("Bot replied:", data)
            return data
        except socket.timeout:
            print("execute() timed out waiting for bot.py")
            return "Sorry, Silas timed out."
        except (ConnectionRefusedError, OSError) as e:
            print("execute() connection error:", e)
            return "Sorry, I couldn't reach Silas right now."

def reminder_checker():
    while True:
        try:
            with socket_lock:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    s.connect((addr, port))
                    s.sendall(b"get due reminder")
                    data = s.recv(4096).decode()

            if data:
                print("Reminder:", data)

        except Exception as e:
            print("Reminder checker error:", e)

        time.sleep(2)

if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    threading.Thread(target=reminder_checker, daemon=True).start()

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
        except sr.RequestError as e:
            print("Google recognizer request error:", e)
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

        if wake_word:
            response = execute(text)
            return jsonify({
                "text": text,
                "wake_word": wake_word,
                "response": response
            })
        else:
            return jsonify({
                "text": text,
                "wake_word": wake_word
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
