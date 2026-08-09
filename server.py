import torch
from bot import GPT, CharTokenizer, load_model, BLOCK_SIZE
import os
import socket
import threading
import psutil
import time
import json
import re
from datetime import datetime, timedelta

class Silas:
    def __init__(self):
        model_path = "model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found. Train it first by running chatbot.py "
                "in this same directory (it will train on data.txt and save model.pt)."
            )
        self.model, self.tokenizer = load_model(model_path)

        self.reminders = []

        if os.path.exists("reminders.json"):
            with open("reminders.json", "r", encoding="utf-8") as f:
                self.reminders = json.load(f)

    def load_memory(self):
        history = []
        if os.path.exists("memory.txt"):
            with open("memory.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("user:"):
                        history.append({"role": "user", "content": line[len("user:"):].strip()})
                    elif line.startswith("silas:"):
                        history.append({"role": "assistant", "content": line[len("silas:"):].strip()})
        return history

    def mem(self, text):
        with open("memory.txt", "a") as f:
            f.write(text + "\n")

    def ask_model(self, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
        self.mem(f"user: {prompt}")
        formatted_prompt = f"User: {prompt}\nBot:"

        idx = torch.tensor([self.tokenizer.encode(formatted_prompt)], dtype=torch.long)

        with torch.no_grad():
            out = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

        generated = self.tokenizer.decode(out[0].tolist())

        reply = generated[len(formatted_prompt):]
        reply = reply.split("User:")[0].strip()

        if not reply:
            reply = "Sorry, I don't have a good response for that yet."

        self.mem(f"silas: {reply}")
        return reply

    def server_status(self):
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        parts = [
            f"CPU is at {cpu} percent",
            f"Memory is at {ram.percent} percent",
            f"Disk is at {disk.percent} percent used",
        ]

        return ". ".join(parts)

    def save_reminders(self):
        with open("reminders.json", "w", encoding="utf-8") as f:
            json.dump(self.reminders, f, indent=4)


    def delete_reminders(self):
        self.reminders = [
            reminder
            for reminder in self.reminders
            if not reminder["completed"]
        ]

        self.save_reminders()


    def get_due_reminders(self):
        now = datetime.now()

        due = []

        for reminder in self.reminders:
            if reminder["completed"]:
                continue

            reminder_time = datetime.fromisoformat(
                reminder["time"]
            )

            if now >= reminder_time:
                due.append(reminder)

        return due

    def parse_reminder(self, text):
        text = text.lower().strip()

        text = " ".join(text.split())

        pattern = (
            r"(?:set\s+(?:a\s+)?reminder|"
            r"remind\s+me)"
            r"\s+(?:for\s+)?"
            r"(today|tomorrow)"
            r"\s+at\s+"
            r"(\d{1,2})"
            r"(?:\s*:\s*(\d{2}))?"
            r"\s*(am|pm)?"
            r"(?:\s+(?:that\s+)?(?:i\s+)?(?:have\s+to\s+|need\s+to\s+|should\s+)?to?\s*)?"
            r"(.+)$"
        )

        match = re.search(pattern, text)

        if not match:
            return None

        day = match.group(1)
        hour = int(match.group(2))
        minute = int(match.group(3) or 0)
        am_pm = match.group(4)
        reminder_text = match.group(5).strip()

        if am_pm:
            if am_pm == "pm" and hour != 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0

        if hour > 23 or minute > 59:
            return None

        now = datetime.now()

        if day == "tomorrow":
            reminder_date = now.date() + timedelta(days=1)
        else:
            reminder_date = now.date()

        reminder_time = datetime(
            reminder_date.year,
            reminder_date.month,
            reminder_date.day,
            hour,
            minute
        )

        if not reminder_text:
            return None

        return {
            "time": reminder_time.isoformat(),
            "message": reminder_text,
            "completed": False
        }

    def add_reminder(self, reminder):
        self.reminders.append(reminder)
        self.save_reminders()

    def handle_reminder(self, text):
        reminder = self.parse_reminder(text)

        if reminder is None:
            return (
                "I couldn't understand the reminder. "
                "Try saying something like "
                "set a reminder for tomorrow at 4pm to call John."
            )

        reminder_time = datetime.fromisoformat(
            reminder["time"]
        )

        if reminder_time <= datetime.now():
            return "That reminder time has already passed."

        self.add_reminder(reminder)

        spoken_time = reminder_time.strftime(
            "%A at %I:%M %p"
        )

        return (
            f"Okay, I'll remind you {spoken_time} "
            f"to {reminder['message']}."
        )

if __name__ == "__main__":
    silas = Silas()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 1337))
    s.listen()

    def connection(conn, addr):
        try:
            while True:
                data = conn.recv(1024)

                if not data:
                    break

                message = data.decode().strip()

                if not message:
                    continue

                if "server status" in message.lower() or "how are you feeling" in message.lower() or 'how you feeling' in message.lower():
                    response = silas.server_status()
                if "get due reminder" in message.lower():
                    reminder = silas.get_due_reminders()

                    if reminder:
                        response = f"Reminder: {reminder[0]['message']}"
                    else:
                        response = ""
                elif "remind me" in message.lower() or "set reminder" in message.lower():
                    response = silas.handle_reminder(message)
                else:
                    response = silas.ask_model(message)
                conn.sendall(response.encode())
        except Exception as e:
            print(f"Error from {addr}: {e}")

    while True:
        conn, addr = s.accept()
        print(f"Connection from: {addr}")

        threading.Thread(target=connection, args=(conn, addr), daemon=True).start()
