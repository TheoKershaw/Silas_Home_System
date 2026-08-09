import torch
from bot import GPT, CharTokenizer, load_model, BLOCK_SIZE
import os
import socket
import threading
import psutil

class Silas:
    def __init__(self):
        model_path = "model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found. Train it first by running chatbot.py "
                "in this same directory (it will train on data.txt and save model.pt)."
            )
        self.model, self.tokenizer = load_model(model_path)

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

                if "server status" in message or "how are you feeling" in message or 'how you feeling' in message:
                    response = silas.server_status()
                else:
                    response = silas.ask_model(message)
                conn.sendall(response.encode())
        except Exception as e:
            print(f"Error from {addr}: {e}")

    while True:
        conn, addr = s.accept()
        print(f"Connection from: {addr}")

        threading.Thread(target=connection, args=(conn, addr), daemon=True).start()