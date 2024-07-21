import socket
from queue import Queue
from threading import Thread
from Real_Time_Trans_Project.helper.helper import transcribe_audio, load_models

def handle_client_connection(client_socket, data_queue):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        data_queue.put(data)

def main():
    server_address = ('localhost', 8000)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print(f"Server listening on {server_address}")

    data_queue = Queue()
    audio_model, translation_model, tokenizer = load_models("large-v3", False)  # Adjust as needed

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    client_thread = Thread(target=handle_client_connection, args=(client_socket, data_queue))
    client_thread.start()

    transcribe_thread = Thread(target=transcribe_audio, args=(data_queue, audio_model, translation_model, tokenizer, 3))
    transcribe_thread.start()

    client_thread.join()
    transcribe_thread.join()
    server_socket.close()

if __name__ == "__main__":
    main()
