import socket
import pyaudio

def main():
    server_address = ('localhost', 8000)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    try:
        while True:
            data = stream.read(1024)
            sock.sendall(data)
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing connection")
        sock.close()
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
