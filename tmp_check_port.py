import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', 8501))
    print('open')
except Exception as e:
    print('closed', repr(e))
finally:
    s.close()