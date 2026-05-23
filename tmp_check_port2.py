import socket
with open('tmp_port_result2.txt', 'w', encoding='utf-8') as f:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(('127.0.0.1', 8501))
        f.write('open')
    except Exception as e:
        f.write('closed:' + repr(e))
    finally:
        s.close()