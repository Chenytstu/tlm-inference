from socket import *
import dill
import struct
import time
import sys

from FixedPoint.FixedPoint import FXfamily, FXnum
import Configs.communication as config

def pack(data, port=config.protocol):
    if port == 0 and type(data) is FXnum:
        byte_data = b"FX"
        byte_data += data.family.fraction_bits.to_bytes(1, "big")
        byte_data += struct.pack('d', float(data))
        return byte_data
    return dill.dumps(data)

def unpack(data, port=config.protocol):
    if port == 0 and data[:2] == b"FX":
        family = FXfamily(int().from_bytes(data[2:3], "big"))
        return family(struct.unpack('d', data[3:])[0])
    return dill.loads(data)

def send(data, addr=config.default_host, port=config.default_port, LOG=config.LOG):
    s = socket(AF_INET, SOCK_STREAM)
    for i in range(config.MAX_Connection_Time):
        try:
            s.connect((addr, port))
        except ConnectionRefusedError as e:
            time.sleep(config.connection_interval_in_s)
            s.close()
            if i == config.MAX_Connection_Time - 1:
                raise Exception(f'[ERROR] Connection failed after {config.MAX_Connection_Time} times.')
            s = socket(AF_INET, SOCK_STREAM)
            if LOG:
                print(f"[WARNING] Re-try to connect {addr}:{port}")
            continue
        break
    if LOG:
        print(f"[INFO] Connection Established with {addr}:{port}")
    start = time.time()
    data = pack(data)
    s.sendall(data)
    if LOG:
        print(f"[INFO] Data Sent to {addr}:{port}, size: {sys.getsizeof(data)}, time: {time.time() - start}")
    s.close()
    return sys.getsizeof(data)

def recv(addr=config.default_host, port=config.default_port, LOG=config.LOG):
    s = socket(AF_INET, SOCK_STREAM)
    try:
        s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        s.bind((addr, port))
        s.listen(10)
    except ConnectionRefusedError as e:
        print(e)
        sys.exit(1)
    if LOG:
        print(f"[INFO] Connection Established with {addr}:{port}")
    data = b''
    comm, _ = s.accept()
    while True:
        packet = comm.recv(config.buffer)
        if packet is None or len(packet) == 0:
            break
        data += packet
    if LOG:
        print(f"[INFO] Data received from {addr}:{port}, size: {sys.getsizeof(data)}")
    return unpack(data)

if __name__ == "__main__":
    if sys.argv[1] == "1":
        famcfrac = FXfamily(64)
        import numpy as np
        num = np.array([famcfrac(-123.123415), famcfrac(123.123123415)])
        send(num)
    else:
        num = recv()
        print(num[0])
        