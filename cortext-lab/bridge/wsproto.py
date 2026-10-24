"""Minimal RFC 6455 WebSocket helpers (stdlib only, server side).

Just enough for CorText Lab: text frames, ping/pong, close, fragmentation.
"""
import base64
import hashlib
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_PAYLOAD = 4 * 1024 * 1024


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode()).digest()).decode()


def _read_exact(rfile, n):
    buf = b""
    while len(buf) < n:
        chunk = rfile.read(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def read_message(rfile, wfile_lock_send):
    """Return the next complete text/binary message as str/bytes, or None on close.

    `wfile_lock_send(opcode, payload)` is used to answer pings and closes.
    """
    message = b""
    msg_opcode = None
    while True:
        b1, b2 = _read_exact(rfile, 2)
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", _read_exact(rfile, 2))
        elif length == 127:
            (length,) = struct.unpack(">Q", _read_exact(rfile, 8))
        if length > MAX_PAYLOAD:
            raise ConnectionError("payload too large")
        mask = _read_exact(rfile, 4) if masked else None
        payload = _read_exact(rfile, length) if length else b""
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            wfile_lock_send(0x8, payload[:2])
            return None
        if opcode == 0x9:  # ping
            wfile_lock_send(0xA, payload)
            continue
        if opcode == 0xA:  # pong
            continue
        if opcode in (0x1, 0x2):
            msg_opcode = opcode
            message = payload
        elif opcode == 0x0:
            message += payload
        if fin:
            return message.decode("utf-8") if msg_opcode == 0x1 else message


def build_frame(opcode: int, payload: bytes) -> bytes:
    head = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack(">H", n)
    else:
        head += bytes([127]) + struct.pack(">Q", n)
    return head + payload
