"""Minimal RFC 6455 WebSocket server pieces (stdlib only): handshake, text frames, ping/pong, close."""
from __future__ import annotations

import base64
import hashlib
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_PAYLOAD = 4 * 1024 * 1024


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode()).digest()).decode()


def _read_exact(rfile, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = rfile.read(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def build_frame(opcode: int, payload: bytes, mask: bytes | None = None) -> bytes:
    head = bytes([0x80 | opcode])
    n = len(payload)
    mbit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mbit | n])
    elif n < 65536:
        head += bytes([mbit | 126]) + struct.pack(">H", n)
    else:
        head += bytes([mbit | 127]) + struct.pack(">Q", n)
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return head + mask + payload
    return head + payload


def read_message(rfile, send_raw) -> str | bytes | None:
    """Next complete message (str for text), or None when the peer closes.

    ``send_raw(opcode, payload)`` answers pings and closes.
    """
    message, msg_opcode = b"", None
    while True:
        b1, b2 = _read_exact(rfile, 2)
        fin, opcode = bool(b1 & 0x80), b1 & 0x0F
        masked, length = bool(b2 & 0x80), b2 & 0x7F
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
        if opcode == 0x8:
            send_raw(0x8, payload[:2])
            return None
        if opcode == 0x9:
            send_raw(0xA, payload)
            continue
        if opcode == 0xA:
            continue
        if opcode in (0x1, 0x2):
            msg_opcode, message = opcode, payload
        elif opcode == 0x0:
            message += payload
        if fin:
            return message.decode("utf-8") if msg_opcode == 0x1 else message


def handshake(request, first_bytes: bytes = b"") -> dict[str, str] | None:
    """Read the HTTP upgrade request from a socket and reply 101. Returns the headers, or None."""
    data = first_bytes
    while b"\r\n\r\n" not in data:
        chunk = request.recv(4096)
        if not chunk:
            return None
        data += chunk
    lines = data.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    key = headers.get("sec-websocket-key")
    if not key:
        request.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        return None
    request.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                     f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n").encode())
    return headers
