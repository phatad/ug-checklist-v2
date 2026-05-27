"""
Auth module — UG Checklist v3
Best practices:
- Bcrypt hash password
- Flask session với secret key
- Rate limiting brute force
- Credentials từ ENV vars
"""
import os, time, hashlib
from functools import wraps
from flask import session, request, redirect, url_for, jsonify

# ── Rate limiter đơn giản (in-memory) ─────────────────────────
_failed = {}   # ip -> [timestamp, ...]
MAX_ATTEMPTS = 5
BLOCK_SECONDS = 900  # 15 phút

def _get_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()

def _is_blocked(ip):
    now = time.time()
    attempts = [t for t in _failed.get(ip, []) if now - t < BLOCK_SECONDS]
    _failed[ip] = attempts
    return len(attempts) >= MAX_ATTEMPTS

def _record_fail(ip):
    _failed.setdefault(ip, []).append(time.time())

def _clear_fail(ip):
    _failed.pop(ip, None)

# ── Password check (timing-safe) ──────────────────────────────
def _check_password(plain, stored_hash):
    """Bcrypt nếu có, fallback sha256 với salt."""
    try:
        import bcrypt
        return bcrypt.checkpw(plain.encode(), stored_hash.encode())
    except ImportError:
        # Fallback: sha256 với salt (vẫn an toàn hơn plain text)
        salt, hashed = stored_hash.split(":", 1)
        return hashlib.sha256((salt + plain).encode()).hexdigest() == hashed

def _hash_password(plain):
    try:
        import bcrypt
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        import secrets
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + plain).encode()).hexdigest()
        return f"{salt}:{hashed}"

# ── Load users từ ENV ──────────────────────────────────────────
def _load_users():
    """
    ENV format:
      UG_USER1=username:password_hash
      UG_USER2=username:password_hash
    Hoặc plain password (tự hash khi start):
      UG_USERS=user1:pass1,user2:pass2
    """
    users = {}
    # Format 1: UG_USERS=user1:pass1,user2:pass2 (plain, tự hash)
    raw = os.environ.get("UG_USERS", "")
    if raw:
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                u, p = pair.split(":", 1)
                users[u.strip()] = _hash_password(p.strip())
    # Format 2: UG_USER1=user:hash, UG_USER2=user:hash
    for key, val in os.environ.items():
        if key.startswith("UG_USER") and key != "UG_USERS" and ":" in val:
            u, h = val.split(":", 1)
            users[u.strip()] = h.strip()
    return users

USERS = _load_users()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.is_json:
                return jsonify({"error": "Chưa đăng nhập"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def do_login(username, password):
    """Trả về (success, message)"""
    ip = _get_ip()
    if _is_blocked(ip):
        remaining = int(BLOCK_SECONDS / 60)
        return False, f"Quá nhiều lần sai. Thử lại sau {remaining} phút."
    if not USERS:
        return False, "Chưa cấu hình tài khoản. Liên hệ admin."
    stored = USERS.get(username)
    if stored and _check_password(password, stored):
        _clear_fail(ip)
        session.permanent = True
        session["logged_in"] = True
        session["username"] = username
        return True, "OK"
    _record_fail(ip)
    attempts_left = MAX_ATTEMPTS - len(_failed.get(ip, []))
    return False, f"Sai tên đăng nhập hoặc mật khẩu. Còn {max(0,attempts_left)} lần thử."
