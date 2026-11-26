import base64
import hashlib
import re
from functools import lru_cache

import bcrypt

from db import EmailTaken, create_user, find_user

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


class AuthError(Exception):
    """A sign-in or registration problem whose message is fit to show the user."""


def _prepare(password):
    # bcrypt only reads 72 bytes (and bcrypt 5 rejects longer input), so hash first to keep it all.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password):
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password, password_hash):
    return bcrypt.checkpw(_prepare(password), password_hash.encode("ascii"))


@lru_cache(maxsize=1)
def _decoy_hash():
    return hash_password("decoy password for unknown accounts")


def normalise_email(email):
    return email.strip().lower()


def register(full_name, email, password, confirm_password):
    full_name = " ".join(full_name.split())
    email = normalise_email(email)
    if not full_name:
        raise AuthError("Enter your name.")
    if len(full_name) > 120:
        raise AuthError("Keep the name under 120 characters.")
    if len(email) > 254 or not EMAIL_PATTERN.match(email):
        raise AuthError("Enter a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Use at least {MIN_PASSWORD_LENGTH} characters for the password.")
    if password != confirm_password:
        raise AuthError("The two passwords do not match.")
    try:
        return create_user(email, full_name, hash_password(password))
    except EmailTaken:
        raise AuthError("An account with this email already exists. Sign in instead.") from None


def authenticate(email, password):
    user = find_user(normalise_email(email))
    if user is None:
        # Spend the same bcrypt time as a real check so response time doesn't reveal unknown emails.
        verify_password(password, _decoy_hash())
        raise AuthError("Incorrect email or password.")
    if not verify_password(password, user["password_hash"]):
        raise AuthError("Incorrect email or password.")
    return {"id": user["id"], "email": user["email"], "full_name": user["full_name"]}
