"""
ISBN normalization helpers.

The three source datasets format ISBNs inconsistently (hyphens, missing
leading zeros, ISBN-10 vs ISBN-13 mix). Everything is normalized to
ISBN-13 before touching the DB so joins actually work.

Functions return None on invalid input rather than raising - a bad ISBN
in a 50k-row batch shouldn't kill the whole load.
"""

from __future__ import annotations


def _strip(value: str | None) -> str:
    """Strip everything except digits and trailing 'X'."""
    if value is None:
        return ""
    return "".join(c for c in value.upper() if c.isdigit() or c == "X")


def is_valid_isbn10(raw: str | None) -> bool:
    digits = _strip(raw)
    if len(digits) != 10:
        return False
    total = 0
    for position, char in enumerate(digits):
        weight = 10 - position
        if char == "X":
            if position != 9:
                return False
            value = 10
        else:
            value = int(char)
        total += value * weight
    return total % 11 == 0


def is_valid_isbn13(raw: str | None) -> bool:
    digits = _strip(raw)
    if len(digits) != 13 or "X" in digits:
        return False
    total = 0
    for position, char in enumerate(digits):
        weight = 1 if position % 2 == 0 else 3
        total += int(char) * weight
    return total % 10 == 0


def isbn10_to_isbn13(raw: str | None) -> str | None:
    """Convert a valid ISBN-10 to ISBN-13. Returns None if input is invalid."""
    if not is_valid_isbn10(raw):
        return None
    digits = _strip(raw)
    prefix = "978" + digits[:9]
    total = 0
    for position, char in enumerate(prefix):
        weight = 1 if position % 2 == 0 else 3
        total += int(char) * weight
    check_digit = (10 - (total % 10)) % 10
    return prefix + str(check_digit)


def normalize_isbn13(raw: str | None) -> str | None:
    """Return a clean 13-digit ISBN-13 string, or None if the input is junk.

    Accepts ISBN-10 or ISBN-13 in any format (hyphens, spaces, etc.).
    This is the function ingest scripts should call on every ISBN field.
    """
    digits = _strip(raw)
    if len(digits) == 13 and is_valid_isbn13(digits):
        return digits
    if len(digits) == 10 and is_valid_isbn10(digits):
        return isbn10_to_isbn13(digits)
    return None


def normalize_isbn10(raw: str | None) -> str | None:
    """Return a clean 10-char ISBN-10 string, or None if unusable.

    Does not back-convert ISBN-13 to ISBN-10 - many ISBN-13s in the 979-
    prefix range have no valid ISBN-10 equivalent.
    """
    digits = _strip(raw)
    if len(digits) == 10 and is_valid_isbn10(digits):
        return digits
    return None
