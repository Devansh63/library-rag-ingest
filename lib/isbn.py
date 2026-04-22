"""
ISBN normalization and conversion helpers.

Why this exists: the three source datasets format ISBNs inconsistently.
Zenodo's Best Books Ever dump sometimes uses ISBN-10 with hyphens, the CMU
dataset often has no ISBN at all, and the UCSD graph uses ISBN-13 strings
that sometimes include leading zeros lost during JSON parsing. We also
need to join rows across sources on a single canonical key, which means
converting every ISBN-10 we see to ISBN-13 so joins actually match.

The functions here intentionally return None on invalid input rather than
raising. The ingestion pipeline logs and drops bad ISBNs; it does not crash
on them, because a single malformed row in a 50k-row dataset should not
kill a batch load.
"""

from __future__ import annotations


def _strip(value: str | None) -> str:
    """Remove everything that is not a digit or a trailing 'X' check char."""
    if value is None:
        return ""
    # ISBN-10 can end in 'X' as a check digit representing 10.
    return "".join(c for c in value.upper() if c.isdigit() or c == "X")


def is_valid_isbn10(raw: str | None) -> bool:
    """Return True if `raw` parses to a valid ISBN-10 after stripping."""
    digits = _strip(raw)
    if len(digits) != 10:
        return False

    # ISBN-10 check: sum of (digit * position) mod 11 == 0, where position
    # runs 10 down to 1. The last digit may be 'X' meaning 10.
    total = 0
    for position, char in enumerate(digits):
        weight = 10 - position
        if char == "X":
            # 'X' is only valid as the final check digit.
            if position != 9:
                return False
            value = 10
        else:
            value = int(char)
        total += value * weight
    return total % 11 == 0


def is_valid_isbn13(raw: str | None) -> bool:
    """Return True if `raw` parses to a valid ISBN-13 after stripping."""
    digits = _strip(raw)
    if len(digits) != 13 or "X" in digits:
        return False

    # ISBN-13 check: alternating weights of 1 and 3, sum mod 10 == 0.
    total = 0
    for position, char in enumerate(digits):
        weight = 1 if position % 2 == 0 else 3
        total += int(char) * weight
    return total % 10 == 0


def isbn10_to_isbn13(raw: str | None) -> str | None:
    """Convert a valid ISBN-10 to its canonical ISBN-13 form.

    Returns None if the input is not a valid ISBN-10, so callers can treat
    a None result as "unusable, skip this row's ISBN field".
    """
    if not is_valid_isbn10(raw):
        return None

    digits = _strip(raw)
    # ISBN-13 = '978' + first 9 digits of ISBN-10 + new check digit.
    prefix = "978" + digits[:9]

    total = 0
    for position, char in enumerate(prefix):
        weight = 1 if position % 2 == 0 else 3
        total += int(char) * weight
    check_digit = (10 - (total % 10)) % 10
    return prefix + str(check_digit)


def normalize_isbn13(raw: str | None) -> str | None:
    """Return a clean 13-digit ISBN-13 string, or None if the input is junk.

    Accepts either an ISBN-10 or an ISBN-13 in any format (hyphens, spaces,
    leading/trailing whitespace). This is the function ingestion code should
    call on every incoming ISBN-like field.
    """
    digits = _strip(raw)

    if len(digits) == 13 and is_valid_isbn13(digits):
        return digits
    if len(digits) == 10 and is_valid_isbn10(digits):
        return isbn10_to_isbn13(digits)
    return None


def normalize_isbn10(raw: str | None) -> str | None:
    """Return a clean 10-character ISBN-10 string, or None if unusable.

    Used to preserve the original ISBN-10 in `books.isbn10` when we have one.
    We do NOT back-convert ISBN-13 to ISBN-10 here, because many ISBN-13s
    (anything in the 979- prefix range) have no valid ISBN-10 equivalent.
    """
    digits = _strip(raw)
    if len(digits) == 10 and is_valid_isbn10(digits):
        return digits
    return None
