"""Inventory management — borrow, return, availability."""
from __future__ import annotations

from datetime import date, timedelta

from app.core.config import settings
from app.core.db import execute_query, execute_write_fetch


def get_book_availability(isbn13: str) -> dict:
    sql = """
        SELECT b.isbn13, b.title, b.authors,
            COUNT(i.id) AS total_copies,
            COUNT(i.id) FILTER (WHERE i.is_active AND NOT EXISTS(
                SELECT 1 FROM borrows br WHERE br.inventory_id = i.id AND br.status = 'active'
            )) AS available_copies
        FROM books b
        LEFT JOIN inventory i ON i.isbn13 = b.isbn13
        WHERE b.isbn13 = %(isbn13)s
        GROUP BY b.isbn13, b.title, b.authors
    """
    rows = execute_query(sql, {"isbn13": isbn13})
    return dict(rows[0]) if rows else {"error": "Book not found"}


def borrow_book(isbn13: str, user_id: str) -> dict:
    available = execute_query("""
        SELECT i.id AS inventory_id FROM inventory i
        WHERE i.isbn13 = %(isbn13)s AND i.is_active = true
        AND NOT EXISTS (SELECT 1 FROM borrows br WHERE br.inventory_id = i.id AND br.status = 'active')
        LIMIT 1
    """, {"isbn13": isbn13})
    if not available:
        return {"error": "No copies available"}
    inv_id = available[0]["inventory_id"]
    due = date.today() + timedelta(days=settings.max_borrow_days)
    result = execute_write_fetch("""
        INSERT INTO borrows (inventory_id, user_id, borrow_date, due_date, status, renewed_count)
        VALUES (%(inv_id)s, %(user_id)s, CURRENT_DATE, %(due)s, 'active', 0)
        RETURNING id, borrow_date, due_date
    """, {"inv_id": inv_id, "user_id": user_id, "due": due})
    if result:
        return {"borrow_id": result[0]["id"], "isbn13": isbn13, "due_date": str(result[0]["due_date"]), "status": "active"}
    return {"error": "Failed to create borrow record"}


def return_book(borrow_id: int) -> dict:
    result = execute_write_fetch("""
        UPDATE borrows SET return_date = CURRENT_DATE, status = 'returned'
        WHERE id = %(id)s AND status = 'active'
        RETURNING id, return_date
    """, {"id": borrow_id})
    if not result:
        return {"error": "Borrow record not found or already returned"}
    return {"borrow_id": result[0]["id"], "return_date": str(result[0]["return_date"]), "status": "returned"}


def renew_book(borrow_id: int) -> dict:
    rows = execute_query("SELECT due_date, renewed_count FROM borrows WHERE id = %(id)s AND status = 'active'", {"id": borrow_id})
    if not rows:
        return {"error": "Not found or not active"}
    if rows[0]["renewed_count"] >= settings.max_renewals:
        return {"error": f"Max renewals ({settings.max_renewals}) reached"}
    new_due = rows[0]["due_date"] + timedelta(days=settings.max_borrow_days)
    result = execute_write_fetch("""
        UPDATE borrows SET due_date = %(due)s, renewed_count = renewed_count + 1
        WHERE id = %(id)s RETURNING id, due_date, renewed_count
    """, {"id": borrow_id, "due": new_due})
    if result:
        return {"borrow_id": result[0]["id"], "new_due_date": str(result[0]["due_date"]),
                "renewals_remaining": settings.max_renewals - result[0]["renewed_count"]}
    return {"error": "Renewal failed"}
