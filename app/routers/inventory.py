"""Inventory endpoints: borrow, return, renew, availability."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.inventory import borrow_book, get_book_availability, renew_book, return_book

router = APIRouter(tags=["inventory"])

class BorrowRequest(BaseModel):
    isbn13: str
    user_id: str

class ReturnRequest(BaseModel):
    borrow_id: int

class RenewRequest(BaseModel):
    borrow_id: int

@router.get("/availability/{isbn13}")
def check_availability(isbn13: str):
    result = get_book_availability(isbn13)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.post("/borrow")
def borrow(req: BorrowRequest):
    result = borrow_book(req.isbn13, req.user_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/return")
def return_borrowed(req: ReturnRequest):
    result = return_book(req.borrow_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/renew")
def renew(req: RenewRequest):
    result = renew_book(req.borrow_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
