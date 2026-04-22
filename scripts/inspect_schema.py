"""
Inspect and compare the two candidate Neon Postgres databases.

The CS 410 Group 3 project has two Neon databases, each with a candidate
schema already deployed by teammates. Before writing any ingestion code we
need to know:

    1. What tables and columns exist in each database.
    2. Whether the pgvector extension is installed (required by the paper).
    3. Current row counts (to see if one DB is already partially loaded).
    4. Where the two schemas diverge (so we can pick one or reconcile).

The output of this script is purely diagnostic. It never writes to either
database. Run it with:

    uv run python scripts/inspect_schema.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


# Tables the milestone paper says should exist. We surface missing ones loudly
# so a partially-built schema is obvious at a glance.
EXPECTED_TABLES = (
    "books",
    "inventory",
    "borrows",
    "users",
    "reviews",
    "search_logs",
)


@dataclass
class ColumnInfo:
    """One row from information_schema.columns, trimmed to what we care about."""

    name: str
    data_type: str
    is_nullable: str
    column_default: str | None


@dataclass
class TableInfo:
    """A table and its columns within one database."""

    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int | None = None


@dataclass
class DatabaseSnapshot:
    """Everything we learned about a single Neon database."""

    label: str
    host: str
    tables: dict[str, TableInfo] = field(default_factory=dict)
    pgvector_installed: bool = False
    pgvector_version: str | None = None
    connection_error: str | None = None


def _host_from_url(url: str) -> str:
    """Extract just the host portion of a Postgres URL, for display only."""
    # Quick and dirty; we never log the credential portion.
    try:
        after_at = url.split("@", 1)[1]
        host = after_at.split("/", 1)[0]
        # Strip :port if present.
        return host.split(":", 1)[0]
    except IndexError:
        return "unknown"


def _fetch_columns(cursor: Any, table_name: str) -> list[ColumnInfo]:
    """Read column metadata for one table from information_schema."""
    cursor.execute(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    )
    return [
        ColumnInfo(
            name=row["column_name"],
            data_type=row["data_type"],
            is_nullable=row["is_nullable"],
            column_default=row["column_default"],
        )
        for row in cursor.fetchall()
    ]


def _fetch_row_count(cursor: Any, table_name: str) -> int | None:
    """Return the row count for a table, or None if the count fails."""
    # We quote with format_ident semantics via psycopg2's identifier safety.
    # Since table_name comes from information_schema we trust it, but still
    # use a literal-safe approach.
    try:
        cursor.execute(f'SELECT COUNT(*) AS n FROM public."{table_name}"')
        row = cursor.fetchone()
        return int(row["n"]) if row else None
    except Exception:
        # Don't let a single bad table kill the whole snapshot.
        return None


def inspect_database(label: str, url: str) -> DatabaseSnapshot:
    """Connect to one Neon database and collect a read-only snapshot."""
    snapshot = DatabaseSnapshot(label=label, host=_host_from_url(url))

    try:
        # Neon sometimes needs a moment to wake from idle; a short connect
        # timeout still handles that in practice.
        conn = psycopg2.connect(url, connect_timeout=30)
    except Exception as exc:
        # Surface the failure in the report instead of crashing the whole
        # script, so the other DB still gets inspected.
        snapshot.connection_error = f"{type(exc).__name__}: {exc}"
        return snapshot

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # pgvector check: the paper requires vector(768) columns and HNSW
            # support, so we flag absence up front.
            cursor.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            row = cursor.fetchone()
            if row is not None:
                snapshot.pgvector_installed = True
                snapshot.pgvector_version = row["extversion"]

            # List every table in the public schema (not just expected ones,
            # so we catch anything extra a teammate added).
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            table_names = [row["table_name"] for row in cursor.fetchall()]

            for table_name in table_names:
                table = TableInfo(name=table_name)
                table.columns = _fetch_columns(cursor, table_name)
                table.row_count = _fetch_row_count(cursor, table_name)
                snapshot.tables[table_name] = table
    finally:
        conn.close()

    return snapshot


def _format_column(col: ColumnInfo) -> str:
    null = "" if col.is_nullable == "YES" else " NOT NULL"
    default = f" DEFAULT {col.column_default}" if col.column_default else ""
    return f"  {col.name}: {col.data_type}{null}{default}"


def print_snapshot(snapshot: DatabaseSnapshot) -> None:
    """Human-readable dump of one database snapshot."""
    print("=" * 72)
    print(f"Database: {snapshot.label}")
    print(f"Host:     {snapshot.host}")
    print("=" * 72)

    if snapshot.connection_error:
        print(f"CONNECTION FAILED: {snapshot.connection_error}")
        return

    if snapshot.pgvector_installed:
        print(f"pgvector: INSTALLED (version {snapshot.pgvector_version})")
    else:
        print("pgvector: NOT INSTALLED (blocker for the embedding pipeline)")

    missing = [t for t in EXPECTED_TABLES if t not in snapshot.tables]
    extra = [t for t in snapshot.tables if t not in EXPECTED_TABLES]
    if missing:
        print(f"Missing expected tables: {', '.join(missing)}")
    if extra:
        print(f"Unexpected extra tables: {', '.join(extra)}")

    print()
    for table_name in sorted(snapshot.tables):
        table = snapshot.tables[table_name]
        count_str = f"{table.row_count:,}" if table.row_count is not None else "?"
        print(f"[{table_name}]  rows={count_str}")
        for col in table.columns:
            print(_format_column(col))
        print()


def diff_snapshots(a: DatabaseSnapshot, b: DatabaseSnapshot) -> None:
    """Highlight the structural differences between the two databases."""
    print("=" * 72)
    print(f"DIFF: {a.label} vs {b.label}")
    print("=" * 72)

    if a.connection_error or b.connection_error:
        print("Cannot diff; at least one database failed to connect.")
        return

    only_a = sorted(set(a.tables) - set(b.tables))
    only_b = sorted(set(b.tables) - set(a.tables))
    both = sorted(set(a.tables) & set(b.tables))

    if only_a:
        print(f"Tables only in {a.label}: {', '.join(only_a)}")
    if only_b:
        print(f"Tables only in {b.label}: {', '.join(only_b)}")
    if not only_a and not only_b:
        print("Same table set in both databases.")

    for table_name in both:
        cols_a = {c.name: c for c in a.tables[table_name].columns}
        cols_b = {c.name: c for c in b.tables[table_name].columns}

        missing_in_b = sorted(set(cols_a) - set(cols_b))
        missing_in_a = sorted(set(cols_b) - set(cols_a))

        type_mismatches = []
        for name in sorted(set(cols_a) & set(cols_b)):
            if cols_a[name].data_type != cols_b[name].data_type:
                type_mismatches.append(
                    f"    {name}: {a.label}={cols_a[name].data_type} "
                    f"{b.label}={cols_b[name].data_type}"
                )

        if missing_in_a or missing_in_b or type_mismatches:
            print(f"\n  [{table_name}]")
            if missing_in_b:
                print(f"    columns only in {a.label}: {', '.join(missing_in_b)}")
            if missing_in_a:
                print(f"    columns only in {b.label}: {', '.join(missing_in_a)}")
            for line in type_mismatches:
                print(line)

    print()


def main() -> int:
    load_dotenv()
    url_1 = os.environ.get("DATABASE_URL_1")
    url_2 = os.environ.get("DATABASE_URL_2")

    if not url_1 or not url_2:
        # Fail loudly rather than silently connect to one DB only; this script
        # exists specifically to compare the two candidates.
        print(
            "ERROR: DATABASE_URL_1 and DATABASE_URL_2 must both be set in .env",
            file=sys.stderr,
        )
        return 1

    snapshot_1 = inspect_database("DB_1", url_1)
    snapshot_2 = inspect_database("DB_2", url_2)

    print_snapshot(snapshot_1)
    print()
    print_snapshot(snapshot_2)
    print()
    diff_snapshots(snapshot_1, snapshot_2)

    # Non-zero exit if either DB failed to connect, so CI / shell scripts can
    # detect failure without parsing output.
    if snapshot_1.connection_error or snapshot_2.connection_error:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
