import asyncio
import sqlite3
from pathlib import Path


async def mutate() -> None:
    sqlite3.connect("runtime.db")
    Path("runtime.txt").write_text("unsafe route mutation")
    asyncio.create_task(asyncio.sleep(1))
