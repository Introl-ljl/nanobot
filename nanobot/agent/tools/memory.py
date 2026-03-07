"""Memory retrieval tools with semantic search and contextual read."""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litellm import aembedding
from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.base import Tool
from nanobot.config.loader import load_config
from nanobot.utils.helpers import ensure_dir


@dataclass
class MemoryRetrievalConfig:
    enabled: bool = True
    search_paths: list[str] | None = None
    extra_paths: list[str] | None = None
    index_db_path: str = "memory/.memory_index.sqlite3"
    embedding_model: str = ""
    embedding_provider: str | None = None
    top_k: int = 8
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 150
    sync_debounce_ms: int = 1200
    max_context_lines: int = 80

    @classmethod
    def from_obj(cls, cfg: Any | None) -> "MemoryRetrievalConfig":
        if cfg is None:
            return cls(search_paths=["memory/MEMORY.md", "memory/**/*.md"], extra_paths=[])
        data = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
        return cls(
            enabled=data.get("enabled", True),
            search_paths=data.get("search_paths") or ["memory/MEMORY.md", "memory/**/*.md"],
            extra_paths=data.get("extra_paths") or [],
            index_db_path=data.get("index_db_path", "memory/.memory_index.sqlite3"),
            embedding_model=data.get("embedding_model", ""),
            embedding_provider=data.get("embedding_provider"),
            top_k=data.get("top_k", 8),
            chunk_size_chars=data.get("chunk_size_chars", 1000),
            chunk_overlap_chars=data.get("chunk_overlap_chars", 150),
            sync_debounce_ms=data.get("sync_debounce_ms", 1200),
            max_context_lines=data.get("max_context_lines", 80),
        )


class MemoryIndex:
    """SQLite-backed memory chunk index with incremental sync."""

    def __init__(self, workspace: Path, cfg: MemoryRetrievalConfig):
        self.workspace = workspace.resolve()
        self.cfg = cfg
        db_path = Path(cfg.index_db_path)
        if not db_path.is_absolute():
            db_path = self.workspace / db_path
        ensure_dir(db_path.parent)
        self.db_path = db_path
        self._sync_lock = asyncio.Lock()
        self._last_sync_monotonic = 0.0
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files(
                    path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    heading TEXT,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    UNIQUE(path, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS embeddings(
                    chunk_id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
                """
            )

    def _resolve_patterns(self, extra_paths: list[str] | None = None) -> list[str]:
        patterns = list(self.cfg.search_paths or ["memory/MEMORY.md", "memory/**/*.md"])
        patterns.extend(self.cfg.extra_paths or [])
        patterns.extend(extra_paths or [])
        cleaned = []
        seen = set()
        for p in patterns:
            if p not in seen:
                seen.add(p)
                cleaned.append(p)
        return cleaned

    def _resolve_target_files(self, extra_paths: list[str] | None = None) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        for pat in self._resolve_patterns(extra_paths):
            path = Path(pat).expanduser()
            if path.is_absolute():
                candidates = [path] if path.is_file() else sorted(path.glob("**/*.md")) if path.is_dir() else []
            else:
                if any(ch in pat for ch in ("*", "?", "[")):
                    candidates = sorted(self.workspace.glob(pat))
                else:
                    abs_path = self.workspace / pat
                    if abs_path.is_file():
                        candidates = [abs_path]
                    elif abs_path.is_dir():
                        candidates = sorted(abs_path.glob("**/*.md"))
                    else:
                        candidates = []
            for c in candidates:
                if not c.exists() or not c.is_file() or c.suffix.lower() != ".md":
                    continue
                r = c.resolve()
                if r not in seen:
                    seen.add(r)
                    out.append(r)
        return sorted(out)

    def _path_key(self, file_path: Path) -> str:
        resolved = file_path.resolve()
        try:
            return str(resolved.relative_to(self.workspace))
        except ValueError:
            return str(resolved)

    def _tracked_file_map(self) -> dict[str, Path]:
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files").fetchall()
        out: dict[str, Path] = {}
        for row in rows:
            raw_path = str(row["path"])
            path_obj = Path(raw_path)
            resolved = path_obj.resolve() if path_obj.is_absolute() else (self.workspace / path_obj).resolve()
            out[raw_path] = resolved
        return out

    def _resolve_indexed_file(self, path: str) -> tuple[str, Path]:
        tracked = self._tracked_file_map()
        if path in tracked:
            return path, tracked[path]

        requested = Path(path)
        resolved_request = requested.resolve() if requested.is_absolute() else (self.workspace / requested).resolve()
        for display_path, resolved_path in tracked.items():
            if resolved_path == resolved_request:
                return display_path, resolved_path

        raise PermissionError("path is not an indexed memory file")

    async def sync(self, force: bool = False) -> dict[str, int]:
        async with self._sync_lock:
            now = time.monotonic()
            debounce_s = max(0, self.cfg.sync_debounce_ms) / 1000.0
            if not force and (now - self._last_sync_monotonic) < debounce_s:
                return {"updated_files": 0, "removed_files": 0}

            target_files = self._resolve_target_files()
            current_map = {self._path_key(p): p for p in target_files}

            with self._connect() as conn:
                tracked = {
                    row["path"]: row
                    for row in conn.execute("SELECT path, mtime_ns, size, content_hash FROM files")
                }

                removed = [path for path in tracked if path not in current_map]
                for path in removed:
                    conn.execute("DELETE FROM files WHERE path=?", (path,))
                    conn.execute("DELETE FROM chunks WHERE path=?", (path,))

                updated_files = 0
                for rel_path, abs_path in current_map.items():
                    stat = abs_path.stat()
                    tracked_row = tracked.get(rel_path)
                    if (
                        tracked_row
                        and int(tracked_row["mtime_ns"]) == int(stat.st_mtime_ns)
                        and int(tracked_row["size"]) == int(stat.st_size)
                    ):
                        continue

                    content = abs_path.read_text(encoding="utf-8")
                    content_hash = MemoryStore.content_hash(content)
                    if tracked_row and tracked_row["content_hash"] == content_hash:
                        conn.execute(
                            "UPDATE files SET mtime_ns=?, size=? WHERE path=?",
                            (int(stat.st_mtime_ns), int(stat.st_size), rel_path),
                        )
                        continue

                    chunks = self._chunk_markdown(content)
                    conn.execute("DELETE FROM chunks WHERE path=?", (rel_path,))
                    chunk_rows: list[tuple[int, str]] = []
                    for idx, chunk in enumerate(chunks):
                        cursor = conn.execute(
                            """
                            INSERT INTO chunks(path, chunk_index, start_line, end_line, heading, text, text_hash)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rel_path,
                                idx,
                                chunk["start_line"],
                                chunk["end_line"],
                                chunk["heading"],
                                chunk["text"],
                                MemoryStore.content_hash(chunk["text"]),
                            ),
                        )
                        chunk_rows.append((int(cursor.lastrowid), chunk["text"]))

                    conn.execute(
                        """
                        INSERT INTO files(path, mtime_ns, size, content_hash)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            mtime_ns=excluded.mtime_ns,
                            size=excluded.size,
                            content_hash=excluded.content_hash
                        """,
                        (rel_path, int(stat.st_mtime_ns), int(stat.st_size), content_hash),
                    )
                    conn.execute(
                        """
                        DELETE FROM embeddings
                        WHERE chunk_id NOT IN (SELECT id FROM chunks)
                        """
                    )
                    conn.commit()

                    if chunk_rows and self.cfg.embedding_model.strip():
                        vectors = await self._embed_texts([c[1] for c in chunk_rows])
                        if vectors:
                            conn.executemany(
                                """
                                INSERT INTO embeddings(chunk_id, model, dim, vector_json)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT(chunk_id) DO UPDATE SET
                                    model=excluded.model,
                                    dim=excluded.dim,
                                    vector_json=excluded.vector_json
                                """,
                                [
                                    (
                                        chunk_id,
                                        self.cfg.embedding_model,
                                        len(vector),
                                        json.dumps(vector),
                                    )
                                    for (chunk_id, _), vector in zip(chunk_rows, vectors, strict=False)
                                ],
                            )
                            conn.commit()

                    updated_files += 1

            self._last_sync_monotonic = time.monotonic()
            return {"updated_files": updated_files, "removed_files": len(removed)}

    def _chunk_markdown(self, content: str) -> list[dict[str, Any]]:
        lines = content.splitlines()
        if not lines:
            return []

        sections: list[tuple[str | None, int, list[str]]] = []
        current_heading: str | None = None
        current_start = 1
        current_lines: list[str] = []

        for lineno, line in enumerate(lines, start=1):
            if line.lstrip().startswith("#") and current_lines:
                sections.append((current_heading, current_start, current_lines))
                current_heading = line.lstrip("#").strip() or None
                current_start = lineno
                current_lines = [line]
                continue
            if line.lstrip().startswith("#") and not current_lines:
                current_heading = line.lstrip("#").strip() or None
                current_start = lineno
            current_lines.append(line)

        if current_lines:
            sections.append((current_heading, current_start, current_lines))

        chunks: list[dict[str, Any]] = []
        for heading, section_start, section_lines in sections:
            i = 0
            while i < len(section_lines):
                total_chars = 0
                j = i
                while j < len(section_lines):
                    line_chars = len(section_lines[j]) + 1
                    if j > i and total_chars + line_chars > self.cfg.chunk_size_chars:
                        break
                    total_chars += line_chars
                    j += 1
                if j <= i:
                    j = i + 1

                text = "\n".join(section_lines[i:j]).strip()
                if text:
                    chunks.append(
                        {
                            "heading": heading,
                            "text": text,
                            "start_line": section_start + i,
                            "end_line": section_start + j - 1,
                        }
                    )

                if j >= len(section_lines):
                    break

                if self.cfg.chunk_overlap_chars <= 0:
                    i = j
                    continue

                overlap = 0
                k = j - 1
                while k > i and overlap < self.cfg.chunk_overlap_chars:
                    overlap += len(section_lines[k]) + 1
                    k -= 1
                i = max(i + 1, k + 1)

        return chunks

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        config = load_config()
        model = self.cfg.embedding_model.strip()
        if not model:
            return []
        resolved_model = self._resolve_embedding_model(config, model)
        provider = config.get_provider(model)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "input": texts,
        }
        if provider and provider.api_key:
            kwargs["api_key"] = provider.api_key
        api_base = config.get_api_base(model)
        if api_base:
            kwargs["api_base"] = api_base
        if provider and provider.extra_headers:
            kwargs["extra_headers"] = provider.extra_headers

        resp: Any | None = None
        try:
            resp = await aembedding(**kwargs)
        except Exception as exc:
            retry_models = self._embedding_retry_models(config, model, resolved_model, exc)
            if not retry_models:
                logger.exception("Memory embedding call failed")
                return []
            for retry_model in retry_models:
                retry_kwargs = dict(kwargs)
                retry_kwargs["model"] = retry_model
                try:
                    logger.warning(
                        "Memory embedding provider missing for model '{}', retry with '{}'",
                        resolved_model,
                        retry_model,
                    )
                    resp = await aembedding(**retry_kwargs)
                    break
                except Exception:
                    logger.exception("Memory embedding retry failed for model '{}'", retry_model)
            if resp is None:
                return []

        data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data:
            return []

        vectors: list[list[float]] = []
        for item in data:
            emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if isinstance(emb, list):
                vectors.append([float(v) for v in emb])
        return vectors

    def _resolve_embedding_model(self, config, model: str) -> str:
        if "/" in model:
            return model
        provider_name = self.cfg.embedding_provider or config.get_provider_name(model)
        if provider_name:
            return f"{provider_name}/{model}"
        return model

    def _embedding_retry_models(
        self,
        config,
        model: str,
        resolved_model: str,
        exc: Exception,
    ) -> list[str]:
        if not self._is_missing_provider_error(exc):
            return []
        out: list[str] = []
        seen = {resolved_model}

        provider_candidates = [self.cfg.embedding_provider, config.get_provider_name(model)]
        for provider_name in provider_candidates:
            if not provider_name:
                continue
            if self._model_has_prefix(model, provider_name):
                continue
            retry_model = f"{provider_name}/{model}"
            if retry_model in seen:
                continue
            seen.add(retry_model)
            out.append(retry_model)

        if "/" in model and not self._model_has_prefix(model, "huggingface"):
            retry_model = f"huggingface/{model}"
            if retry_model not in seen:
                out.append(retry_model)
        return out

    @staticmethod
    def _is_missing_provider_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "llm provider not provided" in msg or "provider not provided" in msg

    @staticmethod
    def _model_has_prefix(model: str, prefix: str) -> bool:
        if "/" not in model:
            return False
        model_prefix = model.split("/", 1)[0].lower().replace("-", "_")
        normalized_prefix = prefix.lower().replace("-", "_")
        return model_prefix == normalized_prefix

    async def semantic_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self.cfg.embedding_model.strip():
            raise ValueError("memory embedding model is not configured")
        await self.sync(force=True)
        query_vectors = await self._embed_texts([query])
        if not query_vectors:
            return []
        qv = query_vectors[0]
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.path, c.start_line, c.end_line, c.heading, c.text, e.vector_json
                FROM chunks c
                JOIN embeddings e ON e.chunk_id = c.id
                """
            ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            vec = json.loads(row["vector_json"])
            score = self._cosine_similarity(qv, vec)
            preview = row["text"].replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            scored.append(
                (
                    score,
                    {
                        "path": row["path"],
                        "start_line": int(row["start_line"]),
                        "end_line": int(row["end_line"]),
                        "heading": row["heading"],
                        "score": round(score, 4),
                        "preview": preview,
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        limit = top_k or self.cfg.top_k
        return [item for _, item in scored[:max(1, limit)]]

    async def keyword_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        await self.sync(force=True)
        raw_terms = [term.lower() for term in re.findall(r"\w+", query)]
        terms = [term for term in raw_terms if len(term) >= 2]
        compact_query = re.sub(r"\s+", "", query).strip().lower()
        if compact_query and compact_query not in terms:
            terms.append(compact_query)
        if not terms and query.strip():
            terms = [query.strip().lower()]
        if not terms:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path, start_line, end_line, heading, text
                FROM chunks
                """
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            heading = (row["heading"] or "").lower()
            text = row["text"] or ""
            haystack = text.lower()
            score = 0.0
            for term in terms:
                heading_hits = heading.count(term)
                text_hits = haystack.count(term)
                score += heading_hits * 3.0 + text_hits * 1.0
            if score <= 0:
                continue
            preview = text.replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            scored.append(
                (
                    score,
                    {
                        "path": row["path"],
                        "start_line": int(row["start_line"]),
                        "end_line": int(row["end_line"]),
                        "heading": row["heading"],
                        "score": round(score, 4),
                        "preview": preview,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        limit = top_k or self.cfg.top_k
        return [item for _, item in scored[:max(1, limit)]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return -1.0
        return dot / (norm_a * norm_b)

    def get_context(self, path: str, line: int, context_lines: int) -> dict[str, Any]:
        display_path, file_path = self._resolve_indexed_file(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(path)
        all_lines = file_path.read_text(encoding="utf-8").splitlines()
        max_line = len(all_lines)
        line = max(1, min(line, max_line if max_line > 0 else 1))
        context_lines = max(1, min(context_lines, self.cfg.max_context_lines))
        start = max(1, line - context_lines)
        end = min(max_line, line + context_lines)
        snippet = "\n".join(all_lines[start - 1:end])
        return {
            "path": display_path,
            "line": line,
            "start_line": start,
            "end_line": end,
            "content": snippet,
        }


class MemorySearchTool(Tool):
    """Search markdown memory files using embeddings or keyword fallback."""

    def __init__(self, workspace: Path, cfg: MemoryRetrievalConfig, index: MemoryIndex | None = None):
        self._index = index or MemoryIndex(workspace, cfg)
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return "Search memory markdown files and return candidate snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, top_k: int | None = None, **kwargs: Any) -> str:
        try:
            if self._cfg.embedding_model.strip():
                candidates = await self._index.semantic_search(query, top_k=top_k)
                mode = "semantic"
            else:
                candidates = await self._index.keyword_search(query, top_k=top_k)
                mode = "keyword"
            payload = {
                "query": query,
                "mode": mode,
                "count": len(candidates),
                "candidates": candidates,
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Error: memory_search failed: {e}"

    async def maybe_sync(self, force: bool = False) -> None:
        try:
            await self._index.sync(force=force)
        except Exception:
            logger.exception("Memory index sync failed")


class MemoryGetTool(Tool):
    """Read source context for a memory candidate."""

    def __init__(self, workspace: Path, cfg: MemoryRetrievalConfig, index: MemoryIndex | None = None):
        self._index = index or MemoryIndex(workspace, cfg)
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "memory_get"

    @property
    def description(self) -> str:
        return "Read the source markdown context around a given file path and line number."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path from memory_search candidate."},
                "line": {"type": "integer", "minimum": 1, "description": "Target line number."},
                "context_lines": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["path", "line"],
        }

    async def execute(self, path: str, line: int, context_lines: int = 20, **kwargs: Any) -> str:
        try:
            await self._index.sync(force=False)
            data = self._index.get_context(path, line, context_lines)
            header = (
                f"File: {data['path']}\n"
                f"Focus line: {data['line']}\n"
                f"Context lines: {data['start_line']}-{data['end_line']}\n"
            )
            return header + "\n" + data["content"]
        except Exception as e:
            return f"Error: memory_get failed: {e}"
