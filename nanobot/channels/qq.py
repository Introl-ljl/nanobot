"""QQ channel implementation using the official QQ Bot API semantics."""

from __future__ import annotations

import asyncio
import base64
import json
import inspect
import mimetypes
import re
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.loader import get_data_dir
from nanobot.config.schema import QQConfig
from nanobot.utils.helpers import ensure_dir, safe_filename

try:
    import botpy
    from botpy.message import C2CMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage

QQ_API_BASE = "https://api.sgroup.qq.com"
QQ_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
_TOKEN_EARLY_REFRESH_SECONDS = 300
_REPLY_LIMIT = 4
_REPLY_TTL_SECONDS = 60 * 60
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_MENTION_RE = re.compile(r"<@!?\d+>")
_FACE_TAG_RE = re.compile(r'<faceType=\d+,faceId="[^"]*",ext="([^"]*)">')
_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml"}
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".silk", ".amr", ".m4a"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class QQError(RuntimeError):
    """Raised when the QQ API returns an error."""


class QQReplyLimitTracker:
    """Track passive reply quotas for QQ message IDs."""

    def __init__(self, limit: int = _REPLY_LIMIT, ttl_seconds: int = _REPLY_TTL_SECONDS):
        self.limit = limit
        self.ttl_seconds = ttl_seconds
        self._records: dict[str, tuple[int, float]] = {}

    def check(self, message_id: str | None) -> dict[str, Any]:
        if not message_id:
            return {
                "allowed": False,
                "remaining": 0,
                "should_fallback_to_proactive": True,
                "fallback_reason": "missing",
            }

        now = time.time()
        self._cleanup(now)
        record = self._records.get(message_id)
        if not record:
            return {
                "allowed": True,
                "remaining": self.limit,
                "should_fallback_to_proactive": False,
            }

        count, first_reply_at = record
        if now - first_reply_at > self.ttl_seconds:
            self._records.pop(message_id, None)
            return {
                "allowed": False,
                "remaining": 0,
                "should_fallback_to_proactive": True,
                "fallback_reason": "expired",
            }

        remaining = self.limit - count
        if remaining <= 0:
            return {
                "allowed": False,
                "remaining": 0,
                "should_fallback_to_proactive": True,
                "fallback_reason": "limit_exceeded",
            }

        return {
            "allowed": True,
            "remaining": remaining,
            "should_fallback_to_proactive": False,
        }

    def record(self, message_id: str | None) -> None:
        if not message_id:
            return

        now = time.time()
        count, first_reply_at = self._records.get(message_id, (0, now))
        if now - first_reply_at > self.ttl_seconds:
            count = 0
            first_reply_at = now
        self._records[message_id] = (count + 1, first_reply_at)
        self._cleanup(now)

    def _cleanup(self, now: float | None = None) -> None:
        current = now if now is not None else time.time()
        expired = [
            message_id
            for message_id, (_count, first_reply_at) in self._records.items()
            if current - first_reply_at > self.ttl_seconds
        ]
        for message_id in expired:
            self._records.pop(message_id, None)


_SENTINEL = object()


def _pick(source: Any, *path: str, default: Any = None) -> Any:
    current = source
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, _SENTINEL)
        else:
            current = getattr(current, key, _SENTINEL)
        if current is _SENTINEL:
            return default
    return current


def _parse_face_tags(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        try:
            decoded = base64.b64decode(match.group(1)).decode("utf-8")
            payload = json.loads(decoded)
        except Exception:
            return match.group(0)
        face_name = payload.get("text") or "未知表情"
        return f"【表情: {face_name}】"

    return _FACE_TAG_RE.sub(repl, text)


def _strip_mentions(text: str) -> str:
    cleaned = _MENTION_RE.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _looks_like_remote_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://") or value.startswith("data:")


def _guess_media_kind(name: str, content_type: str | None = None) -> Literal["image", "voice", "video", "file"]:
    lowered_content_type = (content_type or "").lower()
    suffix = Path(name.split("?", 1)[0]).suffix.lower()
    if lowered_content_type.startswith("image/") or suffix in _IMAGE_EXTENSIONS:
        return "image"
    if lowered_content_type.startswith("audio/") or suffix in _AUDIO_EXTENSIONS:
        return "voice"
    if lowered_content_type.startswith("video/") or suffix in _VIDEO_EXTENSIONS:
        return "video"
    return "file"


def _guess_mime_type(path_or_name: str) -> str:
    mime_type, _ = mimetypes.guess_type(path_or_name)
    return mime_type or "application/octet-stream"


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True, guild_messages=True)

    class _Bot(botpy.Client):
        def __init__(self):
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", getattr(self.robot, "name", "unknown"))

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_c2c_message(message)

        async def on_direct_message_create(self, message):
            await channel._on_direct_message(message)

        async def on_group_at_message_create(self, message):
            await channel._on_group_message(message)

        async def on_at_message_create(self, message):
            await channel._on_guild_message(message)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel with C2C, group, channel, media and voice support."""

    name = "qq"

    def __init__(self, config: QQConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque[str] = deque(maxlen=1000)
        self._token_cache: dict[str, Any] | None = None
        self._token_lock = asyncio.Lock()
        self._reply_tracker = QQReplyLimitTracker()
        self._media_dir = ensure_dir(get_data_dir() / "media" / "qq")

    async def start(self) -> None:
        """Start the QQ bot and begin listening for events."""
        secret = self._resolve_client_secret()
        if not self.config.app_id or not secret:
            logger.error("QQ app_id and secret not configured")
            return

        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        self._running = True
        bot_class = _make_bot_class(self)
        self._client = bot_class()

        logger.info("QQ bot started with scenes: {}", ",".join(self.config.enabled_scenes))
        await self._run_bot(secret)

    async def _run_bot(self, secret: str) -> None:
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=secret)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("QQ bot error: {}", exc)
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                logger.debug("QQ close ignored", exc_info=True)
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        target = self._parse_target(msg.chat_id)
        reply_to_id = msg.reply_to or self._coerce_str((msg.metadata or {}).get("message_id"))
        text = msg.content or ""

        if (msg.metadata or {}).get("qq_voice") and text and self.config.tts.enabled:
            tts_path = await self._synthesize_tts(text)
            if tts_path:
                msg.media = [tts_path, *(msg.media or [])]
                if not (msg.metadata or {}).get("qq_voice_keep_text"):
                    text = ""

        media_sources = self._collect_media_sources(msg)
        access_token = await self._get_access_token()

        if media_sources:
            for index, media_source in enumerate(media_sources):
                await self._send_media(
                    access_token=access_token,
                    target=target,
                    source=media_source,
                    content=text if index == 0 else "",
                    reply_to_id=reply_to_id,
                )
                if reply_to_id:
                    reply_to_id = None
            return

        if not text.strip():
            logger.warning("QQ send skipped: empty text and no media")
            return

        await self._send_text(
            access_token=access_token,
            target=target,
            content=text,
            reply_to_id=reply_to_id,
        )

    async def _send_text(
        self,
        *,
        access_token: str,
        target: dict[str, str],
        content: str,
        reply_to_id: str | None,
    ) -> dict[str, Any]:
        passive_allowed = False
        fallback_to_proactive = False
        if reply_to_id:
            limit = self._reply_tracker.check(reply_to_id)
            passive_allowed = bool(limit["allowed"])
            fallback_to_proactive = bool(limit["should_fallback_to_proactive"]) and target["type"] in {"c2c", "group"}

        if passive_allowed:
            result = await self._send_passive_text(access_token, target, content, reply_to_id)
            self._reply_tracker.record(reply_to_id)
            return result

        if reply_to_id and fallback_to_proactive:
            return await self._send_proactive_text(access_token, target, content)

        if target["type"] == "channel":
            return await self._send_channel_text(access_token, target["id"], content, reply_to_id)

        return await self._send_proactive_text(access_token, target, content)

    async def _send_media(
        self,
        *,
        access_token: str,
        target: dict[str, str],
        source: str,
        content: str,
        reply_to_id: str | None,
    ) -> dict[str, Any]:
        kind = _guess_media_kind(source)
        if target["type"] == "channel":
            if content.strip():
                return await self._send_channel_text(access_token, target["id"], content, reply_to_id)
            raise QQError("channel media sending is not supported by this QQ channel implementation")

        upload = await self._upload_media(
            access_token=access_token,
            target=target,
            source=source,
            kind=kind,
        )
        result = await self._send_uploaded_media(
            access_token=access_token,
            target=target,
            file_info=upload["file_info"],
            reply_to_id=reply_to_id,
            content=content,
        )
        if reply_to_id:
            self._reply_tracker.record(reply_to_id)
        return result

    async def _send_passive_text(
        self,
        access_token: str,
        target: dict[str, str],
        content: str,
        reply_to_id: str,
    ) -> dict[str, Any]:
        if target["type"] in {"c2c", "dm"}:
            return await self._api_request(
                access_token,
                "POST",
                f"/v2/users/{target['id']}/messages",
                self._build_text_body(content, reply_to_id),
            )
        if target["type"] == "group":
            return await self._api_request(
                access_token,
                "POST",
                f"/v2/groups/{target['id']}/messages",
                self._build_text_body(content, reply_to_id),
            )
        return await self._send_channel_text(access_token, target["id"], content, reply_to_id)

    async def _send_proactive_text(
        self,
        access_token: str,
        target: dict[str, str],
        content: str,
    ) -> dict[str, Any]:
        body = self._build_proactive_text_body(content)
        if target["type"] in {"c2c", "dm"}:
            return await self._api_request(access_token, "POST", f"/v2/users/{target['id']}/messages", body)
        if target["type"] == "group":
            return await self._api_request(access_token, "POST", f"/v2/groups/{target['id']}/messages", body)
        return await self._send_channel_text(access_token, target["id"], content, None)

    async def _send_channel_text(
        self,
        access_token: str,
        channel_id: str,
        content: str,
        reply_to_id: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"content": content}
        if reply_to_id:
            body["msg_id"] = reply_to_id
        return await self._api_request(access_token, "POST", f"/channels/{channel_id}/messages", body)

    async def _upload_media(
        self,
        *,
        access_token: str,
        target: dict[str, str],
        source: str,
        kind: Literal["image", "voice", "video", "file"],
    ) -> dict[str, Any]:
        file_type_map = {
            "image": 1,
            "video": 2,
            "voice": 3,
            "file": 4,
        }
        body: dict[str, Any] = {"file_type": file_type_map[kind], "srv_send_msg": False}
        if _looks_like_remote_url(source) and not source.startswith("data:"):
            body["url"] = source
        else:
            body["file_data"] = await self._read_media_as_base64(source)
            if kind == "file":
                body["file_name"] = safe_filename(Path(source).name)

        if target["type"] in {"c2c", "dm"}:
            return await self._api_request(access_token, "POST", f"/v2/users/{target['id']}/files", body)
        if target["type"] == "group":
            return await self._api_request(access_token, "POST", f"/v2/groups/{target['id']}/files", body)
        raise QQError("QQ channel media upload only supports c2c/group targets")

    async def _send_uploaded_media(
        self,
        *,
        access_token: str,
        target: dict[str, str],
        file_info: str,
        reply_to_id: str | None,
        content: str,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "msg_type": 7,
            "media": {"file_info": file_info},
            "msg_seq": self._next_msg_seq(reply_to_id),
        }
        if content:
            body["content"] = content
        if reply_to_id:
            body["msg_id"] = reply_to_id
        elif target["type"] in {"c2c", "group"}:
            body.pop("msg_seq", None)

        if target["type"] in {"c2c", "dm"}:
            return await self._api_request(access_token, "POST", f"/v2/users/{target['id']}/messages", body)
        if target["type"] == "group":
            return await self._api_request(access_token, "POST", f"/v2/groups/{target['id']}/messages", body)
        raise QQError("QQ uploaded media send only supports c2c/group targets")

    def _build_text_body(self, content: str, reply_to_id: str | None) -> dict[str, Any]:
        msg_seq = self._next_msg_seq(reply_to_id)
        if self.config.markdown_support:
            body: dict[str, Any] = {
                "markdown": {"content": content},
                "msg_type": 2,
                "msg_seq": msg_seq,
            }
        else:
            body = {"content": content, "msg_type": 0, "msg_seq": msg_seq}
        if reply_to_id:
            body["msg_id"] = reply_to_id
        return body

    def _build_proactive_text_body(self, content: str) -> dict[str, Any]:
        if not content.strip():
            raise QQError("QQ proactive message content cannot be empty")
        if self.config.markdown_support:
            return {"markdown": {"content": content}, "msg_type": 2}
        return {"content": content, "msg_type": 0}

    async def _get_access_token(self) -> str:
        async with self._token_lock:
            if self._token_cache and self._token_cache["expires_at"] > time.time() + _TOKEN_EARLY_REFRESH_SECONDS:
                return str(self._token_cache["token"])

            secret = self._resolve_client_secret()
            if not self.config.app_id or not secret:
                raise QQError("QQ app_id and secret not configured")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    QQ_TOKEN_URL,
                    json={"appId": self.config.app_id, "clientSecret": secret},
                )
                response.raise_for_status()
                payload = response.json()

            token = payload.get("access_token")
            if not token:
                raise QQError(f"Failed to get QQ access token: {payload}")

            expires_in = int(payload.get("expires_in") or 7200)
            self._token_cache = {
                "token": token,
                "expires_at": time.time() + expires_in,
            }
            return str(token)

    async def _api_request(
        self,
        access_token: str,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"QQBot {access_token}",
            "Content-Type": "application/json",
        }
        timeout = 120.0 if "/files" in path else 30.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, f"{QQ_API_BASE}{path}", headers=headers, json=body)
        if response.status_code >= 400:
            detail = response.text[:500]
            raise QQError(f"QQ API error [{path}] {response.status_code}: {detail}")
        if not response.text.strip():
            return {}
        return response.json()

    async def _read_media_as_base64(self, source: str) -> str:
        if source.startswith("data:"):
            if "," not in source:
                raise QQError("Invalid data URL media payload")
            return source.split(",", 1)[1]
        if _looks_like_remote_url(source):
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(source)
                response.raise_for_status()
                return base64.b64encode(response.content).decode("ascii")
        path = Path(source).expanduser()
        if not path.is_file():
            raise QQError(f"QQ media file not found: {source}")
        if path.stat().st_size > _MAX_ATTACHMENT_BYTES:
            raise QQError(f"QQ media file too large: {source}")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _resolve_client_secret(self) -> str:
        if self.config.secret:
            return self.config.secret
        if self.config.client_secret_file:
            try:
                return Path(self.config.client_secret_file).expanduser().read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.error("Failed reading QQ client_secret_file {}: {}", self.config.client_secret_file, exc)
        return ""

    def _parse_target(self, target: str) -> dict[str, str]:
        normalized = (target or "").replace("qq:", "", 1)
        if normalized.startswith("c2c:"):
            return {"type": "c2c", "id": normalized[4:]}
        if normalized.startswith("group:"):
            return {"type": "group", "id": normalized[6:]}
        if normalized.startswith("channel:"):
            return {"type": "channel", "id": normalized[8:]}
        if normalized.startswith("dm:"):
            return {"type": "dm", "id": normalized[3:]}
        return {"type": "c2c", "id": normalized}

    def _next_msg_seq(self, seed: str | None) -> int:
        base = int(time.time() * 1000) & 0xFFFF
        if not seed:
            return base or 1
        return (base ^ (hash(seed) & 0xFFFF)) or 1

    def _collect_media_sources(self, msg: OutboundMessage) -> list[str]:
        sources: list[str] = list(msg.media or [])
        attachments = (msg.metadata or {}).get("attachments")
        if isinstance(attachments, list):
            for attachment in attachments:
                if isinstance(attachment, str):
                    sources.append(attachment)
                elif isinstance(attachment, dict):
                    value = attachment.get("path") or attachment.get("url")
                    if isinstance(value, str) and value:
                        sources.append(value)
        return sources

    async def _on_c2c_message(self, data: Any) -> None:
        await self._on_message(data, event_type="c2c")

    async def _on_direct_message(self, data: Any) -> None:
        await self._on_message(data, event_type="dm")

    async def _on_group_message(self, data: Any) -> None:
        await self._on_message(data, event_type="group")

    async def _on_guild_message(self, data: Any) -> None:
        await self._on_message(data, event_type="channel")

    async def _on_message(self, data: Any, event_type: Literal["c2c", "group", "channel", "dm"] | None = None) -> None:
        try:
            message_id = self._coerce_str(_pick(data, "id"))
            if message_id and message_id in self._processed_ids:
                return
            if message_id:
                self._processed_ids.append(message_id)

            inferred_type = event_type or self._infer_event_type(data)
            if inferred_type not in set(self.config.enabled_scenes):
                return

            sender_id = self._resolve_sender_id(data, inferred_type)
            if not sender_id:
                return
            if not self._is_inbound_allowed(sender_id, inferred_type):
                return

            raw_content = self._coerce_str(_pick(data, "content"))
            normalized_content = _strip_mentions(_parse_face_tags(raw_content))
            attachments = self._extract_attachments(data)
            if inspect.isawaitable(attachments):
                attachments = await attachments
            voice_text = self._transcribe_voice_attachments(attachments)
            if inspect.isawaitable(voice_text):
                voice_text = await voice_text

            if inferred_type == "group" and self._should_ignore_group_message(raw_content, data):
                return

            content_parts = [part for part in [normalized_content, voice_text] if part]
            if not content_parts and attachments:
                content_parts.append("[media message]")
            content = "\n".join(content_parts).strip()
            if not content:
                return

            chat_id = self._build_chat_id(data, inferred_type, sender_id)
            metadata = {
                "message_id": message_id,
                "chat_type": inferred_type,
                "reply_target": chat_id,
                "guild_id": self._coerce_str(_pick(data, "guild_id")),
                "channel_id": self._coerce_str(_pick(data, "channel_id")),
                "group_id": self._coerce_str(_pick(data, "group_id")),
                "group_openid": self._coerce_str(_pick(data, "group_openid")),
                "openid": sender_id,
                "attachments": attachments,
                "mentions_me": self._message_mentions_bot(raw_content),
            }
            media_paths = [item["path"] for item in attachments if isinstance(item.get("path"), str)]

            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media_paths,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Error handling QQ message")

    def _infer_event_type(self, data: Any) -> Literal["c2c", "group", "channel", "dm"]:
        if _pick(data, "group_openid") or _pick(data, "group_id"):
            return "group"
        if _pick(data, "channel_id"):
            return "channel"
        if _pick(data, "guild_id") and not _pick(data, "channel_id"):
            return "dm"
        return "c2c"

    def _resolve_sender_id(self, data: Any, event_type: str) -> str:
        author = _pick(data, "author", default={})
        if event_type == "group":
            return self._coerce_str(_pick(author, "member_openid") or _pick(author, "id"))
        if event_type in {"c2c", "dm", "channel"}:
            return self._coerce_str(
                _pick(author, "user_openid")
                or _pick(author, "id")
                or _pick(author, "union_openid")
            )
        return ""

    def _build_chat_id(self, data: Any, event_type: str, sender_id: str) -> str:
        if event_type == "group":
            return f"group:{self._coerce_str(_pick(data, 'group_openid') or _pick(data, 'group_id'))}"
        if event_type == "channel":
            return f"channel:{self._coerce_str(_pick(data, 'channel_id'))}"
        if event_type == "dm":
            return f"dm:{sender_id}"
        return f"c2c:{sender_id}"

    def _is_inbound_allowed(self, sender_id: str, event_type: str) -> bool:
        if event_type in {"c2c", "dm"}:
            if self.config.dm_policy == "open":
                return True
            return self.is_allowed(sender_id)
        return self.is_allowed(sender_id)

    def _should_ignore_group_message(self, raw_content: str, data: Any) -> bool:
        if self.config.group_policy == "open":
            return False
        group_id = self._coerce_str(_pick(data, "group_openid") or _pick(data, "group_id"))
        if self.config.group_policy == "allowlist":
            return group_id not in set(self.config.group_allow_from)
        if self.config.react_to_mentions_only:
            return not self._message_mentions_bot(raw_content)
        return False

    def _message_mentions_bot(self, raw_content: str) -> bool:
        return bool(_MENTION_RE.search(raw_content or ""))

    async def _extract_attachments(self, data: Any) -> list[dict[str, Any]]:
        attachments = _pick(data, "attachments", default=[])
        if not isinstance(attachments, list):
            return []

        normalized: list[dict[str, Any]] = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            url = self._coerce_str(attachment.get("voice_wav_url") or attachment.get("url"))
            content_type = self._coerce_str(attachment.get("content_type"))
            filename = self._coerce_str(attachment.get("filename")) or "attachment"
            kind = _guess_media_kind(filename or url, content_type)
            item: dict[str, Any] = {
                "type": kind,
                "url": self._coerce_str(attachment.get("url")),
                "voice_wav_url": self._coerce_str(attachment.get("voice_wav_url")),
                "filename": filename,
                "content_type": content_type,
            }
            if url:
                try:
                    path = await self._download_attachment(url, filename)
                    item["path"] = str(path)
                except Exception as exc:
                    logger.warning("Failed to download QQ attachment {}: {}", url, exc)
            normalized.append(item)
        return normalized

    async def _download_attachment(self, url: str, filename: str) -> Path:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        safe_name = safe_filename(filename) or "attachment"
        target = self._media_dir / f"{int(time.time() * 1000)}_{safe_name}"
        target.write_bytes(response.content)
        return target

    async def _transcribe_voice_attachments(self, attachments: list[dict[str, Any]]) -> str:
        if not self.config.stt.enabled:
            return ""
        transcripts: list[str] = []
        for attachment in attachments:
            if attachment.get("type") != "voice":
                continue
            path = attachment.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                text = await self._transcribe_file(Path(path))
            except Exception as exc:
                logger.warning("QQ voice transcription failed for {}: {}", path, exc)
                continue
            if text:
                transcripts.append(f"[语音转写]\n{text}")
        return "\n".join(transcripts)

    async def _transcribe_file(self, path: Path) -> str:
        cfg = self.config.stt
        if not cfg.enabled:
            return ""
        if not cfg.base_url or not cfg.api_key:
            raise QQError("QQ STT enabled but base_url/api_key not configured")
        mime_type = _guess_mime_type(path.name)
        files = {"file": (path.name, path.read_bytes(), mime_type)}
        data = {"model": cfg.model}
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{cfg.base_url.rstrip('/')}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
            payload = response.json()
        return self._coerce_str(payload.get("text")).strip()

    async def _synthesize_tts(self, text: str) -> str | None:
        cfg = self.config.tts
        if not cfg.enabled:
            return None
        if not cfg.base_url or not cfg.api_key or not cfg.model or not cfg.voice:
            raise QQError("QQ TTS enabled but provider fields are incomplete")
        payload = {
            "model": cfg.model,
            "voice": cfg.voice,
            "input": text,
            "response_format": cfg.response_format,
        }
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{cfg.base_url.rstrip('/')}/audio/speech",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        suffix = f".{cfg.response_format.lstrip('.')}" if cfg.response_format else ".mp3"
        target = self._media_dir / f"tts_{int(time.time() * 1000)}{suffix}"
        target.write_bytes(response.content)
        return str(target)

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value)
