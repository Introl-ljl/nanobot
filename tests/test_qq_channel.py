from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.qq import QQChannel, QQReplyLimitTracker
from nanobot.config.schema import QQConfig


@pytest.fixture
def qq_channel(tmp_path, monkeypatch) -> QQChannel:
    monkeypatch.setattr("nanobot.channels.qq.get_data_dir", lambda: tmp_path)
    return QQChannel(
        QQConfig(enabled=True, app_id="app", secret="sec", allow_from=["*"]),
        MessageBus(),
    )


def test_qq_config_defaults_cover_extended_fields() -> None:
    cfg = QQConfig()

    assert cfg.enabled_scenes == ["c2c", "group", "channel", "dm"]
    assert cfg.dm_policy == "open"
    assert cfg.markdown_support is True
    assert cfg.group_policy == "mention"
    assert cfg.audio_format_policy.upload_direct_formats == [".wav", ".mp3", ".silk"]


def test_parse_target_supports_prefixed_and_plain_ids(qq_channel: QQChannel) -> None:
    assert qq_channel._parse_target("c2c:abc") == {"type": "c2c", "id": "abc"}
    assert qq_channel._parse_target("group:grp") == {"type": "group", "id": "grp"}
    assert qq_channel._parse_target("channel:chn") == {"type": "channel", "id": "chn"}
    assert qq_channel._parse_target("dm:user") == {"type": "dm", "id": "user"}
    assert qq_channel._parse_target("plain-openid") == {"type": "c2c", "id": "plain-openid"}


def test_reply_limit_tracker_falls_back_after_limit() -> None:
    tracker = QQReplyLimitTracker(limit=2, ttl_seconds=3600)

    assert tracker.check("m1")["allowed"] is True
    tracker.record("m1")
    assert tracker.check("m1")["allowed"] is True
    tracker.record("m1")
    result = tracker.check("m1")
    assert result["allowed"] is False
    assert result["should_fallback_to_proactive"] is True
    assert result["fallback_reason"] == "limit_exceeded"


@pytest.mark.asyncio
async def test_send_routes_group_media_upload(monkeypatch, qq_channel: QQChannel) -> None:
    calls: list[tuple[str, str, dict]] = []

    async def fake_token() -> str:
        return "token"

    async def fake_api(access_token: str, method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body or {}))
        if path.endswith("/files"):
            return {"file_info": "file-info"}
        return {"id": "msg1"}

    async def fake_b64(source: str) -> str:
        assert source == "/tmp/a.png"
        return "YmFzZTY0"

    monkeypatch.setattr(qq_channel, "_get_access_token", fake_token)
    monkeypatch.setattr(qq_channel, "_api_request", fake_api)
    monkeypatch.setattr(qq_channel, "_read_media_as_base64", fake_b64)

    msg = OutboundMessage(
        channel="qq",
        chat_id="group:group-openid",
        content="hello",
        media=["/tmp/a.png"],
        metadata={"message_id": "source-1"},
    )
    await qq_channel.send(msg)

    assert calls[0][1] == "/v2/groups/group-openid/files"
    assert calls[1][1] == "/v2/groups/group-openid/messages"
    assert calls[1][2]["media"] == {"file_info": "file-info"}
    assert calls[1][2]["content"] == "hello"


@pytest.mark.asyncio
async def test_send_falls_back_to_proactive_for_expired_reply(monkeypatch, qq_channel: QQChannel) -> None:
    calls: list[tuple[str, str, dict]] = []

    async def fake_token() -> str:
        return "token"

    async def fake_api(access_token: str, method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body or {}))
        return {"id": "msg1"}

    monkeypatch.setattr(qq_channel, "_get_access_token", fake_token)
    monkeypatch.setattr(qq_channel, "_api_request", fake_api)
    monkeypatch.setattr(
        qq_channel._reply_tracker,
        "check",
        lambda message_id: {
            "allowed": False,
            "remaining": 0,
            "should_fallback_to_proactive": True,
            "fallback_reason": "expired",
        },
    )

    await qq_channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="c2c:user-openid",
            content="hello",
            metadata={"message_id": "source-1"},
        )
    )

    assert calls == [(
        "POST",
        "/v2/users/user-openid/messages",
        {"markdown": {"content": "hello"}, "msg_type": 2},
    )]


@pytest.mark.asyncio
async def test_send_routes_dm_to_user_api(monkeypatch, qq_channel: QQChannel) -> None:
    calls: list[tuple[str, str, dict]] = []

    async def fake_token() -> str:
        return "token"

    async def fake_api(access_token: str, method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body or {}))
        return {"id": "msg1"}

    monkeypatch.setattr(qq_channel, "_get_access_token", fake_token)
    monkeypatch.setattr(qq_channel, "_api_request", fake_api)

    await qq_channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="dm:user-openid",
            content="hello dm",
            metadata={"message_id": "source-1"},
        )
    )

    assert len(calls) == 1
    assert calls[0][0] == "POST"
    assert calls[0][1] == "/v2/users/user-openid/messages"
    assert calls[0][2]["markdown"] == {"content": "hello dm"}
    assert calls[0][2]["msg_type"] == 2
    assert calls[0][2]["msg_id"] == "source-1"


@pytest.mark.asyncio
async def test_send_sends_all_media_sources(monkeypatch, qq_channel: QQChannel) -> None:
    sent: list[tuple[str, str, str | None]] = []

    async def fake_token() -> str:
        return "token"

    async def fake_send_media(*, access_token: str, target: dict[str, str], source: str, content: str, reply_to_id: str | None):
        sent.append((target["id"], source, reply_to_id))
        return {"id": f"msg-{len(sent)}"}

    monkeypatch.setattr(qq_channel, "_get_access_token", fake_token)
    monkeypatch.setattr(qq_channel, "_send_media", fake_send_media)

    await qq_channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="group:group-openid",
            content="batch",
            media=["/tmp/a.png", "/tmp/b.png"],
            metadata={"message_id": "source-1"},
        )
    )

    assert sent == [
        ("group-openid", "/tmp/a.png", "source-1"),
        ("group-openid", "/tmp/b.png", None),
    ]


@pytest.mark.asyncio
async def test_on_message_maps_group_event_and_strips_mentions(monkeypatch, qq_channel: QQChannel, tmp_path: Path) -> None:
    handled: list[dict] = []

    async def fake_handle_message(**kwargs):
        handled.append(kwargs)

    async def fake_extract_attachments(_data):
        return [{"type": "image", "path": str(tmp_path / "img.png"), "filename": "img.png"}]

    monkeypatch.setattr(qq_channel, "_handle_message", fake_handle_message)
    monkeypatch.setattr(qq_channel, "_extract_attachments", fake_extract_attachments)
    monkeypatch.setattr(qq_channel, "_transcribe_voice_attachments", lambda attachments: "")

    data = SimpleNamespace(
        id="mid-1",
        content='<@12345> 你好 <faceType=1,faceId="13",ext="eyJ0ZXh0Ijoi5ZGK54mZIn0=">',
        group_openid="group-1",
        author=SimpleNamespace(member_openid="user-1"),
    )
    await qq_channel._on_message(data, event_type="group")

    assert len(handled) == 1
    assert handled[0]["sender_id"] == "user-1"
    assert handled[0]["chat_id"] == "group:group-1"
    assert handled[0]["content"] == "你好 【表情: 告牙】"
    assert handled[0]["media"] == [str(tmp_path / "img.png")]
    assert handled[0]["metadata"]["chat_type"] == "group"
    assert handled[0]["metadata"]["mentions_me"] is True


@pytest.mark.asyncio
async def test_on_message_ignores_group_without_mention_by_default(monkeypatch, qq_channel: QQChannel) -> None:
    called = False

    async def fake_handle_message(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(qq_channel, "_handle_message", fake_handle_message)
    monkeypatch.setattr(qq_channel, "_extract_attachments", lambda _data: [])
    monkeypatch.setattr(qq_channel, "_transcribe_voice_attachments", lambda attachments: "")

    data = SimpleNamespace(
        id="mid-2",
        content="大家好",
        group_openid="group-1",
        author=SimpleNamespace(member_openid="user-1"),
    )
    await qq_channel._on_message(data, event_type="group")

    assert called is False


@pytest.mark.asyncio
async def test_on_message_allows_c2c_when_dm_policy_open(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("nanobot.channels.qq.get_data_dir", lambda: tmp_path)
    channel = QQChannel(
        QQConfig(enabled=True, app_id="app", secret="sec", allow_from=[], dm_policy="open"),
        MessageBus(),
    )
    handled: list[dict] = []

    async def fake_handle_message(**kwargs):
        handled.append(kwargs)

    monkeypatch.setattr(channel, "_handle_message", fake_handle_message)
    monkeypatch.setattr(channel, "_extract_attachments", lambda _data: [])
    monkeypatch.setattr(channel, "_transcribe_voice_attachments", lambda attachments: "")

    data = SimpleNamespace(
        id="mid-dm-open",
        content="你好",
        author=SimpleNamespace(user_openid="user-1"),
    )
    await channel._on_message(data, event_type="c2c")

    assert len(handled) == 1
    assert handled[0]["chat_id"] == "c2c:user-1"


@pytest.mark.asyncio
async def test_on_message_blocks_c2c_when_dm_policy_allowlist_and_sender_not_allowed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("nanobot.channels.qq.get_data_dir", lambda: tmp_path)
    channel = QQChannel(
        QQConfig(enabled=True, app_id="app", secret="sec", allow_from=["allowed-user"], dm_policy="allowlist"),
        MessageBus(),
    )
    called = False

    async def fake_handle_message(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(channel, "_handle_message", fake_handle_message)
    monkeypatch.setattr(channel, "_extract_attachments", lambda _data: [])
    monkeypatch.setattr(channel, "_transcribe_voice_attachments", lambda attachments: "")

    data = SimpleNamespace(
        id="mid-dm-deny",
        content="你好",
        author=SimpleNamespace(user_openid="user-1"),
    )
    await channel._on_message(data, event_type="c2c")

    assert called is False


@pytest.mark.asyncio
async def test_on_message_allows_c2c_when_dm_policy_pairing_and_sender_allowlisted(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("nanobot.channels.qq.get_data_dir", lambda: tmp_path)
    channel = QQChannel(
        QQConfig(enabled=True, app_id="app", secret="sec", allow_from=["user-1"], dm_policy="pairing"),
        MessageBus(),
    )
    handled: list[dict] = []

    async def fake_handle_message(**kwargs):
        handled.append(kwargs)

    monkeypatch.setattr(channel, "_handle_message", fake_handle_message)
    monkeypatch.setattr(channel, "_extract_attachments", lambda _data: [])
    monkeypatch.setattr(channel, "_transcribe_voice_attachments", lambda attachments: "")

    data = SimpleNamespace(
        id="mid-dm-pairing",
        content="你好",
        author=SimpleNamespace(user_openid="user-1"),
    )
    await channel._on_message(data, event_type="c2c")

    assert len(handled) == 1
    assert handled[0]["sender_id"] == "user-1"
