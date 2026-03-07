from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, _extract_post_content
from nanobot.config.schema import FeishuConfig


def test_extract_post_content_supports_post_wrapper_shape() -> None:
    payload = {
        "post": {
            "zh_cn": {
                "title": "日报",
                "content": [
                    [
                        {"tag": "text", "text": "完成"},
                        {"tag": "img", "image_key": "img_1"},
                    ]
                ],
            }
        }
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "日报 完成"
    assert image_keys == ["img_1"]


def test_extract_post_content_keeps_direct_shape_behavior() -> None:
    payload = {
        "title": "Daily",
        "content": [
            [
                {"tag": "text", "text": "report"},
                {"tag": "img", "image_key": "img_a"},
                {"tag": "img", "image_key": "img_b"},
            ]
        ],
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "Daily report"
    assert image_keys == ["img_a", "img_b"]


@pytest.mark.asyncio
async def test_send_adds_done_reaction_on_final_reply(monkeypatch) -> None:
    class _InlineLoop:
        async def run_in_executor(self, _executor, fn, *args):
            return fn(*args)

    monkeypatch.setattr("nanobot.channels.feishu.asyncio.get_running_loop", lambda: _InlineLoop())
    channel = FeishuChannel(
        FeishuConfig(enabled=True, app_id="a", app_secret="b", allow_from=["*"], done_react_emoji="DONE"),
        MessageBus(),
    )
    channel._client = object()
    channel._send_message_sync = lambda *args, **kwargs: True
    channel._add_reaction = AsyncMock()

    msg = OutboundMessage(
        channel="feishu",
        chat_id="ou_test",
        content="finished",
        metadata={"message_id": "om_123"},
    )
    await channel.send(msg)

    channel._add_reaction.assert_awaited_once_with("om_123", "DONE")


@pytest.mark.asyncio
async def test_send_skips_done_reaction_for_progress_frame(monkeypatch) -> None:
    class _InlineLoop:
        async def run_in_executor(self, _executor, fn, *args):
            return fn(*args)

    monkeypatch.setattr("nanobot.channels.feishu.asyncio.get_running_loop", lambda: _InlineLoop())
    channel = FeishuChannel(
        FeishuConfig(enabled=True, app_id="a", app_secret="b", allow_from=["*"], done_react_emoji="DONE"),
        MessageBus(),
    )
    channel._client = object()
    channel._send_message_sync = lambda *args, **kwargs: True
    channel._add_reaction = AsyncMock()

    msg = OutboundMessage(
        channel="feishu",
        chat_id="ou_test",
        content="working...",
        metadata={"message_id": "om_123", "_progress": True},
    )
    await channel.send(msg)

    channel._add_reaction.assert_not_awaited()
