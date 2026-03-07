from pathlib import Path
from unittest.mock import patch

from nanobot.agent.skills import SkillsLoader
from nanobot.cli.commands import _send_channel_message_once
from nanobot.config.schema import Config


def test_codex_agent_skill_is_listed() -> None:
    loader = SkillsLoader(Path('./test-workspace'))

    skills = loader.list_skills(filter_unavailable=False)

    assert any(skill['name'] == 'codex-agent' for skill in skills)


def test_codex_agent_skill_description_mentions_codex() -> None:
    loader = SkillsLoader(Path('./test-workspace'))

    metadata = loader.get_skill_metadata('codex-agent')

    assert metadata is not None
    assert 'Codex CLI' in metadata['description']


def test_send_channel_message_once_supports_feishu() -> None:
    config = Config()
    config.channels.feishu.app_id = 'app_id'
    config.channels.feishu.app_secret = 'app_secret'

    with patch('nanobot.config.loader.load_config', return_value=config), \
         patch('nanobot.channels.feishu.FeishuChannel.start') as mock_start, \
         patch('nanobot.channels.feishu.FeishuChannel.send') as mock_send, \
         patch('nanobot.channels.feishu.FeishuChannel.stop') as mock_stop:
        import asyncio

        asyncio.run(_send_channel_message_once('feishu', 'ou_xxx', 'hello'))

    assert mock_start.called
    assert mock_send.called
    assert mock_stop.called


def test_send_channel_message_once_uses_sender_only_for_telegram() -> None:
    config = Config()
    config.channels.telegram.token = 'bot-token'

    with patch('nanobot.config.loader.load_config', return_value=config), \
         patch('nanobot.channels.telegram.TelegramChannel.start_sender_only') as mock_start_sender_only, \
         patch('nanobot.channels.telegram.TelegramChannel.start') as mock_start, \
         patch('nanobot.channels.telegram.TelegramChannel.send') as mock_send, \
         patch('nanobot.channels.telegram.TelegramChannel.stop') as mock_stop:
        import asyncio

        asyncio.run(_send_channel_message_once('telegram', '123456', 'hello'))

    assert mock_start_sender_only.called
    assert not mock_start.called
    assert mock_send.called
    assert mock_stop.called
