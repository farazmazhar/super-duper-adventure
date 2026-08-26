"""Moderation pre-check tests: mock client, toggle, fail-open."""

from __future__ import annotations

from apps.guardrails.moderation import ModerationClient, check_moderation


class FakeChatCompletions:
    """Mimics openai.resources.chat.completions.Completions.create()."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.last_messages = None

    def create(self, model: str, messages: list, max_tokens: int, temperature: float):
        self.last_messages = messages
        return FakeResponse(self.label)


class FakeCompletions:
    def __init__(self, label: str) -> None:
        self.completions = FakeChatCompletions(label)


class FakeChat:
    def __init__(self, label: str) -> None:
        self.completions = FakeChatCompletions(label)


class FakeResponse:
    def __init__(self, label: str) -> None:
        self.choices = [FakeChoice(label)]


class FakeChoice:
    def __init__(self, label: str) -> None:
        self.message = FakeMessage(label)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeClient:
    """Drop-in OpenAI client with .chat.completions.create."""

    def __init__(self, label: str = "safe") -> None:
        self.chat = FakeChat(label)


def make_client(label: str = "safe") -> ModerationClient:
    return ModerationClient(
        api_key="k", base_url="x", model="m",
        client_factory=lambda: FakeClient(label),
    )


def test_moderation_pass() -> None:
    client = make_client("safe")
    r = check_moderation("Which customers are churning?", client=client)
    assert r.severity == "pass"
    assert r.passed


def test_moderation_block_on_unsafe() -> None:
    client = make_client("unsafe")
    r = check_moderation("I will hurt someone", client=client)
    assert r.severity == "block"
    assert not r.passed


def test_moderation_disabled() -> None:
    client = make_client("unsafe")
    r = check_moderation("anything", client=client, enabled=False)
    assert r.severity == "pass"
    assert "disabled" in r.message


def test_moderation_fail_open_on_error() -> None:
    class ExplodingClient:
        def __init__(self) -> None:
            class ExplodingCompletions:
                def create(self, **kwargs):
                    raise RuntimeError("provider down")

            class ExplodingChat:
                completions = ExplodingCompletions()

            self.chat = ExplodingChat()

    client = ModerationClient(api_key="k", base_url="x", model="m", client_factory=lambda: ExplodingClient())
    r = check_moderation("Which customer needs attention?", client=client)
    assert r.severity == "pass"  # fail-open
    assert r.passed
    assert "skipped" in r.message
