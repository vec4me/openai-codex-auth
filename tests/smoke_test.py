import tempfile
from pathlib import Path

import openai_codex_auth


def main() -> None:
    assert openai_codex_auth.DEFAULT_CODEX_API_BASE.startswith("https://chatgpt.com/")
    assert openai_codex_auth.DEFAULT_AUTH_PATH.name == "auth.json"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            openai_codex_auth.CodexAuth(Path(tmp) / "missing.json").token()
        except openai_codex_auth.CodexAuthError as exc:
            assert "codex login" in str(exc)
        else:
            raise AssertionError("missing credential must raise CodexAuthError")


if __name__ == "__main__":
    main()
