# Releasing to PyPI

The PyPI package for this repo is:

https://pypi.org/project/openai-codex-auth/

The GitHub repo is:

https://github.com/hrbatra/openai-codex-auth

PyPI releases are immutable. Every package update needs a new version number,
even if the change is only README/docs.

`dspy-codex-auth` depends on this package. When a change here is needed by
`dspy-codex-auth`, release this package first, then bump the minimum version
in `dspy-codex-auth`'s `pyproject.toml`.

## Local Release

1. Start from a clean checkout.

```bash
cd /Users/rsika/dev/openai-codex-auth
git status --short
git pull --ff-only origin main
```

2. Make and commit the change.

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
git add .
git commit -m "Describe the change"
git push origin main
```

3. Bump the version.

```bash
uv version --bump patch
```

4. Build and verify the distributions.

```bash
rm -rf dist
uv build --no-sources
uv run --with twine python -m twine check dist/*
uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
```

5. Commit and push the version bump.

```bash
version=$(uv version --short)
git add pyproject.toml uv.lock
git commit -m "Release $version"
git push origin main
```

6. Upload to PyPI.

```bash
uv run --with twine python -m twine upload dist/*
```

7. Verify a fresh install.

```bash
tmpdir=$(mktemp -d /tmp/openai-codex-auth-pypi.XXXXXX)
cd "$tmpdir"
uv init --bare
uv add --refresh openai-codex-auth
uv run python -c "import importlib.metadata as m, openai_codex_auth; print(m.version('openai-codex-auth'))"
rm -rf "$tmpdir"
```

## GitHub Actions Publishing

`.github/workflows/publish.yml` publishes tag pushes through PyPI Trusted
Publishing once PyPI is configured for:

- PyPI project: `openai-codex-auth`
- Owner: `hrbatra`
- Repository: `openai-codex-auth`
- Workflow: `publish.yml`
- Environment: `pypi`

Then release by pushing a tag after the version bump is committed:

```bash
version=$(uv version --short)
git tag "v$version"
git push origin "v$version"
```

Do not reuse tags or PyPI versions.
