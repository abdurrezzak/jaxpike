# Releasing

## Distribution

jaxpike is published on [PyPI](https://pypi.org/project/jaxpike/) as `jaxpike`, installable
with `pip install jaxpike`. No third-party registry, mirror, or hosting account is required.

**A domain is not needed.** Documentation is served from GitHub Pages at
<https://abdurrezzak.github.io/jaxpike/>, which is free for public repositories and is
configured in `.github/workflows/docs.yml`. A custom domain would be presentation only: add a
`CNAME` and point a DNS record at Pages, and nothing in the package or the build changes. Note
that GitHub Pages is unavailable for private repositories on the free plan, so the docs site
requires the repository to be public.

Uploads use [trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI verifies the
identity of the release workflow directly, so no API token is stored in the repository or in
CI. The publisher is registered against project `jaxpike`, owner `abdurrezzak`, repository
`jaxpike`, workflow `release.yml`, environment `pypi`.

## Cutting a release

1. Update `version` in `pyproject.toml` and `__version__` in `src/jaxpike/__init__.py`. They
   must match; nothing enforces it automatically.
2. Add a section to `CHANGELOG.md`.
3. Verify locally:

   ```bash
   .venv/bin/pytest
   .venv/bin/ruff check . && .venv/bin/ruff format --check .
   uv build && uvx twine check dist/*
   ```

4. Commit, tag `vX.Y.Z`, and push both.
5. Create a GitHub release for the tag. Publishing the release triggers `release.yml`, which
   builds, checks and uploads to PyPI.

## Things that cannot be fixed after the fact

**PyPI metadata is immutable per release.** A wrong `Documentation` URL, a broken README link or
a missing classifier requires a new version — releases can be yanked but never edited. Read
`[project.urls]` once more before tagging.

**The README is rendered standalone on PyPI**, so every link and image in it must be an absolute
URL. Relative paths resolve against `pypi.org` and 404, and relative images do not render at
all.

**Uploading publishes the source permanently.** The sdist contains the full source tree
regardless of repository visibility.
