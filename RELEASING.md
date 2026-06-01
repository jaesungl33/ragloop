# Releasing ragloop

ragloop publishes to PyPI as **`ragloop-agentic`** automatically when a GitHub
Release is published, using [PyPI Trusted Publishing][tp] (OIDC — no tokens
stored anywhere). This doc is the checklist for cutting a release.

## One-time setup (already done for 0.1.0)

- **PyPI Trusted Publisher** registered at <https://pypi.org/manage/account/publishing/>:
  project `ragloop-agentic`, owner `jaesungl33`, repo `ragloop`,
  workflow `publish.yml`, environment `pypi`.
- **Workflow:** `.github/workflows/publish.yml` (triggers on `release: published`).

If you ever see `invalid-publisher` in the publish logs, the Trusted Publisher
entry above is missing or doesn't match — re-check those five values.

## Cutting a release

1. **Pick the version** following [SemVer](https://semver.org): `MAJOR.MINOR.PATCH`.
   - PATCH (`0.1.1`) — bug fixes, no API change.
   - MINOR (`0.2.0`) — new features, backward compatible.
   - MAJOR (`1.0.0`) — breaking changes.

2. **Bump `version`** in `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.1"
   ```

3. **Update `CHANGELOG.md`** — add a new `## [0.1.1] - YYYY-MM-DD` section
   (Added / Changed / Fixed / Removed) and the link reference at the bottom.

4. **Verify locally** before tagging:
   ```bash
   .venv/bin/python -m pytest -q          # tests green
   .venv/bin/python -m build              # builds ragloop_agentic-<version>.{whl,tar.gz}
   .venv/bin/python -m twine check dist/* # metadata valid (pip install twine if missing)
   ```

5. **Commit + push** the bump:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release v0.1.1"
   git push origin main
   ```

6. **Create the GitHub Release** — this is what triggers the PyPI publish:
   ```bash
   gh release create v0.1.1 --title "ragloop v0.1.1" --notes-file CHANGELOG.md
   ```
   (Or write custom notes inline with `--notes "..."`.)

7. **Watch it publish and confirm:**
   ```bash
   gh run watch --repo jaesungl33/ragloop                 # wait for green
   curl -s https://pypi.org/pypi/ragloop-agentic/json \
     | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"
   ```
   It should print the new version.

## If the publish fails

- **`invalid-publisher`** — Trusted Publisher config mismatch (see one-time setup).
- **`File already exists`** — that version is already on PyPI; PyPI is immutable.
  Bump to a new version and release again (you cannot overwrite or re-upload).
- **Re-run a failed publish** (after fixing config) without a new release:
  ```bash
  gh run rerun <run-id> --repo jaesungl33/ragloop
  ```

## Notes

- The distribution name is `ragloop-agentic`; the import name stays `ragloop`.
- `Requires-Python >= 3.10`. CI tests 3.10–3.12.
- Never commit secrets or `dist/` (both are gitignored).

[tp]: https://docs.pypi.org/trusted-publishers/
