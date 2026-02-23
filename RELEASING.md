# Releasing `mice` to PyPI

This repository publishes to PyPI using GitHub Trusted Publishing (OIDC).

## One-time setup

1. In PyPI (`mice` project), add a **Pending publisher**:
   - **Provider**: GitHub
   - **Owner**: `<your-github-owner>`
   - **Repository**: `mice` (or your exact repo name)
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

2. In GitHub repo settings, ensure the workflow exists at:
   - `.github/workflows/publish.yml`

3. (Optional) Configure the GitHub environment `pypi` with protection rules.

## Release process

1. Update package version in `pyproject.toml`:
   - `[project].version = "X.Y.Z"`

2. Commit and push:

   ```bash
   git add pyproject.toml
   git commit -m "Release X.Y.Z"
   git push
   ```

3. Create and push tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. GitHub Actions will:
   - build wheel + sdist
   - run `twine check dist/*`
   - publish to PyPI via Trusted Publishing

## Manual trigger

You can also run the workflow manually from GitHub Actions (`workflow_dispatch`).

## Troubleshooting

- **"No matching publisher" on PyPI**:
  Ensure PyPI pending publisher exactly matches:
  - repository owner
  - repository name
  - workflow file name (`publish.yml`)
  - environment name (`pypi`)

- **Version already exists**:
  Bump version in `pyproject.toml` and push a new tag.

