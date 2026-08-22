---
numbering:
    headings: false
---
(sec:contributing)=
# Contributing

Users may contribute to the Rattlesnake project by [cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) or [forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the Rattlesnake [repository](https://github.com/sandialabs/rattlesnake-vibration-controller).

Direct **cloning** is reserved for authorized collaborators of the Rattlesnake repository; however, because the project is open-source, all other contributors can obtain their own copy by **forking** the repository.

## Cloning

* Cloning is a Git action.
* It creates a copy of a repository on **your physical computer**.
* It allows you to edit files and run code locally. Collaborators "clone" the original repository directly because they have permission to "push" (save) their changes directly back to the main project.
* External contributors usually "clone" *their own fork*.

## Forking

* Forking is a GitHub action.
* It is a personal copy of the entire project on **your own GitHub account**.
* It acts as a bridge for external contributors. You can make any changes you want to your fork without affecting the original project. When you are ready to share those changes, you submit a [**Pull Request**](https://github.com/sandialabs/rattlesnake-vibration-controller/pulls) to the original repository.

## Getting the Source Code

**Collaborators and team members** should **clone** the repository:

```bash
git clone git@github.com:sandialabs/rattlesnake-vibration-controller.git
```

Others should first **fork** the repository to their own GitHub account. Once forked, you can then clone your personal version of the repo to work on it locally.

## Installation

A [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) is **highly** recommended.
This ensures project dependencies do not conflict with the system-wide Python installation.

Two approaches are documented below: 

* the traditional `venv` + `pip` workflow, and 
* [`uv`](https://docs.astral.sh/uv/), a faster, modern alternative that every other command in this document (testing, linting, formatting) assumes is installed. New contributors are encouraged to use `uv` and we present that first.

### Using `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast, modern Python package and project manager, written in Rust. It replaces `pip`, `venv`, and several other tools with a single command line interface, and it is what every other section of this document (testing, linting, formatting) assumes you are using.

Compared to `pip` and `venv`, `uv`:

- Is **significantly faster** at resolving and installing dependencies, thanks to a Rust-based resolver and aggressive caching.
- **Manages the virtual environment for you.** Commands like `uv run` and `uv sync` create and use a `.venv` automatically, so there is no separate activate/deactivate step.
- Uses a **lockfile** (`uv.lock`) to guarantee that everyone — and CI — installs the exact same dependency versions, which plain `pip` does not do without extra tooling.
- Can **install and manage Python itself**, so a separate Python version manager is not required.

#### Installing `uv`

Follow the official [installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform, or use one of the quick install commands:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Setting Up the Environment

From the repository root:

```bash
uv sync --all-extras --dev
```

This creates a `.venv` automatically (if one does not already exist), installs Rattlesnake in editable mode, installs all `[dev]` dependencies, and pins everything to the versions recorded in `uv.lock`.

To run a command inside that environment without manually activating it, prefix it with `uv run`, e.g.:

```bash
uv run python -c "import rattlesnake; print(rattlesnake.__file__)"
```

This `uv run <command>` pattern is used throughout the rest of this document.

### Using `venv` and `pip` (if `uv` is not an option)

Create a new virtual environment folder within your project directory. It is conventional to name this folder `.venv`.

```bash
# macOS / Linux / Windows
python3 -m venv .venv
```

Activate the environment to tell your shell to use the Python interpreter and pip packages located inside the `.venv` folder using the command appropriate to your system:

```bash
source .venv/bin/activate  # macOS / Linux:
.venv\Scripts\Activate.ps1 # Windows (PowerShell):
.venv\Scripts\activate.bat # Windows (Command Prompt), DOS
```

Once activated, your terminal prompt will typically show `(.venv)`. You can now install dependencies safely.

```bash
# Install the entire Rattlesnake development in editable mode
pip install -e .[dev]

# Additional packages can be installed on an as-needed basis.  For example, to install
# the "requests" package
pip install requests
```

Confirm that your shell is pointing to the correct Python binary.

```bash
# macOS / Linux
which python

# Windows
where python
```

The output should point to a path inside your project's `.venv` folder.

To exit the virtual environment and return to the global system stack:

```bash
deactivate
```

> **Best Practice:** Never commit the `.venv` directory to version control. Add `.venv/` to your `.gitignore` file.

## Development

Before pushing changes, contributors should check code quality locally rather than relying solely on CI to catch problems. This means running the test suite (pytest), linting (pylint), format checking (ruff), code coverage, and confirming that the Jupyter Book documentation still builds — all on your own machine. Catching issues locally is faster than waiting on a CI run, and it keeps the CI pipeline green for everyone else.

### Test

Rattlesnake uses [pytest](https://docs.pytest.org/en/stable/) for its test suite. Tests are split into groups based on how long they take to run:

```bash
tests/
tests/short/
tests/long/
```

`tests/long` contains slower, heavier tests (e.g., full qualification runs), while `tests/short` and the top-level `tests/` files run quickly and are meant to give fast feedback during development.

To run the **full test suite** locally with `uv`:

```bash
uv run pytest tests/
```

To run only the **fast tests**, matching CI's default scope:

```bash
uv run pytest tests/short
```

To run a **single test file**, e.g.,

```bash
uv run pytest tests/short/test_environment_manager.py
```

To also collect a **coverage report** for the entire repository while testing:

```bash
uv run pytest tests/ --cov=rattlesnake --cov-report=term-missing
```

To collect a coverage report for a **single test file**, e.g.,

```bash
uv run pytest tests/short/test_environment_manager.py --cov=rattlesnake.environment_manager --cov-report=term-missing
```

```{note}
This mirrors the scope CI uses: pushes and pull requests targeting `main` or
`dev` run the full suite (`tests`, including `tests/long`); everything else
runs just `tests/short` by default, unless `[all tests]` appears in the
commit message, which forces the full suite on any branch. For example:

    (dev-cicd) > git commit -m 'test feature foo with [all tests]'
```

### Lint

Rattlesnake uses **pylint** for static code analysis and **ruff** for code formatting. Both tools are configured to work together with `uv` to catch issues early and maintain consistent code quality.

#### Pylint

To lint the source code locally, use `uv` to run pylint:

```bash
# Lint a specific file, e.g.,
uv run pylint src/rattlesnake/utilities.py

# Lint the entire source directory
uv run pylint src/rattlesnake

# Lint and show only line-too-long warnings
uv run pylint src/rattlesnake --disable=all --enable=line-too-long
```

#### Ruff Format

```bash
# Auto-format a specific file, e.g.,
uv run ruff format src/rattlesnake/utilities.py

# Auto-format the entire source directory
uv run ruff format src/rattlesnake/
```

#### Checking Formatting Without Modifying Files

Before pushing, you can check whether any files are out of compliance with the project's formatting rules without actually rewriting them by adding the `--check` flag:

```bash
uv run ruff format --check src/rattlesnake
```

This is the same command the `lint` job in `ci.yml` runs on every push. It is intentionally **non-blocking** in CI — a formatting drift surfaces as a warning annotation on the workflow run rather than failing the job, so it will not block a merge. If it reports files that need formatting, run `uv run ruff format src/rattlesnake` locally (as shown above) to fix them before committing.

### Documentation

The online documentation is made with [Jupyter Book](https://jupyterbook.org). Start from the

```bash
rattlesnake-vibration-controller/documentation
```

folder.

#### Local Build

Within the `documentation` folder, the `myst.yml` file specifies how Jupyter Book should build the documentation.  Importantly, it links to Markdown files that contain the book's content.

```sh
jupyter book build --html --strict
```

This will build the Jupyter Book documentation.

```{warning}
The foregoing command may not work behind a corporate firewall, in which case, the simpler `jupyter book build` command should still work.
```

The output will be similar to:

```sh
building myst-cli session with API URL: https://api.mystmd.org
(node:93011) Warning: `--localstorage-file` was provided without a valid path
(Use `node --trace-warnings ...` to show where the warning was created)
🌎 Building Jupyter Book (via myst) site
📖 Built book/src/_generated/random_vibration_run_doc.md in 64 ms.
📖 Built book/src/chapter_13.md in 124 ms.
📖 Built book/src/notation.md in 117 ms.
📖 Built book/src/contributing.md in 117 ms.
<--(snip)-->
📚 Built 32 pages for project in 813 ms.
```

To view the Jupyter Book output locally:

```sh
jupyter book start
```

The output will be similar to:

```sh
📚 Built 32 pages for project in 974 ms.
<--(snip)-->
🔌 Server started on port 3000!  🥳 🎉

        👉  http://localhost:3000  👈
```

In a local web browser, navigate to the web address indicated above.

#### Bibliography

The documentation uses the `myst-nb` and standard MyST bibliography support.

1. Prepare your bibliography file:

References are stored in `documentation/book/bibliography.bib` using the standard BibLaTeX (`.bib`) format. Populate the file with references, e.g., 

```bibtex
@book{knuth1986computer,
  title={The Computer Science of TeX and Metafont: An Inaugural Lecture},
  author={Knuth, Donald E},
  year={1986},
  publisher={American Mathematical Society}
}
```

2. Configure `myst.yml`

The bibliography is configured in `documentation/myst.yml` under the `project.bibliography` section:

```yaml
project:
  bibliography:
    - book/bibliography.bib
```

3. Add in-text citations

In a markdown file, use the `cite` role to reference an entry by its key:

`{cite}` `knuth1986computer`

4. Build the book

Run the `jupyter book build` command from the `documentation` directory. The build system will automatically process the citations and generate the bibliography.

```sh
cd documentation
jupyter book build
```

## Continuous Integration/Continuous Deployment (CI/CD)

The CI/CD pipeline comprises two GitHub Actions workflows, `ci.yml` and `release.yml`, described in detail below.

### Synopsis

`ci.yml` — Continuous Integration

Triggered by:

* A push to *any* branch
* A pull request targeting `main`/`dev`
* Manual `workflow_dispatch`
  * *Example:* re-running CI on demand from the GitHub Actions UI, such as forcing the full `pytest_matrix` via `test_level=full` on a branch that wouldn't otherwise trigger it.
* `workflow_call`
  * *Example:* `release.yml` invokes `ci.yml` as its `test` job through the `workflow_call` mechanism.

Six jobs, four of which share a gate:

* **`pytest_matrix`, `lint`, `coverage`** — run when `code_changed == 'true'` **or** the branch (or PR base branch) is `main`/`dev`.
* **`docs_jupyter_book`** — runs when `docs_changed == 'true'` **or** the branch (or PR base branch) is `main`/`dev`.
* **`deploy`** — runs on `main`/`dev` regardless of whether `code_changed` is `true` or `false`, and regardless of whether `docs_changed` is `true` or `false`.

1. **changes**
   * Uses [dorny/paths-filter](https://github.com/dorny/paths-filter) to detect whether docs and/or code files changed. Sets the job outputs `docs_changed` and `code_changed` (each `true`/`false`), which downstream jobs use to streamline the CI process by skipping unnecessary jobs.
2. **pytest_matrix**
   * Runs tests on all combinations of [macOS, Ubuntu, Windows] × [3.11, 3.12] of Python using `pip install .[dev]`.  PyQt wheel compatibility requires use of `pip` instead of `uv`. Test scope is adaptive:
     * Default: `tests/short`
     * Full suite triggered by:
       * commit message containing `[all tests]`, or
       * manual dispatch with `test_level=full`, or
       * branch is `main` or `dev`
3. **lint**
   * Runs `uv run ruff format --check src/rattlesnake` first. This step is **non-blocking** — a
  formatting drift surfaces as a `::warning::` annotation on the workflow run instead of failing
  the job, so it never turns the workflow red.
   * Then runs `pylint src/rattlesnake` via `uv`, captures output, then calls `report_lint.py` to
  generate an HTML lint report artifact.
   * Both checks share one job (checkout + `uv sync`) instead of running in separate jobs, saving
  a redundant environment setup per workflow run.
4. **coverage**
   * Runs `pytest --cov` via `uv` with the same adaptive test scope, then calls `report_coverage.py` to generate an HTML coverage report artifact.
5. **docs_jupyter_book**
   * Updates `myst.yml` metadata via `report_jupyter_book.py`, then builds the Jupyter Book.
6. **deploy**
   * Assembles all artifacts into a `pages/` tree, generates the dashboard (`report_dashboard.py`), creates SVG badges, then clones the `gh-pages` branch, replaces only the current branch's subdirectory (`main/` or `dev/`), and pushes the result back with plain `git` (not `peaceiris/actions-gh-pages` or `actions/deploy-pages` — see the comments in `ci.yml` for why both were rejected).

`release.yml` — Release Pipeline

Triggered by a `v*` **tag push**, but never a **branch push** (not even a branch push to `main` or `dev`). Once triggered, two conditions (both checked in `validate_tag`) decide where, if anywhere, the release publishes to:

* **Branch:** the tag must be reachable from `main` or `dev`. A tag on any branch that is not `main`/`dev` fails the `validate_tag` job; a failed `validate_tag` job prevents any releases to TestPyPI or PyPI.
* **Version string:** a prerelease version (`a`/`b`/`rc`/`.dev` segments) publishes to **TestPyPI**; a stable or `.post` version publishes to **PyPI**.
  * The `main` branch can publish to either TestPyPI or PyPI. 
  * The `dev` branch can publish to either TestPyPI or PyPI.

Six sequential jobs:

1. **validate_tag**
   * Verifies the tag was created on the `main` or `dev` branch, that it conforms to PEP 440, and that it is strictly newer than all existing tags.
   * Computes an `is_prerelease` job output using `packaging.version.Version(...).is_prerelease`. This is the **single source of truth** consumed by every downstream job that needs to distinguish a prerelease from a production release — nothing downstream re-derives it with its own tag matching.
2. **test**
   * Calls `ci.yml` as a reusable workflow (`workflow_call`).
3. **build**
   * Runs `uv build` and generates a Supply chain Levels for Software Artifacts (SLSA, aka "salsa") provenance attestation for the dist artifacts.
4. **github-release**
   * Creates a GitHub Release with auto-generated notes and attaches the `dist` files.
   * `prerelease:` is set directly from `validate_tag`'s `is_prerelease` output.
5. **publish_testpypi** / **publish_pypi**
   * Two separate jobs, mutually exclusive via `if: needs.validate_tag.outputs.is_prerelease == 'true'` / `'false'`. 
     * Each has a hardcoded `environment:` (`testpypi` / `pypi`) and hardcoded publish target — no ternary expression to read or evaluate.
     * In the Actions UI this shows as one job succeeding and the other skipped, so which registry a run published to is visible at a glance from the job list alone, and each job's last step also writes an explicit one-line status (e.g., "📦 Published `v1.2.3` to **production PyPI**") to the run's Summary tab.
   * Splitting into two jobs (rather than two steps in one job) is required because GitHub Actions environments (including the `pypi` environment's required-reviewers approval gate) are configured per-job, not per-step.

### Efficiency

When a user pushes to the repository, the `changes` job in the main workflow
determines the **types** of the files that were committed.  The job determines
if only `docs` (documentation) files changed, only `code` (source code, project code)
files changed, or both.  

#### Updates to `docs` only

For example, upon pushing updates only to a markdown file (i.e., `*.md`),
the job makes this determination:

```bash
📂 Docs changed: true
📂 Docs files: documentation/book/src/contributing.md
💻 Code changed: false
💻 Code files:
```

In this scenario, only jobs that rely on updates to documentation file
types are run.  This avoids running unnecessary tests that *don't* rely on documentation updates.

:::{figure} figures/cicd_doc_change_only.svg
:name: fig-docs-only
:align: center

CI/CD workflow execution for documentation-only changes.
:::

#### Updates to `code` only

For example, upon pushing updates to source code (e.g., `*.py`),
the job makes this determination:

```bash
📂 Docs changed: false
📂 Docs files: 
💻 Code changed: true
💻 Code files: src/rattlesnake/cicd/report_dashboard.py src/rattlesnake/cicd/report_jupyter_book.py src/rattlesnake/cicd/report_lint.py tests/test_cicd_utilities.py
```

Only the `pytest_matrix`, `lint`, and `coverage` jobs will be run.  The `docs_jupyter_book` and `deploy` jobs will be skipped.

#### All test

Regardless of the file type, if either the `main` or the `dev` branch is the target
of an update, *all tests* are run, for example,

:::{figure} figures/cicd_all_jobs.svg
:name: fig-all-jobs
:align: center

Full suite of CI/CD jobs triggered for main or dev branch updates.
:::

Running the full suite is significantly more time-consuming than executing only the specific tests relevant to the modified files.

#### Matrix scope

The `pytest_matrix` job runs across combinations of operating systems and Python versions. The scope is adaptive:

- **Feature branches** — runs only `ubuntu-latest` × `3.12` (1 runner). This keeps per-push feedback fast.
- **`main` and `dev` branches** — runs the full matrix: `macos-latest`, `ubuntu-latest`, and `windows-latest` × `3.11` and `3.12` (6 runners). Full cross-platform coverage is enforced before anything reaches a release branch.

This means OS-specific or Python-version-specific bugs are caught on `main`/`dev` before a release, without slowing down every feature branch push.

### Preflight

The `preflight` command is a local CI/CD readiness check. It mirrors the checks that GitHub Actions would run on a push, allowing developers to catch errors before they reach the pipeline.

```sh
uv run preflight
```

largely automates the manual steps listed above in the [Development](#development) section.

#### Modes and options

By default, `preflight` matches CI's scope on non-`main`/`dev` branches: ruff format check and full pylint on `src/rattlesnake/`. When pytest is re-enabled, the default scope will also run `tests/ --ignore=tests/long`; use `--all-tests` to include `tests/long/` (matching CI on `main`/`dev`).

```{note}
pytest steps are currently skipped pending resolution of a test hang in `test_acquisition.py` (see TODO in `src/rattlesnake/preflight.py`). Only ruff format check and pylint run until that issue is resolved with Dan.
```

option | description
--- | ---
*(none)* | Default scope: ruff format check + pylint (+ pytest `tests/ --ignore=tests/long` when re-enabled)
`--all-tests` | Full suite including `tests/long/`; matches CI on `main`/`dev`
`--coverage` | Adds `--cov=rattlesnake --cov-report=term-missing` to the pytest run (no effect while pytest is disabled)
`--tag TAG` | Validates `TAG` before pushing a release: checks current branch is `main` or `dev`, that the tag conforms to PEP 440, and that it is strictly newer than all existing tags. Runs before lint and tests.
`--docs` | Builds the Jupyter Book with `--strict`; matches the `docs_jupyter_book` CI job. Requires network access to `api.mystmd.org`.
`--no-sync` | Skips `uv sync` (useful when offline or behind a firewall)
`--skip-network-check` | Skips the initial PyPI connectivity check
`--force` | Continues even if the network or sync checks fail

#### Examples

```sh
uv run preflight                            # default scope
uv run preflight --all-tests                # full suite
uv run preflight --coverage                 # default scope + coverage report
uv run preflight --all-tests --coverage     # full suite + coverage report
uv run preflight --tag v1.0.0rc1            # validate tag, then default scope
uv run preflight --tag v1.0.0 --all-tests   # validate tag, then full suite
uv run preflight --docs                     # build Jupyter Book
uv run preflight --no-sync                  # skip dependency sync
uv run preflight --force                    # continue past network/sync failures
uv run preflight --skip-network-check       # skip initial PyPI connectivity check
```

### Trusted Publishing

In `release.yml` we have removed the manual `-p ${{ secrets.PYPI_TOKEN }}`.  The industry standard is now [**Trusted Publishing**](https://docs.pypi.org/trusted-publishers/) (also called OpenID Connect or OIDC).  You configure this in your PyPI project settings once, and GitHub Actions authenticates securely without you needing to store and rotate secrets.

> OpenID Connect (OIDC) provides a flexible, credential-free mechanism for delegating publishing authority for a PyPI package to a trusted third party service, like GitHub Actions.  PyPI users and projects can use trusted publishers to automate their release processes, without needing to use API tokens or passwords.

To configure Trusted Publishing, you tell PyPI, "Trust any code from this specific GitHub repository and workflow."  This removes the need to manage long-lived API tokens or passwords in your secrets.

Steps:

* In `release.yml`, `publish_testpypi` and `publish_pypi` are two separate jobs, each hardcoded to its own environment (`testpypi` / `pypi` respectively). Which one actually runs is decided once, upstream, in `validate_tag`'s `is_prerelease` output:

```yaml
  publish_testpypi:
    environment: testpypi
    if: needs.validate_tag.outputs.is_prerelease == 'true'

  publish_pypi:
    environment: pypi
    if: needs.validate_tag.outputs.is_prerelease == 'false'
```

The GitHub repository itself must have both a `pypi` and a `testpypi` environment:

On the GitHub repo:

* Click on the **Settings** tab (usually the last tab on the right in the top navigation bar).
* On the left-hand sidebar, look for the **Environments** link (it's under the "Code and automation" section).
  * If the environment doesn't exist yet:
    * Click the **New environment** button.
    * Name the environment `pypi` (and then make a second item called `testpypi`) and click **Configure environment**.
  * If it does exist but is named differently, you can click on it to rename it or delete it and create a new one.
* For a basic setup using Trusted Publishing, you don't actually need to add any secrets or configuration on this page. Just having the environment named testpypi exist is enough to link it to your workflow.
* Optionally, we add the following protections:
  * Under the **Deployment branches and tags**, under the **No Restriction** button, select **Selected branches and tags**.
  * Click **Add deployment branch or tag rule**.
  * Select **Ref type: Tag**.
  * Set the **Name Pattern:** to `v*`.  This ensures that *only* version tags can ever use this environment, adding a layer of security.

Finally, the PyPI (respectively, Test PyPI) site needs to be configured.

* Log into your [PyPI](https://pypi.org) (or [Test PyPI](https://test.pypi.org)) account
* Go to your project's **Manage** page (or your account's **Publishing** settings if you are setting it up for the first time)
* Look for the **Publishing** tab
* Click **Add new publisher**
* Select **GitHub** as the source
* Enter the following details:
  * Owner: sandialabs
  * Repository name: rattlesnake-vibration-controller
  * Workflow name: `release.yml` (this must match your filename in your `.github/workflows/` directory)
  * Environment name: You can leave this blank or name it `pypi` (if you use it in your YAML).  We used 
    * `pypi` for live publishing to the PyPI site, and
    * `testpypi` for test publishing to the TestPyPI site.
  * Click the **Add** button

### Tags and Semantic Versioning

We follow [PEP 440](https://peps.python.org/pep-0440/) (the Python standard for versioning), which requires version strings to follow this specific structure:

```bash
N.N.N[{a|b|rc}N][.postN][.devN]
```

The `validate_tag` job in `release.yml` enforces that a tag can be added only when the
branch is `main` or `dev`, that the tag follows PEP 440, and that the version is
strictly newer than all existing tags.

#### Example Tags

Following are **prerelease tags**:

tag | description
--- | ---
`v1.1.0a1` | The first **alpha** for version 1.1.0
`v1.1.0b2` | The second **beta** for version 1.1.0
`v1.1.0rc1` | The first **release candidate** for version 1.1.0

A **release candidate** is made during the final testing stage before a full release.

Following are **stable release tags** (e.g., starting from the `v1.0.0` release):

tag | description
--- | ----
`v1.0.1` | **Patch Release**: Backwards-compatible bug fixes
`v1.1.0` | **Minor Release**: New features that are backwards-compatible
`v2.0.0` | **Major Release**: Significant changes or breaking API updates

Following are **Development** and **Post-Release** tags:

tag | description
--- | ---
`v1.1.0.dev1` | A version currently under development
`v1.0.0.post1` | Fix a minor error in the release process, such as a fix of a typo in the documentation, without changing the code

### Release on Tag

Following is an example of creating a release with a tag.

#### Create a Prerelease

To create a prerelease on TestPyPI:

The tag must be pushed from `main` or `dev` (see [Synopsis](#synopsis)); the convention (but not requirement) is to tag from `dev`, so that release-candidate tags can be tested before merging to `main`.

* On the `dev` branch, create a tag and then push, e.g.,

```sh
# Ensure you are on the dev branch
git checkout dev
git pull

# View existing tags, if any
git tag

# Create the new tag, e.g.,
git tag -a v1.0.0rc1 -m "Test of prerelease version 1.0.0, release candidate 1"

# Push the tag to GitHub
git push origin v1.0.0rc1
```

#### Create a Release

To create a release on PyPI:

The tag must be pushed from `main` or `dev` (see [Synopsis](#synopsis)); the convention (but not requirement) is to cut production PyPI releases from `main`, after merging in the validated work from `dev`, so that `main` reliably reflects what has actually shipped.

* Merge the `dev` branch into the `main` branch.
* On the `main` branch, create a tag using `git tag` and push it to the `main` branch on GitHub, e.g.,

```sh
# Ensure you are on the main branch
git checkout main
git pull

# View existing tags, if any
git tag

# Create the new tag, e.g.,
git tag -a v1.0.0 -m "Release version 1.0.0"

# On the main branch, push the tag to GitHub
git push origin v1.0.0
```

### Manual Approval Gate

By default, a tag push triggers the full release pipeline automatically — including the final publish to PyPI — with no human checkpoint. The **manual approval gate** pauses the `publish_pypi` job and requires a named reviewer to explicitly approve before the package is uploaded to PyPI.

This is an industry-standard safeguard for production releases. It gives a release manager a final opportunity to confirm that the correct tag is being published, the changelog looks right, and no last-minute issues have been flagged.

The approval gate applies only to the production `pypi` environment. The `testpypi` environment (used for prereleases) does not require approval, since prereleases are low-risk by design.

#### Setup (GitHub Settings UI)

No changes to `release.yml` are required. `publish_pypi` is hardcoded to `environment: pypi` and `publish_testpypi` is hardcoded to `environment: testpypi` — GitHub uses whichever environment name the running job declares as the hook to enforce the approval rule, so it only ever applies to `publish_pypi`.

1. Navigate to the repository on GitHub.
2. Click the **Settings** tab.
3. In the left sidebar under **Code and automation**, click **Environments**.
4. Click on the **pypi** environment.
5. Under **Deployment protection rules**, check the box next to **Required reviewers**.
6. In the text field that appears, type the GitHub username(s) or team name(s) who are authorized to approve a PyPI release. Add up to 6 reviewers.
7. Click **Save protection rules**.

When a release tag is pushed, the pipeline will run `validate_tag`, `test`, `build`, and `github-release` automatically. For a stable/`.post` tag, the `publish_pypi` job will then pause with status **Waiting** (`publish_testpypi` is skipped, since `is_prerelease` is false). The designated reviewer(s) will receive a GitHub notification and must click **Review deployments → Approve and deploy** before the package is uploaded to PyPI.

If no reviewer approves within 30 days, the deployment times out and must be re-triggered.
