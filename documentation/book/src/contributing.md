---
numbering:
    headings: false
---
(sec:contributing)=
# Contributing

## Installation

A [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) is **highly** recommended.
This ensures project dependencies do not conflict with the system-wide Python installation.

Create a new virtual environment folder within your project directory. It is conventional to name this folder `.venv`.

```bash
# macOS / Linux / Windows
python3 -m venv .venv
```

Activating the environment tells your shell to use the Python interpreter and pip packages located inside the `.venv` folder.

#### macOS / Linux:

```bash
source .venv/bin/activate
```

#### Windows (PowerShell):

```sh
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt), DOS

```sh
.venv\Scripts\activate.bat
```

Once activated, your terminal prompt will typically show `(.venv)`. You can now install dependencies safely.

```bash
# Install specific packages, for example, the "requests" package
pip install requests

# Or install the entire Rattlesnake development in editable mode
pip install -e .[dev]
```

Confirm that your shell is pointing to the correct Python binary.

```sh
# macOS / Linux
which python

# Windows
where python
```

The output should point to a path inside your project's `.venv` folder.

To exit the virtual environment and return to the global system stack:

```sh
deactivate
```

> **Best Practice:** Never commit the `.venv` directory to version control. Add `.venv/` to your `.gitignore` file.

See [*Install packages in a virutal environment using pip and venv*](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for more information.

Now that your have a virtual environment, you are read to populate that virtual environment with libraries specific to your virtual environment.

## Documentation

The online documentation is made with [Jupyter Book](https://jupyterbook.org).  Following are instructions for setting up a local development environment, building the book locally, and publishing the updates to the repository.

Install documentation dependencies either with `pip`

```sh
pip install "jupyter-book>=2.0.0"
```

or with [uv](https://docs.astral.sh/uv/)

```sh
uv add "jupyter-book"
```

### Local Build

```sh
cd rattlesnake-vibration-controller/documentation
```

Within this `documentation` folder, the `myst.yml` file specifies how Jupyter Book should build the documentation.  Importantly, it links to Markdown files that contain the book's content.

```sh
cd rattlesnake-vibration-controller/documentation
jupyter book build --html --strict
```

This will build the Jupyter Book documentation.

```{warning}
The foregoing command may not work behind a corporate firewall.
```

The output will be similar to this:

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

The ouput will be similar to:

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