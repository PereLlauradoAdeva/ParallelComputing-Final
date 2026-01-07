### Installing Python versions

Install the two versions of Python and create two *venv*, one for each version:
```
uv python install 3.14
uv python install 3.14+freethreaded
uv venv --python 3.14 .venv-3.14
uv venv --python 3.14+freethreaded .venv-3.14ft
```

### Activating the python versions

Run the sample programs alternating the GIL (3.14) and no-GIL (3.14t) versions of Python:

#### Activate GIL Python
```
source .venv-3.14/bin/activate; UV_PROJECT_ENV=.venv-3.14 uv run --active python script.py
```

#### Acitvate no-GIL Python

```
source .venv-3.14ft/bin/activate; UV_PROJECT_ENV=.venv-3.14ft uv run --active python script.py
```

### Ensuring a disabled GIL

Extra check, to avoid extensions that disable GIL:

```
export PYTHON_GIL=0
python -X gil=0 <SCRIPT.py>
```

This should not be necessary given the lack of dependencies in this project.