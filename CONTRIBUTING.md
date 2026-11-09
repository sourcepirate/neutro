# Contributing to Neutro

Thanks for helping make Neutro easier to read and learn from. Small, focused
changes are welcome — especially documentation fixes, clearer examples, and
tests.

## Setup

```bash
git clone https://github.com/sourcepirate/neutro.git
cd neutro
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

## Before you open a PR

```bash
pytest -q
python scripts/check_docs.py
mkdocs build --strict
```

- `pytest -q` must pass.
- `scripts/check_docs.py` must report no broken links or missing source paths.
- `mkdocs build --strict` must build the site without warnings. Install the
  docs tooling with `pip install -e ".[docs]"`, then run `mkdocs serve` to
  preview locally.
- If you changed a source file under `neutro/layers/`, `neutro/models/`,
  `neutro/engine/`, or `neutro/autograd/`, update the matching page under
  `docs/` (see `Agents.md`).

## Documentation style

- **Start with the problem.** What does this layer/function let the reader do?
- **Show the shapes.** Write `(B, D) @ (D, U) -> (B, U)`, not just "matrix
  multiply."
- **Show expected output.** If a snippet prints something, include the output.
- **Explain "why", not just "what".** Especially for anything cached for
  `backward`, any masking trick, and any non-obvious axis.
- **Prefer a worked example over a wall of math.** Put the derivation after
  the concrete example.
- **Link to the symbol, don't copy the whole file.** Copied source drifts.
  Quote only the lines needed for the explanation.
- **Do not cite line numbers** like "see line 42"; they go stale as soon as
  anyone adds a docstring. Cite the function or class name instead.

### Learning path vs reference

The docs use four kinds of pages:

- **Tutorials** teach by doing (`docs/tutorials/`).
- **How-to guides** solve a specific task (`docs/how-to/`).
- **Explanation** deep dives explain why (`docs/layers/`, `docs/autograd/`).
- **Reference** describes the API and status (`docs/IMPLEMENTATION_STATUS.md`,
  module docstrings).

If a page starts with a story, it is probably explanation. If it starts with
"run this", it is a tutorial. Keep them separate.

## Adding a layer

Follow [API and naming conventions](docs/API_CONVENTIONS.md).

1. Create `neutro/layers/<family>/<name>.py`.
2. Inherit from `Layer`.
3. Implement `build(input_shape)` and allocate parameters in `self.params`.
4. Implement `forward(inputs, training=False)`.
5. Implement `compute_output_shape(input_shape)`.
6. Prefer `Tensor` operations so the base `Layer.backward` works. Only write
   `backward` by hand if the manual chain rule is the point.
7. Export it from `neutro/layers/__init__.py`.
8. Add `docs/layers/<family>/<name>.md` with: what it does, the math, shapes, a
   line-by-line walkthrough of `build`/`forward`/`backward`, and a runnable
   example.
9. Add a test that checks output shape and, where possible, a numerical
   gradient check.
10. Run `pytest -q` and `python scripts/check_docs.py`.

## Adding an example

- Put it in `examples/` with a numeric prefix if it belongs on the learning
  path.
- Add a module docstring with the exact run command.
- Print shapes and a concrete result so the reader knows it worked.
- Avoid `sys.path` hacks when possible; run from the repository root.
- Update `examples/README.md` and, if it is a first-stop example,
  `docs/how-to/run-examples.md`.

## Pull request checklist

- [ ] The change is focused and explained in the PR description.
- [ ] `pytest -q` passes.
- [ ] `python scripts/check_docs.py` passes.
- [ ] New public behavior has a docstring.
- [ ] New shapes are documented.
- [ ] Docs/examples that mention the changed behavior are updated.
- [ ] No unrelated files, artifacts, or generated data are committed.

## Reporting a bug

Open an issue with:

- the exact command you ran,
- the full traceback,
- your Python version and `numpy`/`scipy` versions,
- whether you are using a virtual environment.

## Code of conduct

Be kind, be patient, and assume good faith. This project exists to make hard
ideas approachable.
