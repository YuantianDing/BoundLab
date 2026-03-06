# Contributing

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/YuantianDing/boundlab.git
   cd boundlab
   ```

2. Install in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:

   ```bash
   pytest
   ```

## Building Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings in Google style
