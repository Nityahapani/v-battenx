# Contributing

See [doc/contributing.md](doc/contributing.md) for the full guide.

## Quick reference

```bash
# Build
g++ -std=c++17 -O2 -shared -fPIC -I. -Iinclude -I/usr/include/eigen3 \
  src/vbatten_x_impl.cc src/c_api.cc \
  -o python-package/vbatten_x/_vbatten_x.so

# Test
pytest tests/python/ tests/physics/ -q

# Format
black python-package/ --line-length 100
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
PRs require: all tests passing, CHANGELOG entry, no ABI break.
