# orjson Development Guide

This is Anthropic's custom fork of orjson with modifications for internal use.

## Key Modifications from Upstream
- Re-enabled NaN and Infinity support
- Added PyTorch serialization when `orjson.OPT_SERIALIZE_NUMPY` is set
- Handle zero-dimensional arrays
- Modified recursion depth for build compatibility

## Development Conventions

### Testing
**The pytests must stay green.** To run tests:
1. Build: `maturin build --release --strip`
2. Install: `uv pip install ~/code/orjson/target/wheels/orjson-<path-to-wheel>`
3. Run: `cd test && pytest`

### Version Management
- Use upstream version with `-post<N>` suffix (e.g., `3.10.14-post2`)
- Note: Rust prefers dash over dot for post versions

### Building Wheels
Build on three architectures:
1. **Mac laptop**: `maturin build --release --strip`
2. **Spark armbox** (arm64): Clone fresh from GitHub, build same command
3. **Manylinux** (x86_64): Use Docker (see Dockerfile in repo)

### Uploading
After building on each platform:
```bash
maturin upload --repository local target/wheels/orjson-<path-to-wheel>
```

## Important Notes
- Always verify version with: `python -c "import orjson; print(orjson.__version__)"`
- See go/fork for branch/tag management
- See go/builds and go/pin-deps for deployment process