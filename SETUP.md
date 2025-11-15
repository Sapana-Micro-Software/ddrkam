# Setup Instructions

## Repository Push

The code has been committed locally. To push to GitHub:

1. Create the private repository at: https://github.com/Sapana-Micro-Software/ddrkam
2. Then run: `git push -u origin main`

Or if you prefer SSH:
```bash
git remote set-url origin git@github.com:Sapana-Micro-Software/ddrkam.git
git push -u origin main
```

## Building the Framework

### C/C++ Library
```bash
make
```

### Running Tests
```bash
make test
```

### Objective-C Framework
The framework files are in the `DDRKAM/` directory. To build as an Xcode framework:
1. Open Xcode
2. Create a new Framework project
3. Add the DDRKAM source files
4. Configure build settings for macOS/VisionOS targets

## Project Structure

```
Deep-Rung-Kutta/
├── include/          # C header files
├── src/              # C implementation files
├── DDRKAM/           # Objective-C framework
├── tests/            # Test suites
├── docs/             # Documentation (LaTeX)
├── lib/              # Built libraries (after make)
├── obj/              # Object files (after make)
└── bin/              # Test binaries (after make test)
```

## All Tests Passing ✓

- Runge-Kutta 3rd order: PASS
- Hierarchical RK: PASS
- No linter errors
