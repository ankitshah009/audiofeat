# CRUSH Configuration for audiofeat

## Build/Install
pip install -e .

## Linting
pylint $(git ls-files '*.py')

## Testing
# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_spectral.py

# Run a specific test function
python -m pytest tests/test_spectral.py::test_spectral_centroid

## Code Style Guidelines

### Imports
- Use absolute imports when possible
- Group imports in three sections (standard library, third-party, local) separated by blank lines
- Import modules, not individual functions, unless there's a good reason

### Formatting
- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length is 88 characters (Black default)
- Use spaces around operators and after commas
- Use blank lines to separate functions and class definitions

### Types
- Use type hints for function parameters and return values
- Import type hints from `typing` module
- Use `torch.Tensor` for PyTorch tensors

### Naming Conventions
- Use snake_case for functions, variables, and module names
- Use PascalCase for class names
- Use UPPERCASE for constants
- Function names should be verbs or verb phrases
- Variable names should be descriptive but concise

### Error Handling
- Handle errors gracefully with appropriate try/except blocks
- Raise appropriate exceptions with descriptive messages
- Prefer using built-in exceptions when possible

### Testing
- Each feature module should have corresponding tests
- Tests should cover normal cases, edge cases, and error conditions
- Use pytest for writing tests
- Test files should be named test_*.py

### Documentation
- Use Google-style docstrings for functions and classes
- Document all parameters, return values, and exceptions
- Include example usage in module docstrings when appropriate

### Feature Implementation
- Each audio feature should be implemented as a function in its own file
- Functions should accept `torch.Tensor` inputs
- Sample rate should be passed as a parameter when relevant
- Functions should handle different audio lengths gracefully
- Return values should be `torch.Tensor` objects