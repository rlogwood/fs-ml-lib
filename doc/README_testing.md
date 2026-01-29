# pytest Testing 101

## Capture stdout
Use pytest’s capsys (or capfd) fixture.
```python
def test_output(capsys):
    my_func_that_prints()
    captured = capsys.readouterr()
    assert "expected text" in captured.out
```
- If you need to capture stderr, use `captured.err`. 
- For code that uses low-level file descriptors (e.g., C extensions), use `capfd` instead.
- If the output is logged via logging, use `caplog` instead of capsys.
