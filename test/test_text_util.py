import text_util as tu


def test_something(capsys):
    tu.demo_text_print()
    captured = capsys.readouterr()
    expected_text = (
        "Running as a notebook:False\n"
        "Running inside in pycharm:True\n"
        "Running inside a pycharm notebook:False\n"
        "styled_display uses HTML for styling text output.\n"
        "[31mShow Red[0m\n"
        "[1mShow Bold[0m\n"
        "[1m[3m[33mShow Combo Yellow, Italic, Bold[0m[0m\n"
        "[1m[34mShow Bold and Blue[0m\n"
        "<IPython.core.display.HTML object>\n"
    )
    assert captured.out == expected_text
