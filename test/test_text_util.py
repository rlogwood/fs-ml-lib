import lib.text_util as tu


def test_something(capsys):
    tu.demo_text_print()
    captured = capsys.readouterr()
    expected_text = (
        "\x1b[31mShow Red\x1b[0m\n"
        "\x1b[1mShow Bold\x1b[0m\n"
        "\x1b[1;3m\x1b[33mShow Combo Yellow, Italic, Bold\x1b[0m\x1b[0m\n"
        "\x1b[1;34mShow Bold and Blue\x1b[0m\n"
        "<IPython.core.display.HTML object>\n"
    )
    assert captured.out == expected_text


