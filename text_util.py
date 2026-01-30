from enum import Enum
from IPython.display import display, HTML
import sys
import os

def is_notebook():
    """Check if running in a Jupyter notebook/IPython environment."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except (ImportError, NameError):
        pass
    return False


def is_pycharm():
    """Check if the current environment is hosted by PyCharm or other JetBrains IDEs."""
    # 1. Check for environment variables set by JetBrains IDEs
    # PyCharm sometimes uses different markers in Terminal vs. Run vs. Debug vs. Jupyter
    jetbrains_markers = ["PYCHARM", "JB_IDE", "INTELLIJ_ARGS"]
    if any(any(marker in key.upper() for marker in jetbrains_markers) for key in os.environ):
        return True

    # 2. Check for the JetBrains terminal emulator specifically
    if os.environ.get("TERMINAL_EMULATOR") == "JetBrains-JediTerm":
        return True


    # 'pydev_ipython' and 'pydevd' are injected by PyCharm for its rich integration
    #  but these modules will also be present in jupyter lab
    #  any(m in sys.modules for m in ["pydev_ipython", "pydevd"]):

    return False


def is_pycharm_notebook():
    """Check if running specifically in a PyCharm Jupyter Notebook."""
    return is_notebook() and is_pycharm()


class Color(Enum):
    """Color enum for terminal and HTML text styling."""
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    LIGHT_BLUE = "light_blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    ORANGE = "orange"
    SANDYBROWN = "sandybrown"


# ANSI color codes for terminal output
_ANSI_CODES: dict[Color, str] = {
    Color.RED: "31",
    Color.GREEN: "32",
    Color.YELLOW: "33",
    Color.BLUE: "34",
    Color.LIGHT_BLUE: "1;34",
    Color.MAGENTA: "35",
    Color.CYAN: "36",
    Color.WHITE: "37",
    Color.ORANGE: "38;5;208",  # Extended 256-color code for Orange
    Color.SANDYBROWN: "38;5;215",  # Extended 256-color code for SandyBrown
}

def styled_display_html(txt: str, font_size: int = 20, color: Color = Color.ORANGE, bold: bool = True, italic: bool = False):
    """
    Displays styled text using HTML in Jupyter Notebook.
    """
    style = f"font-size: {font_size}px; color: {color.value};"
    content = txt
    if bold:
        content = f"<b>{content}</b>"
    if italic:
        content = f"<i>{content}</i>"

    # Using a div with display:block and clear:both to ensure it starts on a new line 
    display(HTML(f'<div style="{style} display: block; clear: both;">{content}</div>'))

def styled_display_text(txt: str, color: Color = Color.ORANGE, bold: bool = True, italic: bool = False):
    if color:
        txt = colored_text(txt, color)

    if bold:
        txt = bold_text(txt)

    if italic:
        txt = italic_text(txt)

    print(txt)

# NOTE: pycharm 2025.3.2 has a bug will not render styled_display_html in a notebook
# For styled display, use styled_display_text instead if we are running in a pycharm notebook
def styled_display(txt: str, font_size: int = 20, color: Color = Color.ORANGE, bold: bool = True, italic: bool = False):
    if is_pycharm_notebook():
        styled_display_text(txt, color, bold, italic)
    else:
        styled_display_html(txt, font_size, color, bold, italic)



def error_text(text):
    return bold_and_colored_text(text, Color.RED)

def bold_text(text):
    return f"\033[1m{text}\033[0m"

def colored_text(text: str, color: Color) -> str:
    color_code = _ANSI_CODES.get(color, "37")
    return f"\033[{color_code}m{text}\033[0m"

def bold_and_colored_text(text: str, color: Color) -> str:
    color_code = _ANSI_CODES.get(color, "37")
    return f"\033[1m\033[{color_code}m{text}\033[0m"

def italic_text(text):
    return f"\033[3m{text}\033[0m"

def italic_and_bold_text(text):
    return f"\033[1m\033[3m{text}\033[0m"

def print_heading(text: str) -> None:
    styled_display(text, font_size=20, color=Color.ORANGE, bold=True, italic=False)

def print_sub_heading(text: str) -> None:
    styled_display(text, font_size=15, color=Color.SANDYBROWN, bold=True, italic=False)

def demo_text_print() -> None:
    print(f"Running as a notebook:{is_notebook()}")
    print(f"Running inside in pycharm:{is_pycharm()}")
    print(f"Running inside a pycharm notebook:{is_pycharm_notebook()}")

    if is_pycharm_notebook():
        print(f"styled_display uses ANSI escape codes for styling text output.")
    else:
        print(f"styled_display uses HTML for styling text output.")

    print(colored_text("Show Red", Color.RED))
    print(bold_text("Show Bold"))
    print(italic_and_bold_text(colored_text("Show Combo Yellow, Italic, Bold", Color.YELLOW)))
    print(bold_and_colored_text("Show Bold and Blue", Color.BLUE))
    styled_display("Show Styled")


if __name__ == "__main__":
    demo_text_print()
