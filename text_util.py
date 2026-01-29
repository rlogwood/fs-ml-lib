from IPython.display import display, HTML

def styled_display(txt, font_size=20, color='orange', bold=True, italic=False):
    """
    Displays styled text using HTML in Jupyter Notebook.

    Parameters:
        txt (str): The text to be displayed.
        font_size (int): Font size in pixels (default=12).
        color (str): Color of the text (default='black').
        bold (bool): Whether the text should be bold (default=False).
        italic (bool): Whether the text should be italic (default=False).

    Returns:
        None
    """
    style = f"font-size: {font_size}px; color: {color};"

    if bold:
        txt = f"<b>{txt}</b>"
    if italic:
        txt = f"<i>{txt}</i>"

    display(HTML(f'<span style="{style}">{txt}</span>'))


def bold_text(text):
    return f"\033[1m{text}\033[0m"

def colored_text(text, color):
    colors = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "light_blue": "[1;34",
        "magenta": "35",
        "cyan": "36",
    }
    color_code = colors.get(color.lower(), "37")  # Default is white
    return f"\033[{color_code}m{text}\033[0m"

def bold_and_colored_text(text, color):
    colors = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
    }
    color_code = colors.get(color.lower(), "37")  # Default is white
    return f"\033[1;{color_code}m{text}\033[0m"

def italic_text(text):
    return f"\033[3m{text}\033[0m"

def italic_and_bold_text(text):
    return f"\033[1;3m{text}\033[0m"


def print_heading(text):
    styled_display(text, font_size=20, color='orange', bold=True, italic=False)

def print_sub_heading(text):
    styled_display(text, font_size=15, color='sandybrown', bold=True, italic=False)

def demo_text_print():
    print(colored_text("Show Red", "red"))
    print(bold_text("Show Bold"))
    print(italic_and_bold_text(colored_text("Show Combo Yellow, Italic, Bold", "yellow")))
    print(bold_and_colored_text("Show Bold and Blue", "blue"))
    styled_display("Show Styled")


if __name__ == "__main__":
    demo_text_print()
