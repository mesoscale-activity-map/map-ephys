# -*- mode: python -*-

import inspect

from IPython.core.display import HTML, display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def show_source(function):
    code = inspect.getsource(function)
    style = HtmlFormatter().get_style_defs('.highlight')
    html = highlight(code, PythonLexer(), HtmlFormatter(style='colorful'))
    display(HTML('<style>{}</style>'.format(style)))
    display(HTML(html))
