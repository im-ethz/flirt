# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

import re

# The master toctree document.
master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'FLIRT'


# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
def find_version():
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__version__"), open('../flirt/__init__.py').read())
    return result.group(1)


version = find_version()
# The full version, including alpha/beta/rc tags.
release = version


# General information about the project.
def find_author():
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__author__"), open('../flirt/__init__.py').read())
    return str(result.group(1))


author = find_author()
copyright = "2020, " + author

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'm2r2'
]

autodoc_default_options = {
    'undoc-members': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'monokai'  # 'default', 'monokai'
# nbsphinx_codecell_lexer = 'default'  # Doesn't do anything :/

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    # 'logo_only': True,
    'display_version': False,
    'style_nav_header_background': '#343131',
}

html_logo = 'img/flirt-white.png'

html_favicon = 'img/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'flirtdoc'

# -- Options for LaTeX output ------------------------------------------
pdf_title = u'FLIRT'
author_field = u'Official Documentation'

latex_elements = {
    'sphinxsetup': r"""
        VerbatimColor={RGB}{38,50,56},
        verbatimwithframe=false,
        """
    # Background color of chunks
    # '

    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc,
     'flirt.tex',
     pdf_title,
     author_field,
     'manual'),
]

# Other
add_module_names = True  # so functions aren’t prepended with the name of the package/module
add_function_parentheses = True  # to ensure that parentheses are added to the end of all function names

# set up the types of member to check that are documented
members_to_watch = ['function', ]


def warn_undocumented_members(app, what, name, obj, options, lines):
    if what in members_to_watch and len(lines) is 0:
        # warn to terminal during build
        print("Warning: " + what + " is undocumented: " + name)
        # or modify the docstring so the rendered output is highlights the omission
        lines.append(".. Warning:: %s '%s' is undocumented" % (what, name))


def setup(app):
    app.connect('autodoc-process-docstring', warn_undocumented_members)
