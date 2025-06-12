"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'Meta-Optimization Framework'
copyright = f'{datetime.now().year}, Ryan Oates, UCSB'
author = 'Ryan Oates'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension ------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Options for autosummary extension --------------------------------------

autosummary_generate = True
autosummary_imported_members = True

# -- Options for napoleon extension -----------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx extension --------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for mathjax extension ------------------------------------------

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'macros': {
            'Psi': r'\Psi',
            'alpha': r'\alpha',
            'beta': r'\beta',
            'lambda': r'\lambda',
            'Rcognitive': r'R_{\text{cognitive}}',
            'Refficiency': r'R_{\text{efficiency}}',
            'Ltotal': r'L_{\text{total}}',
            'Ltask': r'L_{\text{task}}',
        }
    }
}

# -- Custom configuration ---------------------------------------------------

# Add custom roles for mathematical notation
rst_prolog = """
.. role:: math(math)
   :language: latex

.. |Psi| replace:: :math:`\\Psi`
.. |alpha| replace:: :math:`\\alpha`
.. |beta| replace:: :math:`\\beta`
.. |lambda1| replace:: :math:`\\lambda_1`
.. |lambda2| replace:: :math:`\\lambda_2`
"""

# -- Options for MyST parser ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_substitutions = {
    "version": version,
    "project": project,
}

# -- Build customization ----------------------------------------------------

def setup(app):
    """Custom setup function for Sphinx."""
    app.add_css_file('custom.css')
    
    # Add custom directives for research documentation
    from docutils.parsers.rst import directives
    from docutils import nodes
    from sphinx.util.docutils import SphinxDirective
    
    class MathematicalFramework(SphinxDirective):
        """Custom directive for mathematical framework documentation."""
        has_content = True
        required_arguments = 1
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = {
            'equation': directives.unchanged,
            'description': directives.unchanged,
        }
        
        def run(self):
            equation = self.options.get('equation', '')
            description = self.options.get('description', '')
            
            # Create container for mathematical framework
            container = nodes.container()
            container['classes'] = ['mathematical-framework']
            
            # Add title
            title = nodes.title(text=self.arguments[0])
            container += title
            
            # Add equation if provided
            if equation:
                math_node = nodes.math_block(equation, equation)
                container += math_node
            
            # Add description
            if description:
                desc_node = nodes.paragraph(text=description)
                container += desc_node
            
            # Add content
            if self.content:
                content_node = nodes.container()
                self.state.nested_parse(self.content, self.content_offset, content_node)
                container += content_node
            
            return [container]
    
    app.add_directive('mathematical-framework', MathematicalFramework)

# -- HTML output customization ----------------------------------------------

html_title = f"{project} Documentation"
html_short_title = project
html_logo = None
html_favicon = None

# Custom sidebar
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# Footer
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Search
html_search_language = 'en'

# -- LaTeX output configuration ---------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{mathtools}
''',
}

latex_documents = [
    (master_doc, 'meta-optimization-framework.tex', 
     'Meta-Optimization Framework Documentation',
     'Ryan Oates', 'manual'),
]

# -- EPUB output configuration ----------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']