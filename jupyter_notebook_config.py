# Jupyter notebook configuration file

from traitlets.config import get_config

# Get the config object
c = get_config()

# PDF export settings
c.PDFExporter.latex_command = ['xelatex', '{filename}']
c.PDFExporter.template_file = 'latex'
c.PDFExporter.latex_count = 3
c.PDFExporter.verbose = True

# Configure notebook to handle plotly and interactive visualizations
c.PDFExporter.exclude_input_prompt = True
c.PDFExporter.exclude_output_prompt = True

# Template-specific settings
c.TemplateExporter.extra_template_basedirs = ['templates']
c.TemplateExporter.template_extension = '.tplx'

# Figure settings
c.NbConvertBase.display_data_priority = [
    'image/svg+xml',
    'image/png',
    'image/jpeg',
    'text/html',
    'text/latex',
    'text/plain'
]

# Set default figure format for saving
c.InlineBackend.figure_format = 'png'
c.InlineBackend.rc = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': [8.0, 6.0],
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white'
}

# Configure preprocessing
c.PDFExporter.preprocessors = [
    'nbconvert.preprocessors.TagRemovePreprocessor',
    'nbconvert.preprocessors.RegexRemovePreprocessor',
    'nbconvert.preprocessors.ClearOutputPreprocessor',
    'nbconvert.preprocessors.ExecutePreprocessor',
    'nbconvert.preprocessors.SVG2PDFPreprocessor',
]

# Configure tag removal
c.TagRemovePreprocessor.remove_cell_tags = ('remove_cell',)
c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)

# Configure execution settings
c.ExecutePreprocessor.timeout = 600  # seconds
c.ExecutePreprocessor.allow_errors = True

# LaTeX specific settings
c.PDFExporter.latex_packages = {
    'sphinx': ['sphinx'],
    'preamble': {
        'mathspec': ['mathspec'],
        'fontenc': ['T1', 'fontenc'],
        'inputenc': ['utf8x', 'inputenc'],
        'babel': ['english', 'babel'],
        'geometry': ['letterpaper,margin=1in', 'geometry'],
        'ucs': ['ucs'],
    }
}

# Template settings
c.PDFExporter.template_paths = ['templates']
c.PDFExporter.filters = {
    'markdown2latex': 'nbconvert.filters.markdown2latex',
    'markdown2html': 'nbconvert.filters.markdown2html',
    'cite2latex': 'nbconvert.filters.cite2latex',
    'highlight2latex': 'nbconvert.filters.highlight2latex',
    'highlight2html': 'nbconvert.filters.highlight2html',
    'latex_prompt': 'nbconvert.filters.latex.escape_latex',
}

# Configure default cell width
c.PDFExporter.default_cell_width = '100%'

# Enable processing of matplotlib plots
c.PDFExporter.matplotlib_backend = 'Agg'

# Configure figure size conversion factor
c.PDFExporter.fig_width = 8.0
c.PDFExporter.fig_height = 6.0

# Add support for code wrap in latex
c.PDFExporter.wrap_code = True

# Configure code highlighting style
c.PDFExporter.highlight_style = 'tango'

# Configure header and footer
c.PDFExporter.header = ''
c.PDFExporter.footer = ''

# Set default metadata
c.PDFExporter.default_metadata = {
    'Author': 'Jupyter Notebook',
    'Title': 'Notebook Export',
    'Subject': 'Data Analysis',
    'Keywords': 'jupyter, notebook, pdf, export'
}

# Configure paths
import os

notebook_dir = os.path.expanduser('~/notebooks')
if not os.path.exists(notebook_dir):
    os.makedirs(notebook_dir)

# Server settings
c.NotebookApp.notebook_dir = notebook_dir
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True

# Security settings
c.NotebookApp.token = ''
c.NotebookApp.password = ''

# Allow for larger cells
c.PDFExporter.max_cell_length = 100000
