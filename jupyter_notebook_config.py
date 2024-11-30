from traitlets.config import get_config

# Get the config object
c = get_config()

# PDF export settings
c.PDFExporter.latex_command = ['xelatex', '{filename}']
c.PDFExporter.template_file = 'basic'
c.PDFExporter.verbose = True

# Configure notebook to handle plotly and interactive visualizations
c.PDFExporter.exclude_input_prompt = True
c.PDFExporter.exclude_output_prompt = True

# Configure default cell width
c.PDFExporter.latex_count = 3

# Enable processing of matplotlib plots
c.PDFExporter.matplotlib_backend = 'Agg'

# Configure figure size conversion factor
c.PDFExporter.fig_width = 8.0
c.PDFExporter.fig_height = 6.0

# Add support for code wrap in latex
c.PDFExporter.wrap_code = True

# Configure code highlighting style
c.PDFExporter.highlight_style = 'tango'

# Allow for larger cells
c.PDFExporter.max_cell_length = 100000

# Configure preprocessing
c.PDFExporter.preprocessors = [
    'nbconvert.preprocessors.ExecutePreprocessor',
    'nbconvert.preprocessors.ClearOutputPreprocessor',
    'nbconvert.preprocessors.SVG2PDFPreprocessor'
]

# Configure execution settings
c.ExecutePreprocessor.timeout = 600  # seconds
c.ExecutePreprocessor.allow_errors = True
