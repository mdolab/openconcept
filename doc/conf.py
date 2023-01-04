# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import openconcept
import subprocess

from sphinx_mdolab_theme.config import *

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
# sphinx build needs to be able to find the openmdao embed_code plugin
# so we add it to the path
this_directory = os.path.abspath(os.path.dirname(__file__))


def generate_src_docs(dir, top, packages):
    """
    generate_src_docs is a modification of an OpenMDAO source doc generator
    the main difference is it doesn't include all the inherited public API elements
    """
    index_top = """:orphan:

.. _source_documentation:

********************
Source Docs
********************

.. toctree::
   :titlesonly:
   :maxdepth: 1

"""
    package_top = """
.. toctree::
    :maxdepth: 1

"""

    ref_sheet_bottom = """
   :members:
   :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__
   :show-inheritance:

.. toctree::
   :maxdepth: 1
"""

    docs_dir = os.path.dirname(dir)

    doc_dir = os.path.join(docs_dir, "_srcdocs")
    if os.path.isdir(doc_dir):
        import shutil

        shutil.rmtree(doc_dir)

    if not os.path.isdir(doc_dir):
        os.mkdir(doc_dir)

    packages_dir = os.path.join(doc_dir, "packages")
    if not os.path.isdir(packages_dir):
        os.mkdir(packages_dir)

    # look for directories in the package level, one up from docs
    # auto-generate the top-level index.rst file for _srcdocs, based on
    # packages:

    # to improve the order that the user sees in the source docs, put
    # the important packages in this list explicitly. Any new ones that
    # get added will show up at the end.

    # begin writing the '_srcdocs/index.rst' file at mid  level.
    index_filename = os.path.join(doc_dir, "index.rst")
    index = open(index_filename, "w")
    index.write(index_top)

    # auto-generate package header files (e.g. 'openconcept.analysis.rst')
    for package in packages:
        # a package is e.g. openmdao.core, that contains source files
        # a sub_package, is a src file, e.g. openmdao.core.component
        sub_packages = []
        package_filename = os.path.join(packages_dir, "openconcept." + package + ".rst")
        package_name = "openconcept." + package

        # the sub_listing is going into each package dir and listing what's in it
        for sub_listing in sorted(os.listdir(os.path.join(top, package.replace(".", "/")))):
            # don't want to catalog files twice, nor use init files nor test dir
            if (os.path.isdir(sub_listing) and sub_listing != "tests") or (
                sub_listing.endswith(".py") and not sub_listing.startswith("_")
            ):
                # just want the name of e.g. dataxfer not dataxfer.py
                sub_packages.append(sub_listing.rsplit(".")[0])

        if len(sub_packages) > 0:
            # continue to write in the top-level index file.
            # only document non-empty packages -- to avoid errors
            # (e.g. at time of writing, doegenerators, drivers, are empty dirs)

            # specifically don't use os.path.join here.  Even windows wants the
            # stuff in the file to have fwd slashes.
            index.write("   packages/openconcept." + package + "\n")

            # make subpkg directory (e.g. _srcdocs/packages/core) for ref sheets
            package_dir = os.path.join(packages_dir, package)
            os.mkdir(package_dir)

            # create/write a package index file: (e.g. "_srcdocs/packages/openmdao.core.rst")
            package_file = open(package_filename, "w")
            package_file.write(package_name + "\n")
            package_file.write("-" * len(package_name) + "\n")
            package_file.write(package_top)

            for sub_package in sub_packages:
                SKIP_SUBPACKAGES = ["__pycache__"]
                # this line writes subpackage name e.g. "core/component.py"
                # into the corresponding package index file (e.g. "openmdao.core.rst")
                if sub_package not in SKIP_SUBPACKAGES:
                    # specifically don't use os.path.join here.  Even windows wants the
                    # stuff in the file to have fwd slashes.
                    package_file.write("    " + package + "/" + sub_package + "\n")

                    # creates and writes out one reference sheet (e.g. core/component.rst)
                    ref_sheet_filename = os.path.join(package_dir, sub_package + ".rst")
                    ref_sheet = open(ref_sheet_filename, "w")

                    # get the meat of the ref sheet code done
                    filename = sub_package + ".py"
                    ref_sheet.write(".. index:: " + filename + "\n\n")
                    ref_sheet.write(".. _" + package_name + "." + filename + ":\n\n")
                    ref_sheet.write(filename + "\n")
                    ref_sheet.write("-" * len(filename) + "\n\n")
                    ref_sheet.write(".. automodule:: " + package_name + "." + sub_package)

                    # finish and close each reference sheet.
                    ref_sheet.write(ref_sheet_bottom)
                    ref_sheet.close()

            # finish and close each package file
            package_file.close()

    # finish and close top-level index file
    index.close()


def run_file_move_result(file_name, output_files, destination_files, optional_cl_args=[]):
    """
    Run a file (as a subprocess) that produces output file(s) of interest.
    This function then moves the file(s) to a specified location.

    For example, a file may produce a figure that is used in the docs.
    This function can be used to automatically generate the figure in the RTD build
    and move it to a specific location in the RTD build.

    Note that the file is run from the openconcept/doc directory and all relative paths
    are relative to this directory. If the output file name is defined in the script
    using a relative path remember to take it into account.

    Parameters
    ----------
    file_name : str
        Python file to be run
    output_files : list of str
        Output files produced by running file_name
    destination_files : list of str
        Destination paths/file names to move output_file to (must be same length as output_files)
    optional_cl_args : list of str
        Optional command line arguments to add when file_name is run by Python
    """
    # Error check
    if len(output_files) != len(destination_files):
        raise ValueError("The number of output files must be the same as destination file paths")

    # Run the file
    subprocess.run(["python", file_name] + optional_cl_args)

    # Move the files
    for output_file, destination_file in zip(output_files, destination_files):
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        os.replace(output_file, destination_file)


# Patch the Napoleon parser to find Inputs, Outputs, and Options headings in docstrings

from sphinx.ext.napoleon.docstring import NumpyDocstring


def parse_inputs_section(self, section):
    return self._format_fields("Inputs", self._consume_fields())


NumpyDocstring._parse_inputs_section = parse_inputs_section


def parse_options_section(self, section):
    return self._format_fields("Options", self._consume_fields())


NumpyDocstring._parse_options_section = parse_options_section


def parse_outputs_section(self, section):
    return self._format_fields("Outputs", self._consume_fields())


NumpyDocstring._parse_outputs_section = parse_outputs_section


def patched_parse(self):
    self._sections["inputs"] = self._parse_inputs_section
    self._sections["outputs"] = self._parse_outputs_section
    self._sections["options"] = self._parse_options_section
    self._unpatched_parse()


NumpyDocstring._unpatched_parse = NumpyDocstring._parse
NumpyDocstring._parse = patched_parse

# -- Project information -----------------------------------------------------

project = "OpenConcept"
author = "Benjamin J. Brelje and Eytan J. Adler"

import openconcept

# The short X.Y version
version = openconcept.__version__
# The full version, including alpha/beta/rc tags
release = openconcept.__version__ + " alpha"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "1.5"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_mdolab_theme.ext.embed_code",
    "sphinx_mdolab_theme.ext.embed_compare",
    "sphinx_mdolab_theme.ext.embed_n2",
]
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
autoclass_content = "class"

autosummary_generate = []

# Ignore docs errors
nitpick_ignore_regex = [("py:class", ".*")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# This sets the bibtex bibliography file(s) to reference in the documentation
bibtex_bibfiles = ["ref.bib"]

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "openconceptdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
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
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "openconcept.tex", "openconcept Documentation", "Benjamin J. Brelje", "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "openconcept", "openconcept Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "openconcept",
        "openconcept Documentation",
        author,
        "openconcept",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Extension configuration -------------------------------------------------

# -- Run examples to get figures for docs ------------------------------------
run_file_move_result(
    "../openconcept/examples/minimal.py",
    ["minimal_example_results.svg"],
    ["tutorials/assets/minimal_example_results.svg"],
    optional_cl_args=["--hide_visuals"],
)
run_file_move_result(
    "../openconcept/examples/minimal_integrator.py",
    ["minimal_integrator_results.svg"],
    ["tutorials/assets/minimal_integrator_results.svg"],
    optional_cl_args=["--hide_visuals"],
)
run_file_move_result(
    "../openconcept/examples/TBM850.py",
    ["turboprop_takeoff_results.svg", "turboprop_mission_results.svg"],
    ["tutorials/assets/turboprop_takeoff_results.svg", "tutorials/assets/turboprop_mission_results.svg"],
    optional_cl_args=["--hide_visuals"],
)

# Remove the N2 diagrams it also created
files_remove = ["minimal_example_n2.html", "minimal_integrator_n2.html", "turboprop_n2.html"]
for file in files_remove:
    os.remove(file)

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
generate_srcdocs = True

if generate_srcdocs:
    # native way
    # subprocess.call(['sphinx-apidoc','-o','_srcdocs_native','../openconcept'])
    # os.rename('_srcdocs_native/modules.rst','_srcdocs_native/index.rst')
    # openmdao way
    packages = [
        "aerodynamics",
        "aerodynamics.openaerostruct",
        "atmospherics",
        "energy_storage",
        "mission",
        "propulsion",
        "propulsion.systems",
        "thermal",
        "utilities",
        "utilities.math",
    ]
    generate_src_docs(".", "../openconcept", packages)
