"""
html_theme is usually unchanged (rocm_docs_theme).
flavor defines the site header display, select the flavor for the corresponding portals
flavor options: rocm, rocm-docs-home, rocm-blogs, rocm-ds, instinct, ai-developer-hub, local, generic
"""

version_number = "4.0"

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "generic",
    "header_title": f"GEAK {version_number}",
    "header_link": False,
    "version_list_link": False,
    "nav_secondary_items": {
        "Hyperloom": "https://advanced-micro-devices-demo--660.com.readthedocs.build/projects/hyperloom/en/660/",
        "GitHub": "https://github.com/AMD-AGI/GEAK",
        "Community": False,
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
        "Instinct™ Docs": "https://instinct.docs.amd.com/",
        "Infinity Hub": "https://www.amd.com/en/developer/resources/infinity-hub.html",
        "Support": "https://github.com/AMD-AGI/GEAK/issues/new/choose",
    },
    "link_main_doc": False,
}

# This section turns on/off article info
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

# for PDF output on Read the Docs
project = "GEAK"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"  # Defines Table of Content structure definition path

"""
Doxygen Settings
Ensure Doxyfile is located at docs/doxygen.
If the component does not need doxygen, delete this section for optimal build time
"""
# doxygen_root = "doxygen"
# doxysphinx_enabled = True
# doxygen_project = {
#    "name": "doxygen",
#    "path": "doxygen/xml",
# }

# Add more additional package accordingly
extensions = [
    "rocm_docs"
]

myst_fence_as_directive = ["mermaid"]

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "GEAK"