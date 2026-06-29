.. meta::
   :description: GEAK is an AMD AI agent that optimizes GPU kernels for ROCm. It profiles, optimizes, and validates HIP, Triton, and FlyDSL kernels using LLM-guided multi-agent search.
   :keywords: GEAK, ROCm, GPU kernel optimization, AMD, Triton, HIP, AMD Instinct, LLM agent, kernel tuning

GEAK documentation
==================

GEAK (Generating Efficient AI-Centric Kernels) is an agent-driven framework for end-to-end GPU kernel optimization in real codebases, producing reviewable patches backed by profiling, 
testing, and LLM-guided iteration. Supports HIP, Triton, and FlyDSL kernels.

The GEAK public repository is located at `AMD-AGI/GEAK <https://github.com/AMD-AGI/GEAK>`_.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install GEAK <install/install>`

   .. grid-item-card:: How to

      * :doc:`Run the agent <how-to/run-agent>`

   .. grid-item-card:: Conceptual

      * :doc:`GEAK agent loop <conceptual/geak-pipeline>`

   .. grid-item-card:: Reference

      * :doc:`API reference <reference/api-reference>`

For information on contributing to the GEAK code base, see
:doc:`GEAK GitHub repository <https://github.com/AMD-AGI/GEAK/blob/rocm-docs-review/CONTRIBUTING.md>`.
