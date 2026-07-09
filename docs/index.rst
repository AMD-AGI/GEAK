.. meta::
   :description: GEAK is an AMD multi-agent system that optimizes GPU kernels and whole-model serving throughput on ROCm, driven by Claude Code and deterministic JS Workflows.
   :keywords: GEAK, ROCm, GPU kernel optimization, serving throughput, sglang, vLLM, AMD Instinct, Triton, HIP, CK, FlyDSL

GEAK documentation
==================

GEAK (Generating Efficient AI-Centric Kernels) is a multi-agent GPU performance optimizer for
AMD Instinct MI GPUs (CDNA). It ships two deterministic **Workflows**: ``e2e_workflow`` raises the
end-to-end **sglang / vLLM serving throughput** of a whole model, and ``kernel_workflow`` optimizes a
single AMD GPU kernel (Triton, HIP, CK, FlyDSL). Control flow is deterministic JS; LLM agents are called
only for judgement.

The GEAK public repository is located at `AMD-AGI/GEAK <https://github.com/AMD-AGI/GEAK>`_.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install GEAK <install/install>`

   .. grid-item-card:: How to

      * :doc:`Run a workflow <how-to/run-agent>`

   .. grid-item-card:: Conceptual

      * :doc:`GEAK pipeline <conceptual/geak-pipeline>`

   .. grid-item-card:: Reference

      * :doc:`Reference <reference/api-reference>`

For information on contributing to the GEAK code base, see the
`CONTRIBUTING guide <https://github.com/AMD-AGI/GEAK/blob/GEAK_v4/CONTRIBUTING.md>`_.
