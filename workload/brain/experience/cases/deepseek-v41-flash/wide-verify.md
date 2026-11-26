# Wide verify batches

Origin: `deepseek-v41-flash`. Public SGLang release comparison on two MI355X,
TP2/EP1 with DP attention, C64 and GPU KV. Both runs used the ordinary decoder.
During native timed sending, median active requests on the two emitting workers
were 15 and 16 before the change, then 13 on each. Output improved 10.79% and
full-response P90 improved 18.89%. P90 TTFT went from 2.60 to 2.32 seconds.
[Reference](https://inferencex.semianalysis.com/api/v1/reproduce?id=444981) ·
[Treatment](https://inferencex.semianalysis.com/api/v1/reproduce?id=446046).

The release grouped verify rows to reuse KV reads, changed index pages and expert
dispatch. The narrow persistent scorer expects the older page layout. For a new
load, profile scorer and expert costs at the served widths. A row-group comparison
at one compatible layout can isolate reuse. A page-layout port must include every
affected reader and writer. The release measured these changes together.
[Source and release](https://github.com/SemiAnalysisAI/InferenceX/pull/4367).
