# R03 funnel addendum: newly recovered lineage B evidence

This adds eleven lineage-B role records to the [published R03 funnel](r03-candidate-funnel-20261028.md). The previous extract remains preserved. The added evidence is from one pinned historical archive; it adds no new GPU measurement. Actual Claude reviewed and agreed to this addendum on 2026-10-28 at 17:06 UTC.

B’s eight unverified rows can now be distinguished more precisely. Three first-round directions have explicit invalid verifier results; three second-round directions failed before engineer dispatch; the last two have an invalid or absent verifier result. B’s other three rows retain valid slower measurements.

| B rows | Newly available engineer evidence | Verifier evidence | Supported classification |
|---|---|---|---|
| r1_d0, r1_d2 | Same ORIGINAL candidate **and same evaluation**; ratio 1.001398×, gain-ineligible | Each records invalid_measurement; external validity false, primary null | Two archive rows reference one recorded ORIGINAL measurement; no accepted verifier result |
| r1_d1 | Null reduced engineer object; candidate remains unknown | invalid_measurement; external validity false, primary null | Engineer identity remains unresolved; explicit invalid verification |
| r2_d0, r2_d1, r2_d2 | Synthetic failed initialization statuses | Null verifier objects | Existing pre-engineer initialization-failure classification unchanged |
| r3_d0 | Valid/pass engineer result, 0.953579× | Valid/pass, 0.949089×; same recorded candidate | Verified slower candidate |
| r3_d1 | Valid/pass engineer result, 0.991630× | Valid/pass, 0.994713×; **different recorded candidate** | Verified slower candidate; scores are not repeated measurements of the same recorded program |
| r3_d2 | Valid/pass engineer result, 0.954644× | Valid/pass, 0.946959×; same recorded candidate | Verified slower candidate |
| r4_d0 | Engineer-side valid gain record, 1.013810× | invalid_measurement; another candidate ID, external validity false, primary null | Engineer gain remains unverified; reason for invalid verification unresolved |
| r4_d1 | Engineer-side valid gain record, 1.002717× | Null verifier object | Engineer gain remains unverified |

The eleven rows therefore contain **three valid verifier objects, four invalid-measurement objects and four null objects**. Three nulls belong to initialization failures; the fourth is r4_d1. The archive statuses remain three slower and eight unverified.

The two B first-round identity matches join the previously known D r1_d1 match: **three of the four failed-initialization parent rows now carry engineer candidate IDs equal to ORIGINAL**. B r1_d1 remains unknown. Their recorded patch digests equal the zero-byte-file digest; no fresh file stat was performed. Neither observation establishes that no source work was ever produced.

B r1_d0 and r1_d2 both name candidate `9befb93fc09593295428cfe74d6e39a0eedab0350b1e7f38e627400ccc322241` and evaluation `arena-6969f57e3c5c456ea51c2871f45953c4`. Their slightly above-one timing ratio is explicitly gain-ineligible. Do not count those archive rows as two independent measurements or two improvements.

B r3_d1 records engineer candidate `6e91f9d9344f19ba6fa691bc00b7485dd96eec6412aba3d6d4f84510628d3deb` and verifier candidate `74e8d855d74ad1c3310d0cc8dd5502dff4f6f0d78d3e07b6d710e5f949001ab3`. B r4_d0 records engineer candidate `2c4f95ee657e81b6ac1aef9aa63a258de4048138740905842c856d81e09d0beb` and invalid-verifier candidate `4b13d30ec3db7bee683537dff039ac24a4f891ce1d834823db21fbfd6bc656a4`. These identity differences do not establish their cause. The invalid record does not show that the engineer’s other candidate failed numerical correctness.

The four invalid-verifier correctness strings were exported as metadata only. This addendum does not infer `not_checked` or another value. Original complete role returns, returned patch paths, direct source-content joins and invalid-verification causes remain unknown. Missing fields in the reduced archive serializer are not evidence that the original returned object lacked them.

The two late engineer-side gains add no accepted verifier gain or common-qualified finalist. The prior six-verifier-gain candidate list, four initialization failures, all eight timelines and completed GEAK/GEAK-EVO final scores remain unchanged. The evidence still does not isolate a cause for the measured method gap.

Exact field states, candidate/evaluation identities and source hashes are in the [addendum JSON](r03-candidate-funnel-b-update-20261028.json). The input public projection is 70,877 bytes, SHA256 `7c094fb1c7063b26f8665a5551ca5abb5e1cb5f0d42d16a71d94a66649f7d1a1`; it binds the 53,852-byte B archive SHA256 `0593f1fd3ed6f5902bcf3f8e3d7dfd025f5001223d75899f095d7ce47e5fcbdd`.
