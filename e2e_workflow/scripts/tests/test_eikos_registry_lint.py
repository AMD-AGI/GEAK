"""Offline tests for eikos_registry_lint.py: independent discovery of schema roots, references, field
paths and call sites, and two-way reconciliation with the decision registry. Stage 1 only: no capture
hook or replay receipt exists yet, and nothing here claims one does."""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eikos_registry_lint as rl  # noqa: E402

HEADER = """
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const arrStr = { type: 'array', items: { type: 'string' } };
const LLM_TL = { schema: 'geak.agent_timeline/1', events: [] };
"""


def discover(tmp_path, body, name="wf.js"):
    (tmp_path / name).write_text(HEADER + body)
    return rl.discover(root=str(tmp_path), files=(name,))[name]


def paths(found, root):
    return [p["path"] for p in found["roots"][root]["paths"]]


# ------------------------------------------------------------------ discovery
def test_named_schema_fragments_nested_items_and_open_boundaries(tmp_path):
    f = discover(tmp_path, """
const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: { type: 'array', items: obj({ specialty: { type: 'string', enum: ['a', 'b'] } }, []) },
  picks: arrObj, notes: arrStr,
}, ['stop']);
await agent('plan', { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });
""")
    assert f["errors"] == [] and f["ignored"] and "string tag" in f["ignored"][0]   # LLM_TL is not a schema
    assert paths(f, "PLAN_SCHEMA") == ["stop", "reasoning", "directions[]", "directions[].specialty",
                                       "directions[].*", "picks[]", "picks[].*", "notes[]", "*"]
    leaf = [p for p in f["roots"]["PLAN_SCHEMA"]["paths"] if p["path"] == "directions[].specialty"][0]
    assert leaf["enum"] == ["a", "b"]
    assert f["roots"]["PLAN_SCHEMA"]["sites"] == [["tech_lead:plan r${}", "Optimize"]]


def test_alias_inline_literal_and_shorthand_resolve(tmp_path):
    f = discover(tmp_path, """
const BASE = obj({ ok: { type: 'boolean' } }, []);
const ALIAS = BASE;
async function f(schema) { return 1; }
await agent('a', { phase: 'P', label: 'aliased', schema: ALIAS });
await agent('b', { phase: 'P', label: 'inline', schema: { type: 'object', properties: { n: { type: 'number' } } } });
""")
    assert f["errors"] == []
    assert paths(f, "ALIAS") == ["ok", "*"]
    assert paths(f, "inline:inline@P") == ["n", "*"]                    # no additionalProperties = open (JSON Schema)


def test_an_unresolvable_reference_is_an_error_not_an_empty_schema(tmp_path):
    f = discover(tmp_path, """
function mk(extra) { return obj(extra, []); }
const RUNTIME = mk(A.extra);
await agent('a', { phase: 'P', label: 'x', schema: RUNTIME });
await agent('b', { phase: 'P', label: 'y', schema: UNDEFINED_SCHEMA });
""")
    assert "RUNTIME" not in f["roots"] and "UNDEFINED_SCHEMA" not in f["roots"]
    errs = " ".join(f["errors"])
    assert "RUNTIME" in errs and "UNDEFINED_SCHEMA" in errs


def test_shorthand_schema_from_a_parameter_is_an_error(tmp_path):
    f = discover(tmp_path, """
async function call(p, schema) { return agent(p, { phase: 'P', label: 'generic', schema }); }
""")
    assert f["roots"] == {} and any("schema" in e for e in f["errors"])


def test_inline_schema_without_label_or_phase_is_an_error(tmp_path):
    f = discover(tmp_path, "await agent('a', { schema: obj({ ok: { type: 'boolean' } }, []) });\n")
    assert any("without a static label or phase" in e for e in f["errors"])


def test_one_inline_identity_bound_to_two_schemas_is_an_error(tmp_path):
    f = discover(tmp_path, """
await agent('a', { phase: 'P', label: 'same', schema: obj({ a: { type: 'boolean' } }, []) });
await agent('b', { phase: 'P', label: 'same', schema: obj({ b: { type: 'boolean' } }, []) });
""")
    assert any("two different schema expressions" in e for e in f["errors"])


def test_comments_strings_regexes_and_templates_do_not_create_sites(tmp_path):
    f = discover(tmp_path, r"""
// { schema: FAKE_SCHEMA } in a comment
/* { schema: FAKE2 } */
const s = "{ schema: FAKE3 }";
const re = /schema: '[^']*'/;
const t = `${ `nested ${'{ schema: FAKE4 }'}` }`;
const REAL = obj({ ok: { type: 'boolean' } }, []);
await agent('a', { phase: 'P', label: 'real', schema: REAL });
""")
    assert f["errors"] == [] and list(f["roots"]) == ["REAL"]


# ------------------------------------------------------------------ reconciliation
def registry_for(f, rel="wf.js"):
    files = {rel: {rid: {"call_sites": [list(s) for s in r["sites"]],
                         "fields": {p["path"]: {"class": "open_object" if p["kind"] == "open_object" else "free_text"}
                                    for p in r["paths"]}}
                   for rid, r in f["roots"].items()}}
    return {"decisions": [{"id": "D1", "class": "judgment", "capture": "not_implemented"}], "files": files}


@pytest.fixture
def base(tmp_path):
    f = discover(tmp_path, """
const S = obj({ stop: { type: 'boolean' }, why: { type: 'string' } }, []);
await agent('a', { phase: 'P', label: 'plan', schema: S });
""")
    return tmp_path, {"wf.js": f}, registry_for(f)


def run_lint(base, reg):
    tmp, found, _ = base
    return rl.lint(reg, found, root=str(tmp), files=("wf.js",))


def test_a_complete_registry_passes(base):
    assert run_lint(base, base[2]) == []


def test_an_unclassified_field_fails(base):
    reg = copy.deepcopy(base[2])
    del reg["files"]["wf.js"]["S"]["fields"]["stop"]
    assert any("S.stop" in p and "not classified" in p for p in run_lint(base, reg))


def test_a_stale_registered_field_fails(base):
    reg = copy.deepcopy(base[2])
    reg["files"]["wf.js"]["S"]["fields"]["gone"] = {"class": "free_text"}
    assert any("no longer exists" in p for p in run_lint(base, reg))


def test_an_unregistered_or_stale_call_site_fails(base):
    reg = copy.deepcopy(base[2])
    reg["files"]["wf.js"]["S"]["call_sites"] = [["other", "P"]]
    assert any("call sites differ" in p for p in run_lint(base, reg))
    reg = copy.deepcopy(base[2])
    reg["files"]["wf.js"]["OLD"] = {"call_sites": [], "fields": {}}
    assert any("no longer has a call site" in p for p in run_lint(base, reg))


def test_an_unregistered_root_fails(base):
    reg = copy.deepcopy(base[2])
    del reg["files"]["wf.js"]["S"]
    assert any("schema root S" in p and "not in the registry" in p for p in run_lint(base, reg))


@pytest.mark.parametrize("entry, why", [
    ({"class": "judgment"}, "names no registered decision"),
    ({"class": "judgment", "decision": "NOPE"}, "names no registered decision"),
    ({"class": "derived_fact"}, "without a policy citation"),
    ({"class": "maybe"}, "unknown class"),
])
def test_class_requirements(base, entry, why):
    reg = copy.deepcopy(base[2])
    reg["files"]["wf.js"]["S"]["fields"]["stop"] = entry
    assert any(why in p for p in run_lint(base, reg))


def test_an_open_boundary_must_be_classified_open_object(base):
    reg = copy.deepcopy(base[2])
    reg["files"]["wf.js"]["S"]["fields"]["*"] = {"class": "free_text"}
    assert any("open-object boundary" in p for p in run_lint(base, reg))


def test_claiming_capture_needs_a_marker_in_the_source(base):
    reg = copy.deepcopy(base[2])
    reg["decisions"][0]["capture"] = "implemented"
    assert any("no EIKOS-CAPTURE:D1 marker" in p for p in run_lint(base, reg))
    tmp = base[0]
    (tmp / "wf.js").write_text((tmp / "wf.js").read_text() + "\n// EIKOS-CAPTURE:D1\n")
    assert run_lint(base, reg) == []


def test_internal_site_files_must_exist(base):
    reg = copy.deepcopy(base[2])
    reg["decisions"][0]["internal_files"] = ["roles/missing.md"]
    assert any("missing" in p for p in run_lint(base, reg))


# ------------------------------------------------------------------ the real sources (stage-1 fixture)
def test_the_shipped_registry_matches_the_real_workflow_sources():
    """Discovery over the three real entry files reconciles with the shipped registry. This is the
    stage-1 real call-site check; capture/return-path assertions belong to later steps."""
    with open(rl.REGISTRY, encoding="utf-8") as fh:
        reg = json.load(fh)
    found = rl.discover()
    assert rl.lint(reg, found) == []
    assert all(d["capture"] == "not_implemented" for d in reg["decisions"])   # nothing claimed yet
    classes = {e["class"] for f in reg["files"].values() for r in f.values() for e in r["fields"].values()}
    assert classes <= set(rl.CLASSES)



# ------------------------------------------------------------------ review of 356f2ce0: silent passes
# Each case adds a typed judgment or a call site through valid JavaScript; each must now fail.
SCHEMA = "const S = {type:'object',properties:{known:{type:'boolean'},variant:{type:'string'}},additionalProperties:false};\n"
CALL = "await agent('a', {phase:'P',label:'plan r1',schema:S});\n"
BASE = SCHEMA + CALL
EXTRA = "{type:'object',properties:{new_judgment:{type:'boolean'}},additionalProperties:false}"
SILENT_CASES = {
    "quoted_schema_key": (BASE, BASE + "await agent('b',{phase:'P',label:'new','schema':" + EXTRA + "});\n"),
    "computed_literal_schema_key": (BASE, BASE + "await agent('b',{phase:'P',label:'new',['schema']:" + EXTRA + "});\n"),
    "executable_template_interpolation": (BASE, BASE + "const text = `${await agent('b',{phase:'P',label:'new',schema:" + EXTRA + "})}`;\n"),
    "mutated_schema_properties": (BASE, SCHEMA + "S.properties.new_judgment={type:'boolean'};\n" + CALL),
    "parameter_shadows_schema": (BASE, SCHEMA + "async function f(S){" + CALL + "}\n"),
    "new_call_same_label_prefix": (BASE, BASE + "await agent('b',{phase:'P',label:'plan r2',schema:S});\n"),
    "repeated_identical_label": (BASE, BASE + CALL),
    "oneof_nested_judgment": (BASE, BASE.replace("variant:{type:'string'}", "variant:{oneOf:[" + EXTRA + "]}")),
    "nested_array_judgment": (
        BASE.replace("variant:{type:'string'}", "variant:{type:'array',items:{type:'string'}}"),
        BASE.replace("variant:{type:'string'}", "variant:{type:'array',items:{type:'array',items:" + EXTRA + "}}")),
    "default_additional_properties": (BASE, BASE.replace(",additionalProperties:false", "")),
    "schema_valued_additional_properties": (BASE, BASE.replace("additionalProperties:false", "additionalProperties:{type:'boolean'}")),
}


def _registry_from(found, rel):
    return {"decisions": [], "files": {rel: {
        rid: {"call_sites": [list(x) for x in r["sites"]],
              "fields": {p["path"]: {"class": "open_object" if p["kind"] == "open_object" else "measurement_report"}
                         for p in r["paths"]}}
        for rid, r in found[rel]["roots"].items()}}}


@pytest.mark.parametrize("case", sorted(SILENT_CASES))
def test_a_source_change_that_adds_a_decision_surface_cannot_pass_silently(tmp_path, case):
    before, after = SILENT_CASES[case]
    rel = "wf.mjs"
    (tmp_path / rel).write_text(before)
    first = rl.discover(str(tmp_path), (rel,))
    assert first[rel]["errors"] == []
    reg = _registry_from(first, rel)
    assert rl.lint(reg, first, str(tmp_path), (rel,)) == []
    (tmp_path / rel).write_text(after)
    found = rl.discover(str(tmp_path), (rel,))
    assert found[rel]["errors"] or rl.lint(reg, found, str(tmp_path), (rel,)), case


def test_explicitly_closed_objects_have_no_open_boundary(tmp_path):
    f = discover(tmp_path, BASE)
    assert paths(f, "S") == ["known", "variant"]


def test_a_schema_valued_additional_properties_is_walked(tmp_path):
    f = discover(tmp_path, BASE.replace("additionalProperties:false",
                                        "additionalProperties:{type:'object',properties:{x:{type:'boolean'}},additionalProperties:false}"))
    assert paths(f, "S") == ["known", "variant", "*", "<*>.x"]


def test_literal_template_text_mentioning_schema_stays_ignored(tmp_path):
    f = discover(tmp_path, BASE + "const prompt = `return JSON matching {schema: PLAN} please`;\n")
    assert f["errors"] == [] and list(f["roots"]) == ["S"]


def test_a_schema_only_definition_without_a_call_site_is_listed_not_hidden(tmp_path):
    f = discover(tmp_path, "const USED = obj({ ok: { type: 'boolean' } }, []);\n"
                           "const UNUSED = obj({ ran: { type: 'boolean' } }, []);\n"
                           "await agent('a', { phase: 'P', label: 'x', schema: USED });\n")
    assert f["errors"] == [] and f["unused_schemas"] == ["UNUSED"] and "UNUSED" not in f["roots"]


def test_scalar_array_items_are_one_entry_with_their_type(tmp_path):
    f = discover(tmp_path, "const S = obj({ tags: { type: 'array', items: { type: 'string', enum: ['a'] } } }, []);\n"
                           "await agent('a', { phase: 'P', label: 'x', schema: S });\n")
    node = [p for p in f["roots"]["S"]["paths"] if p["path"] == "tags[]"]
    assert len(node) == 1 and node[0]["items_type"] == "string" and node[0]["enum"] == ["a"]


# ------------------------------------------------------------------ review of d41ae3a2: remaining boundaries
S2 = "const S={type:'object',properties:{known:{type:'boolean'},variant:{type:'string'}},additionalProperties:false};\n"
C2 = "await agent('a',{phase:'P',label:'plan',schema:S});\n"
ITEM = "{type:'object',properties:{old_field:{type:'boolean'}},additionalProperties:false}"
ARR = S2.replace("variant:{type:'string'}", "variant:{type:'array',items:" + ITEM + "}") + C2
OBJ = S2.replace("variant:{type:'string'}", "variant:" + ITEM) + C2
SILENT_CASES_2 = {
    "mutation_in_other_const_initializer": (S2 + C2, S2 + "const EXTRA=Object.assign(S.properties,{new_judgment:{type:'boolean'}});\n" + C2),
    "escaped_static_schema_key": (S2 + C2, S2 + C2 + "await agent('b',{phase:'P',label:'new','sch\\u0065ma':" + ITEM + "});\n"),
    "one_property_object_array": (ARR, ARR.replace("old_field", "new_judgment")),
    "object_array_union": (OBJ, OBJ.replace("variant:{type:'object'", "variant:{type:['object','array'],items:"
                                             + ITEM.replace("old_field", "new_judgment"))),
}


@pytest.mark.parametrize("case", sorted(SILENT_CASES_2))
def test_remaining_boundaries_cannot_pass_silently(tmp_path, case):
    before, after = SILENT_CASES_2[case]
    rel = "wf.mjs"
    (tmp_path / rel).write_text(before)
    first = rl.discover(str(tmp_path), (rel,))
    assert first[rel]["errors"] == []
    reg = _registry_from(first, rel)
    assert rl.lint(reg, first, str(tmp_path), (rel,)) == []
    (tmp_path / rel).write_text(after)
    found = rl.discover(str(tmp_path), (rel,))
    assert found[rel]["errors"] or rl.lint(reg, found, str(tmp_path), (rel,)), case


@pytest.mark.parametrize("lit, value", [("'sch\\u0065ma'", "schema"), ("'a\\x41\\u{42}c'", "aABc"),
                                        ("'it\\'s'", "it's"), ('"q\\"x"', 'q"x'), ("'a\\\nb'", "ab")])
def test_string_keys_are_compared_by_value(lit, value):
    assert rl.js_string_value(lit) == value


def test_an_escaped_identifier_is_rejected_not_skipped(tmp_path):
    f = discover(tmp_path, S2 + C2 + "await agent('b',{phase:'P',label:'new',sch\\u0065ma:" + ITEM + "});\n")
    assert any("unicode-escaped identifier" in e for e in f["errors"])


@pytest.mark.parametrize("init, ok", [
    ("obj({ a: { type: 'boolean' } }, [])", True), ("BASE", True), ("{ type: 'array', items: BASE }", True),
    ("Object.assign({}, BASE)", False), ("JSON.parse(JSON.stringify(BASE))", False), ("BASE.properties", False),
    ("(BASE.x = 1, BASE)", False), ("mk(BASE)", False),
])
def test_only_construction_forms_may_build_on_a_schema(tmp_path, init, ok):
    f = discover(tmp_path, "const BASE = obj({ k: { type: 'boolean' } }, []);\n"
                           "const X = " + init + ";\n"
                           "await agent('a', { phase: 'P', label: 'x', schema: X });\n")
    assert (f["errors"] == []) == ok, f["errors"]


def test_items_typed_against_object_keywords_are_rejected(tmp_path):
    f = discover(tmp_path, "const S = { type: 'string', properties: { a: { type: 'boolean' } } };\n"
                           "await agent('a', { phase: 'P', label: 'x', schema: S });\n")
    assert any("object keywords on a schema typed" in e for e in f["errors"])


# ------------------------------------------------------------------ review of ce829685: unconstrained schemas
UNTYPED_BASE = "const S={type:'object',properties:{value:VALUE},additionalProperties:false};\n" \
               "await agent('a',{label:'plan',phase:'P',schema:S});\n"


@pytest.mark.parametrize("before, after", [
    ("{type:'string'}", "{}"),                                                   # field becomes "anything"
    ("{type:'array',items:{type:'string'}}", "{type:'array',items:{}}"),         # items become "anything"
])
def test_an_unconstrained_schema_is_an_open_boundary_not_a_scalar(tmp_path, before, after):
    rel = "wf.mjs"
    (tmp_path / rel).write_text(UNTYPED_BASE.replace("VALUE", before))
    first = rl.discover(str(tmp_path), (rel,))
    reg = _registry_from(first, rel)
    assert rl.lint(reg, first, str(tmp_path), (rel,)) == []
    (tmp_path / rel).write_text(UNTYPED_BASE.replace("VALUE", after))
    found = rl.discover(str(tmp_path), (rel,))
    probs = rl.lint(reg, found, str(tmp_path), (rel,))
    assert any("open-object boundary" in p or "not classified" in p for p in probs), probs


def test_unconstrained_paths_are_reported_as_open(tmp_path):
    f = discover(tmp_path, UNTYPED_BASE.replace("VALUE", "{}"))
    assert [(p["path"], p["kind"]) for p in f["roots"]["S"]["paths"]] == [("value", "open_object")]
    f = discover(tmp_path, UNTYPED_BASE.replace("VALUE", "{enum:['a','b']}"))
    assert [(p["path"], p["kind"]) for p in f["roots"]["S"]["paths"]] == [("value", "leaf")]   # enum constrains it
