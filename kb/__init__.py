"""The knowledge-base plane, shared by both workflows.

Nothing in here knows what a kernel is or what an e2e serving run is. It owns the two things the
two workflows have in common — the ADDRESS a record is filed under and the STORE it is filed in —
and leaves the workflow-specific halves (which ladder to build, what to put in `value`) to the
callers in kernel_workflow/ and e2e_workflow/.

    identity.py      canonical ids and the fallback ladders, for both domains
    store_local.py   a KB Store held on disk, in the shape the service uses
    store_remote.py  the HTTP plane, wearing store_local's interface
    store_client.py  the standalone HTTP client store_remote is built on
    retract.py       taking back a record on a store that has no delete
    remote_upload.py CLI: push exported records at either plane

This lives at the repo root rather than under kernel_workflow/scripts/ — where it grew — because
e2e_workflow imports it too, and a workflow should not have to depend on a sibling workflow's
directory being present to read its own knowledge.

The modules are imported absolutely (`from kb.store_local import ...`), so an entry point that is
executed as a file rather than imported has to put the repo root on sys.path first. Every such
caller does this from its own __file__ and nothing here assumes a cwd.
"""
