"""Read task ownership without confusing standalone kernel members with task names."""


def selected_identities(row):
    values = row.get("selected_identities")
    if values is None:
        values = row.get("selected_device_kernels") or []
    return [value.get("device_kernel") if isinstance(value, dict) else value
            for value in values]


def task_records(plan):
    # The standalone skill names a row `task`; older slot deliveries used `name`.
    for row in (plan or {}).get("tasks") or []:
        name = row.get("name") or row.get("task") or row.get("id") or row.get("short_name")
        yield name, selected_identities(row)
