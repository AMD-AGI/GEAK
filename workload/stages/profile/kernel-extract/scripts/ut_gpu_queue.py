#!/usr/bin/env python3
"""Host GPU broker for CPU-only UT authors. Requests contain argv, never tensor data.

The broker outlives the controller long enough to drain its containers/processes on
controller death. Host flock descriptors stay open until the whole GPU job exits.
"""
import ctypes
import fcntl
import glob
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().split(') ', 1)[1].split()
        return fields[19] if fields[0] not in ('Z', 'X') else None
    except (OSError, IndexError):
        return None


def release(fds):
    for fd in fds:
        os.close(fd)


def try_lease(pool, count, lock_dir, idle):
    """Never wait while holding a partial set; share gpu_lock.sh's physical locks."""
    fds, selected = [], []
    try:
        os.makedirs(lock_dir, exist_ok=True)
        for gpu in sorted(pool):
            fd = os.open(os.path.join(lock_dir, 'gpu_%d.lock' % gpu), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not idle(gpu):
                    os.close(fd)
                    continue
            except BlockingIOError:
                os.close(fd)
                continue
            except BaseException:
                os.close(fd)
                raise
            fds.append(fd)
            selected.append(gpu)
            if len(selected) == count:
                return selected, fds
        return [], []
    finally:
        if len(selected) != count:
            release(fds)


def gpu_idle(gpu, cfg):
    if os.environ.get('GEAK_GPU_REQUIRE_IDLE', '1') != '1':
        return True
    dev = Path('/sys/class/drm/renderD%d/device' % (cfg['render_base'] + cfg['render_step'] * gpu))
    if not (dev / 'gpu_busy_percent').exists():
        return True  # same unknown-device policy as gpu_lock.sh; allocation is still required
    try:
        return (int((dev / 'gpu_busy_percent').read_text()) <= int(os.environ.get('GEAK_GPU_MAX_BUSY_PCT', 5))
                and int((dev / 'mem_info_vram_used').read_text()) // 1048576
                <= int(os.environ.get('GEAK_GPU_MAX_VRAM_MB', 1024)))
    except OSError:
        return False


def backend_for(wave, cfg):
    from ut_wave import Docker, Local
    # Old waves used ut_lane_<task>. Retain that name when draining pre-upgrade jobs.
    prefix = cfg.get('container_prefix', os.environ.get('UT_WAVE_CONTAINER_PREFIX', 'ut_lane_'))
    return (Docker(wave, cfg['image'], cfg['render_base'], cfg['render_step'], prefix)
            if cfg['backend'] == 'docker' else Local(wave))


def submit(a):
    """Run inside the CPU lane. Environment/cwd are explicit; no Docker socket needed."""
    from ut_wave import load, save
    wave = os.environ.get('UT_WAVE_DIR')
    owner = os.environ.get('UT_WAVE_OWNER')
    token = os.environ.get('UT_WAVE_TOKEN')
    if not wave or not owner or not token:
        raise SystemExit('ut_wave gpu must run inside an on-demand wave lane')
    cfg = load(os.path.join(wave, 'wave.json'))
    row = next(r for r in cfg['rows'] if r['label'] == os.environ['UT_WAVE_TASK'])
    count = a.count if a.count is not None else int(os.environ.get('UT_WAVE_GPU_COUNT', row['need']))
    if count < row['need'] or count > len(cfg['gpus']):
        raise SystemExit('GPU count must cover required_gpu_count and fit the allocated pool')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or a.timeout <= 0:
        raise SystemExit('a foreground command and a positive timeout are required')
    env = dict(item.split('=', 1) for item in a.env)
    if any(k in env for k in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                              'GPU_IDS', 'GEAK_GPU_ALLOWED', 'GEAK_GPU_LOCKED_SET')):
        raise SystemExit('device mapping is owned by the broker')
    request = os.path.join(wave, 'gpu_queue', uuid.uuid4().hex)
    os.makedirs(request)
    # A task has a single command in flight. Fail a second submit instead of racing
    # task-local source mutation, reports, or caches.
    guard = open(os.path.join(wave, 'state', owner + '.submit.lock'), 'a')
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('this task already has a GPU command in flight')
    save(os.path.join(request, 'request.json'), {
        'owner': owner, 'task': row['label'], 'token': token, 'count': count,
        'command': command, 'cwd': os.path.abspath(a.cwd or row['task_dir']),
        'env': env, 'timeout': a.timeout, 'submitted': time.time(),
    })
    stopped = [False]
    handlers = {s: signal.signal(s, lambda *_: stopped.__setitem__(0, True))
                for s in (signal.SIGTERM, signal.SIGINT)}
    cursor = 0
    try:
        while True:
            if stopped[0]:
                Path(request, 'cancel').touch()
            output = Path(request, 'output.log')
            if output.exists():
                with output.open('rb') as f:
                    f.seek(cursor)
                    data = f.read()
                    cursor = f.tell()
                if data:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
            result = load(os.path.join(request, 'result.json'))
            if result is not None:
                print('\nGPU lease %s: devices=%s wait=%.2fs run=%.2fs rc=%s' % (
                    Path(request).name, result.get('gpus', []), result.get('wait_s', 0),
                    result.get('run_s', 0), result['returncode']), flush=True)
                if result.get('error'):
                    print(result['error'], file=sys.stderr)
                return result['returncode']
            broker = load(os.path.join(wave, 'broker.json'), {})
            if not broker or identity(broker['pid']) != broker['identity']:
                Path(request, 'cancel').touch()
                raise SystemExit('GPU broker exited; request cancelled; inspect gpu_queue before resuming')
            time.sleep(.2)
    finally:
        if not Path(request, 'result.json').exists():
            Path(request, 'cancel').touch()
        guard.close()
        for s, handler in handlers.items():
            signal.signal(s, handler)


def drain_owner(wave, owner):
    """Invalidate this attempt and drain its requests before another writer starts."""
    from ut_wave import load
    Path(wave, 'state', owner + '.token').unlink(missing_ok=True)
    while True:
        pending = []
        for path in Path(wave, 'gpu_queue').glob('*/request.json'):
            if load(str(path), {}).get('owner') == owner and not path.with_name('result.json').exists():
                path.with_name('cancel').touch()
                pending.append(path)
        if not pending:
            return
        broker = load(os.path.join(wave, 'broker.json'), {})
        if not broker or identity(broker['pid']) != broker['identity']:
            raise RuntimeError('GPU broker lost before task drained; do not relaunch writer')
        time.sleep(.2)


def last_command_passed(wave, task):
    """A UT receipt cannot override a failed source audit / cancelled GPU command."""
    from ut_wave import load
    requests = [(load(str(p), {}), p.parent) for p in Path(wave, 'gpu_queue').glob('*/request.json')]
    requests = [(req, path) for req, path in requests if req.get('task') == task]
    if not requests:
        return False
    _, path = max(requests, key=lambda pair: pair[0]['submitted'])
    return load(str(path / 'result.json'), {}).get('returncode') == 0


def broker(wave, controller_pid):
    from ut_wave import load, save, log
    cfg = load(os.path.join(wave, 'wave.json'))
    parent_id = identity(controller_pid)
    backend = backend_for(wave, cfg)
    lock_dir = cfg['lock_dir']
    rows = {r['label']: r for r in cfg['rows']}
    active = {}
    stopping = [parent_id is None]
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stopping.__setitem__(0, True))
    save(os.path.join(wave, 'broker.json'), {'pid': os.getpid(), 'identity': identity(os.getpid())})

    def valid(req):
        token = load(os.path.join(wave, 'state', req.get('owner', '') + '.token'), {})
        return (req.get('task') in rows and token.get('token') == req.get('token')
                and token.get('task') == req.get('task') and backend.alive(req['owner']))

    def finish(path, req, rc, job=None, error=None):
        # Removing a container / draining the local supervisor precedes unlock AND
        # publication of completion: the author may mutate source as soon as it sees it.
        if job:
            while True:
                try:
                    backend.stop(job['name'])
                    break
                except Exception as error:
                    # Losing contact with Docker is not proof the GPU is free.
                    # Retain leases and retry, so no writer/new GPU job overlaps it.
                    log(wave, 'GPU drain pending; retaining locks: %s' % error)
                    time.sleep(2)
            release(job['fds'])
            active.pop(path, None)
        ended = time.time()
        save(os.path.join(path, 'result.json'), {
            'returncode': rc, 'error': error, 'gpus': job['gpus'] if job else [],
            'wait_s': (job['started'] if job else ended) - req['submitted'],
            'run_s': ended - job['started'] if job else 0,
        })

    try:
        while True:
            stopping[0] |= identity(controller_pid) != parent_id
            requests = []
            for file in Path(wave, 'gpu_queue').glob('*/request.json'):
                if not file.with_name('result.json').exists():
                    requests.append((str(file.parent), load(str(file))))
            requests.sort(key=lambda pair: pair[1]['submitted'])
            for path, req in requests:
                job = active.get(path)
                if stopping[0] or Path(path, 'cancel').exists() or not valid(req):
                    finish(path, req, 130, job, 'GPU request cancelled or author/controller exited')
                    continue
                if time.time() - req['submitted'] > req['timeout']:
                    finish(path, req, 124, job, 'GPU command deadline exceeded (queue wait included)')
                    continue
                if job:
                    if not backend.alive(job['name']):
                        result = load(os.path.join(path, 'command.json'), {})
                        finish(path, req, result.get('returncode', 1), job,
                               None if result else 'GPU worker exited without a result')
                    continue
                if any(j['owner'] == req['owner'] for j in active.values()):
                    continue
                count = req.get('count', 0)
                tdir = os.path.realpath(rows[req['task']]['task_dir'])
                if (not isinstance(count, int) or count < rows[req['task']]['need']
                        or count > len(cfg['gpus'])
                        or os.path.commonpath([tdir, os.path.realpath(req['cwd'])]) != tdir):
                    finish(path, req, 2, error='invalid GPU count or cwd outside task')
                    continue
                # Nonblocking group attempt. A waiter retains NO partial allocation;
                # other requests can use those cards while it waits for its full set.
                gpus, fds = try_lease(cfg['gpus'], count, lock_dir, lambda g: gpu_idle(g, cfg))
                if not fds:
                    continue
                name = 'gpu_' + Path(path).name
                job = {'name': name, 'fds': fds, 'gpus': gpus, 'owner': req['owner'], 'started': time.time()}
                active[path] = job
                env = dict(cfg['env'], **req.get('env', {}))
                env['GEAK_GPU_LOCKED_SET'] = ','.join(map(str, gpus))
                argv = [os.path.abspath(__file__), 'execute', path]
                save(os.path.join(path, 'lease.json'), {'gpus': gpus, 'started': job['started']})
                if not backend.start(name, gpus, argv, env):
                    finish(path, req, 1, job, 'GPU backend failed to start')
            if stopping[0]:
                break
            time.sleep(.2)
    finally:
        # Exceptions also drain every job while we still hold its flock descriptors.
        for path, job in list(active.items()):
            finish(path, load(os.path.join(path, 'request.json')), 130, job, 'broker stopping')
        if identity(controller_pid) != parent_id:
            for row in cfg['rows']:
                backend.stop(row['label'])
                backend.stop('timing_' + row['label'])


def descendants():
    """All descendants of this Linux subreaper, including setsid/double-fork workers."""
    parents = {}
    for file in glob.glob('/proc/[0-9]*/stat'):
        try:
            fields = Path(file).read_text().split(') ', 1)[1].split()
            parents[int(file.split('/')[2])] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    found = {os.getpid()}
    while True:
        more = {pid for pid, ppid in parents.items() if ppid in found} - found
        if not more:
            return found - {os.getpid()}
        found.update(more)


def drain_children():
    deadline = time.monotonic() + 2
    while True:
        children = descendants()
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        if not descendants():
            return
        time.sleep(.05)


def execute(path):
    """GPU job supervisor: source checks, foreground command, then descendant drain."""
    from ut_wave import load, save, SKILL
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise RuntimeError('Linux subreaper is required for GPU worker cleanup')
    req = load(os.path.join(path, 'request.json'))
    stopped = [False]
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stopped.__setitem__(0, True))
    env = dict(os.environ)
    env.setdefault('TORCH_EXTENSIONS_DIR', os.path.join(req['cwd'], '.torch_ext'))
    os.makedirs(env['TORCH_EXTENSIONS_DIR'], exist_ok=True)
    guard = [sys.executable, os.path.join(SKILL, 'shared', 'workspace_sources.py'),
             'check', '--workspace', req['cwd']]
    audit = Path(req['cwd'], '.geak', 'invalid_measurements.jsonl')
    audit_size = lambda: audit.stat().st_size if audit.exists() else 0
    before = audit_size()
    rc = 1
    try:
        with open(os.path.join(path, 'output.log'), 'ab', buffering=0) as out:
            rc = subprocess.call(guard, cwd=req['cwd'], env=env, stdout=out, stderr=out)
            if rc == 0 and not stopped[0]:
                # Pin compilation to the visible architecture, as gpu_lock.sh does.
                if env.get('KERNEL_ENV_KEEP_ARCH') != '1':
                    try:
                        info = subprocess.run(['rocminfo'], env=env, capture_output=True, text=True, timeout=15)
                        archs = {line.split()[1] for line in info.stdout.splitlines()
                                 if line.strip().startswith('Name:') and len(line.split()) == 2
                                 and line.split()[1].startswith('gfx') and line.split()[1] != 'gfx000'}
                        if len(archs) > 1:
                            raise RuntimeError('mixed GPU ISAs; refusing an ambiguous compile architecture')
                        if archs:
                            env['PYTORCH_ROCM_ARCH'] = archs.pop()
                            env.setdefault('GPU_ARCHS', env['PYTORCH_ROCM_ARCH'])
                    except FileNotFoundError:
                        pass
                if stopped[0]:
                    rc = 130
                else:
                    p = subprocess.Popen(req['command'], cwd=req['cwd'], env=env, stdout=out, stderr=out)
                    while p.poll() is None and not stopped[0]:
                        time.sleep(.1)
                    rc = 130 if stopped[0] else p.returncode
            drain_children()
            if subprocess.call(guard, cwd=req['cwd'], env=env, stdout=out, stderr=out) or audit_size() != before:
                rc = 86
    finally:
        drain_children()
    save(os.path.join(path, 'command.json'), {'returncode': rc})
    return rc


if __name__ == '__main__':
    if sys.argv[1] == 'broker':
        broker(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'execute':
        sys.exit(execute(sys.argv[2]))
