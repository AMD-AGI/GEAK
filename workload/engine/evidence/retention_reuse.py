"""Optional reuse from completed archives with held descriptors and bounded records."""
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
import tempfile

MAX_ROW_BYTES = 1 << 20
MAX_INDEX_FILES = 1_000_000
MAX_RECORDS_PER_FILE = 6

_OPENAT2_UNAVAILABLE = False
_LIBC = None


class MaintenanceLockBusy(BlockingIOError):
    """The maintenance flock is busy before journal creation; retry is safe."""


class _OpenHow(ctypes.Structure):
    _fields_ = [('flags', ctypes.c_uint64), ('mode', ctypes.c_uint64), ('resolve', ctypes.c_uint64)]


def _openat2(path, flags):
    """Use Linux pathname permissions, retaining the legacy walk when unsupported."""
    global _LIBC, _OPENAT2_UNAVAILABLE
    if (_OPENAT2_UNAVAILABLE or sys.platform != 'linux'
            or platform.machine() not in ('x86_64', 'aarch64')
            or ctypes.sizeof(ctypes.c_void_p) != 8 or not hasattr(os, 'O_PATH')):
        return None
    # Limit the fast path to ordinary non-creating opens used by retention.
    # Other flags keep os.open's existing creation/ignored-bit semantics.
    allowed = os.O_ACCMODE | os.O_DIRECTORY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_PATH
    if (not isinstance(flags, int) or flags < 0 or flags & ~allowed
            or (flags & os.O_ACCMODE) == os.O_ACCMODE
            or (flags & os.O_DIRECTORY and (flags & os.O_ACCMODE) != os.O_RDONLY)):
        return None
    path_only = bool(flags & os.O_PATH)
    if path_only and flags & ~(os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC):
        return None
    raw = os.fsencode(path)
    if b'\0' in raw:
        raise ValueError('embedded null byte')
    # A component walk can support absolute paths longer than Linux PATH_MAX.
    if len(raw) >= 4096:
        return None
    if _LIBC is None:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.syscall.restype = ctypes.c_long
            _LIBC = libc
        except (OSError, AttributeError):
            _OPENAT2_UNAVAILABLE = True
            return None
    # openat2 rejects status flags that legacy open ignores with O_PATH.
    opened_flags = flags | os.O_NOFOLLOW | os.O_CLOEXEC
    if not path_only:
        opened_flags |= os.O_NONBLOCK
    how = _OpenHow(opened_flags, 0, 0x02 | 0x04)
    while True:
        fd = _LIBC.syscall(ctypes.c_long(437), ctypes.c_int(-100), ctypes.c_char_p(raw),
                           ctypes.byref(how), ctypes.c_size_t(ctypes.sizeof(how)))
        if fd >= 0:
            return fd
        error = ctypes.get_errno()
        if error == errno.EINTR:
            continue
        if error == errno.ENOSYS:
            _OPENAT2_UNAVAILABLE = True
            return None
        raise OSError(error, os.strerror(error), str(path))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def symlink_identity(directory):
    """Hash link names and literal targets, never the trees they point into."""
    directory = Path(directory)
    hasher, count = hashlib.sha256(), 0

    def visit(path):
        nonlocal count
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            row = [path.relative_to(directory).as_posix(), os.readlink(path)]
            hasher.update((json.dumps(row, ensure_ascii=True) + '\n').encode())
            count += 1
        elif stat.S_ISDIR(mode):
            for child in sorted(path.iterdir()):
                visit(child)

    visit(directory)
    return count, hasher.hexdigest()


def verify_symlinks(directory, receipt):
    count, sha256 = symlink_identity(directory)
    if (count != receipt.get('copied_symlinks', 0)
            or sha256 != receipt.get('symlink_sha256', digest(b''))):
        raise ValueError('archived symlink identity differs from receipt')


def open_path(path, flags=os.O_RDONLY):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('reuse paths must be absolute without traversal')
    # NO_SYMLINKS/NO_MAGICLINKS apply to the entire lookup. Unlike the legacy
    # O_RDONLY directory walk, searchable ancestors need not be readable.
    if path.name:
        fd = _openat2(path, flags)
        if fd is not None:
            return fd
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        return os.open(path.name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally: os.close(directory)


def raw_file(path, limit=1 << 20):
    fd = open_path(path)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError('reuse metadata must be regular')
        with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(limit + 1)
        if len(raw) > limit: raise ValueError('reuse metadata exceeds cap')
        return raw
    finally: os.close(fd)


def metadata(fd):
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode): raise ValueError('reuse only supports regular files')
    attrs = {name: digest(os.getxattr(fd, name)) for name in sorted(os.listxattr(fd))}
    return {'semantic': {'size': s.st_size, 'mode': stat.S_IMODE(s.st_mode), 'uid': s.st_uid,
                         'gid': s.st_gid, 'mtime_ns': s.st_mtime_ns, 'xattrs': attrs},
            'identity': {'dev': s.st_dev, 'ino': s.st_ino, 'nlink': s.st_nlink, 'ctime_ns': s.st_ctime_ns}}


def pin_fd(fd):
    before = metadata(fd); os.lseek(fd, 0, os.SEEK_SET)
    hasher, left = hashlib.sha256(), before['semantic']['size']
    while left:
        raw = os.read(fd, min(1 << 20, left))
        if not raw: raise ValueError('file shortened during validation')
        hasher.update(raw); left -= len(raw)
    if os.read(fd, 1) or metadata(fd) != before: raise ValueError('file changed during validation')
    before['sha256'] = hasher.hexdigest()
    return before


def pin(path):
    fd = open_path(path)
    try: return pin_fd(fd)
    finally: os.close(fd)


def same_contents(left, right, owners=True):
    keys = set(left['semantic']) - (set() if owners else {'uid', 'gid'})
    return left['sha256'] == right['sha256'] and all(left['semantic'][key] == right['semantic'][key] for key in keys)


def same_inode(left, right):
    return all(left['identity'][key] == right['identity'][key] for key in ('dev', 'ino'))


def named_fd(path, fd):
    # Identity checks need no server read-open/close on the named inode. This
    # avoids NFS temporary-dentry lifetime effects while payload fds stay readable.
    current = open_path(path, getattr(os, 'O_PATH', os.O_RDONLY))
    try:
        a, b = os.fstat(current), os.fstat(fd)
        if stat.S_ISLNK(a.st_mode): raise ValueError('symlink at identity path')
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino): raise ValueError('pathname no longer names held inode')
    finally: os.close(current)


def fd_link(fd, leaf, parent):
    # A destination dir_fd forces linkat; AT_SYMLINK_FOLLOW resolves this held fd,
    # not a checked-and-closed archive pathname. Unsupported hosts copy instead.
    os.link('/proc/self/fd/' + str(fd), leaf, dst_dir_fd=parent, follow_symlinks=True)


def copy_bytes(source, target, size):
    os.lseek(source, 0, os.SEEK_SET); os.lseek(target, 0, os.SEEK_SET)
    left = size
    while left:
        raw = os.read(source, min(1 << 20, left))
        if not raw: raise ValueError('source shortened during copy')
        offset = 0
        while offset < len(raw): offset += os.write(target, raw[offset:])
        left -= len(raw)
    if os.read(source, 1): raise ValueError('source grew during copy')
    # Like copy2, preserve metadata through the held objects, but do not chown.
    shutil.copystat('/proc/self/fd/' + str(source), '/proc/self/fd/' + str(target))
    os.fsync(target)


def scope(task, destination):
    task, destination = Path(task).absolute(), Path(destination).absolute()
    if task.parent.name != 'tasks' or task.parent.parent.name != 'out' or task.parent.parent.parent.name != 'profile':
        raise ValueError('task is not in a declared campaign profile/out/tasks scope')
    run = task.parents[3]
    owners = [p for p in destination.parents if p.name == task.name and p.parent.name == 'runs'
              and p.is_relative_to(run / 'optimize/out')]
    if len(owners) != 1: raise ValueError('archive has no unique same-run/task owner')
    return run, owners[0]


def read_records(stream, receipt, hint, task):
    count = receipt['copied_files']
    if type(count) is not int or not 0 <= count <= MAX_INDEX_FILES: raise ValueError('reuse index file count exceeds cap')
    expected = receipt['reuse']
    max_records = MAX_RECORDS_PER_FILE * count + 2
    if not 1 <= expected['manifest_records'] <= max_records: raise ValueError('manifest record count exceeds cap')
    if not 1 <= expected['manifest_bytes'] <= max_records * MAX_ROW_BYTES: raise ValueError('manifest bytes exceed row bounds')
    index = {}; hasher = hashlib.sha256(); records = size = 0
    while True:
        raw = stream.readline(MAX_ROW_BYTES + 1)
        if not raw: break
        records += 1; size += len(raw)
        if len(raw) > MAX_ROW_BYTES or records > max_records or size > expected['manifest_bytes']:
            raise ValueError('manifest exceeds bounded records')
        hasher.update(raw); row = json.loads(raw)
        if not isinstance(row, dict): raise ValueError('reuse manifest row must be an object')
        if records == 1 and (row.get('contract') != 'retention_files/v2' or row.get('task_dir') != str(task)):
            raise ValueError('reuse manifest identity differs')
        if row.get('event') != 'file': continue
        rel = row['path']
        if not isinstance(rel, str) or rel.startswith('/') or any(p in ('', '.', '..') for p in rel.split('/')) or rel in index:
            raise ValueError('invalid or duplicate reuse manifest path')
        index[rel] = row['destination_pin']
        if len(index) > count: raise ValueError('manifest has too many files')
    if (hasher.hexdigest() != hint['manifest_sha256'] or hasher.hexdigest() != expected['manifest_sha256']
            or records != expected['manifest_records'] or size != expected['manifest_bytes']):
        raise ValueError('corrupt reuse manifest')
    if len(index) != count or sum(p['semantic']['size'] for p in index.values()) != receipt['copied_bytes']:
        raise ValueError('reuse manifest accounting differs')
    return index


class Reuse:
    def __init__(self, task, destination):
        self.task, self.destination = Path(task).absolute(), Path(destination).absolute()
        self.base = None; self.index = {}; self.lock = None; self.owner = None
        self.stats = {'enabled': True, 'copied_files': 0, 'copied_bytes': 0, 'reused_files': 0, 'reused_bytes': 0,
                      'publication_copy_bytes': 0, 'fallbacks': {}, 'manifest': 'retention-files.jsonl',
                      'manifest_bytes': 0, 'manifest_records': 0, 'base_archive': None}
        self.reason = 'no_completed_hint'; self.manifest_digest = hashlib.sha256()
        self.stream = None
        locked = False
        try:
            run, self.owner = scope(self.task, self.destination); self.hint = self.owner / '.retention-reuse.json'
            self.lock = open_path(run / 'control/completed-archive-dedup.lock', os.O_RDWR)
            if not stat.S_ISREG(os.fstat(self.lock).st_mode): raise ValueError('maintenance lock is not regular')
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MaintenanceLockBusy(*exc.args) from exc
            locked = True
        except (BlockingIOError, TimeoutError, InterruptedError):
            # Acquisition control flow is not optional-hint failure. No journal
            # or payload exists yet, so a Busy caller can retry this same archive.
            self.close()
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.reason = type(exc).__name__ + ': ' + str(exc); self.base = None; self.index = {}
        except BaseException:
            self.close()
            raise
        try:
            fd = os.open(self.destination / self.stats['manifest'], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try: self.stream = os.fdopen(fd, 'w')
            except BaseException:
                os.close(fd)
                raise
            self.log('header', contract='retention_files/v2', task_dir=str(self.task))
            if locked:
                try:
                    hint = json.loads(raw_file(self.hint))
                    if not isinstance(hint, dict) or hint.get('contract') != 'retention_reuse_hint/v2' or hint.get('task_dir') != str(self.task):
                        raise ValueError('hint identity differs')
                    base = Path(hint['archive'])
                    if (not base.is_absolute() or '..' in base.parts or base in (self.destination, self.owner)
                            or not base.is_relative_to(self.owner) or base.is_relative_to(self.task)):
                        raise ValueError('hint archive is outside the same run/task owner')
                    retention_raw = raw_file(base / 'retention.json')
                    if digest(retention_raw) != hint['retention_sha256']: raise ValueError('stale retention hint')
                    receipt = json.loads(retention_raw)
                    if not isinstance(receipt, dict) or receipt.get('complete') is not True or receipt.get('task_dir') != str(self.task):
                        raise ValueError('hint archive is not complete for this task')
                    verify_symlinks(base, receipt)
                    if receipt.get('kind') == 'manual':
                        final = json.loads(raw_file(base / 'manual_exit.json'))
                        if not isinstance(final, dict) or any(type(final.get(a)) is not int or final[a] != receipt.get(b) for a,b in
                               (('native_exit_code','native_exit_code'), ('wrapper_exit_code','measure_exit_code'))):
                            raise ValueError('manual hint lacks matching final exits')
                    fd = open_path(base / 'retention-files.jsonl')
                    try:
                        before = metadata(fd)
                        if before['semantic']['size'] != receipt['reuse']['manifest_bytes']: raise ValueError('manifest size differs')
                        with os.fdopen(fd, 'rb', closefd=False) as stream: self.index = read_records(stream, receipt, hint, self.task)
                        if metadata(fd) != before: raise ValueError('manifest changed while reading')
                    finally: os.close(fd)
                    self.base = base; self.reason = 'validated_completed_hint'; self.stats['base_archive'] = str(base)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.reason = type(exc).__name__ + ': ' + str(exc); self.base = None; self.index = {}
            self.stats['base_status'] = self.reason
            self.log('base_context', base_archive=str(self.base) if self.base else None)
        except BaseException:
            self.close()
            raise

    def log(self, event, **fields):
        line = json.dumps(dict(event=event, **fields), sort_keys=True, separators=(',', ':')) + '\n'
        encoded = line.encode()
        if len(encoded) > MAX_ROW_BYTES: raise ValueError('provenance row exceeds cap')
        self.stream.write(line); self.stream.flush(); os.fsync(self.stream.fileno())
        self.manifest_digest.update(encoded)
        self.stats['manifest_bytes'] += len(encoded); self.stats['manifest_records'] += 1

    def file_record(self, rel, mechanism, source, destination, **fields):
        # Source bytes and non-owner metadata equal destination's checked values.
        # Record owner differences explicitly; copy2 never promised chown.
        self.log('file', path=rel, mechanism=mechanism, source_identity=source['identity'],
                 source_owner={key: source['semantic'][key] for key in ('uid','gid')},
                 destination_pin=destination, **fields)

    def independent(self, source, target, rel, reason):
        with contextlib.ExitStack() as stack:
            def held(fd): stack.callback(os.close, fd); return fd
            src = held(open_path(source)); before = pin_fd(src)
            parent = held(open_path(target.parent, os.O_RDONLY | os.O_DIRECTORY))
            self.log('copy_intent', path=rel, reason=reason, destination=str(target))
            # The archive is unpublished until full retention closure. Keep a
            # failed final-name copy as evidence; recovery uses a fresh sibling.
            try:
                dst = held(os.open(target.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                   0o600, dir_fd=parent))
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    raise ValueError('destination raced; existing evidence retained') from exc
                raise
            copy_bytes(src, dst, before['semantic']['size'])
            copied = pin_fd(dst); after = pin_fd(src)
            named_fd(source, src); named_fd(target, dst); named_fd(target.parent, parent)
            if (after != before or not same_contents(copied, before, owners=False)
                    or copied['identity']['nlink'] != 1 or same_inode(copied, before)):
                raise ValueError('source changed, independent copy differs or destination acquired an alias')
            os.fsync(parent)
            self.stats['copied_files'] += 1; self.stats['copied_bytes'] += copied['semantic']['size']
            category = reason.split(':', 1)[0]
            self.stats['fallbacks'][category] = self.stats['fallbacks'].get(category, 0) + 1
            self.file_record(rel, 'independent_fd_copy', after, copied, reason=reason,
                             publication='exclusive_final_copy')

    def copy(self, source, target):
        source, target = Path(source).absolute(), Path(target).absolute()
        rel = str(target.relative_to(self.destination))
        if self.stats['copied_files'] + self.stats['reused_files'] >= MAX_INDEX_FILES: raise ValueError('snapshot file count exceeds reuse cap')
        if target.exists() or target.is_symlink(): raise ValueError('destination already exists; retain partial evidence')
        reason = self.reason if self.base is None else 'new_or_unindexed_file'
        expected = self.index.get(rel)
        if self.base is not None and expected is not None:
            with contextlib.ExitStack() as stack:
                def held(fd): stack.callback(os.close, fd); return fd
                try:
                    src = held(open_path(source)); base = held(open_path(self.base / rel))
                    source_pin = pin_fd(src); base_pin = pin_fd(base)
                    if base_pin != expected: raise ValueError('base identity, bytes or metadata differs from manifest')
                    if not same_contents(source_pin, base_pin): raise ValueError('source bytes or metadata changed')
                    if same_inode(source_pin, base_pin): raise ValueError('archive aliases the live source')
                    if pin_fd(src) != source_pin: raise ValueError('source changed during validation')
                    named_fd(source, src); named_fd(self.base / rel, base)
                except (OSError, ValueError, KeyError, TypeError) as exc: reason = type(exc).__name__ + ': ' + str(exc)
                else:
                    parent = held(open_path(target.parent, os.O_RDONLY | os.O_DIRECTORY))
                    self.log('reuse_intent', path=rel, base_pin=base_pin, source_identity=source_pin['identity'])
                    try: fd_link(base, target.name, parent)
                    except OSError as exc:
                        if exc.errno == errno.EEXIST: raise ValueError('destination raced; existing evidence retained') from exc
                        reason = 'hardlink unavailable: ' + str(exc.errno)
                    else:
                        dst = held(os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent))
                        linked = pin_fd(dst); base_after = pin_fd(base); source_after = pin_fd(src)
                        if (source_after != source_pin or not same_contents(linked, source_pin)
                                or not same_contents(base_after, base_pin) or not same_inode(linked, base_pin)
                                or base_after['identity']['nlink'] != base_pin['identity']['nlink'] + 1):
                            raise ValueError('reuse validation changed; partial aliases retained')
                        named_fd(source, src); named_fd(self.base / rel, base); named_fd(target, dst); named_fd(target.parent, parent)
                        os.fsync(parent)
                        self.stats['reused_files'] += 1; self.stats['reused_bytes'] += linked['semantic']['size']
                        self.file_record(rel, 'hardlink_completed_archive', source_after, linked)
                        return
        return self.independent(source, target, rel, reason)

    def finish(self):
        count = self.stats['copied_files'] + self.stats['reused_files']
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        self.stats['manifest_sha256'] = self.manifest_digest.hexdigest()
        if self.stats['manifest_records'] > MAX_RECORDS_PER_FILE * count + 2:
            self.stats['incomplete_extra_records'] = True
        return dict(self.stats)

    def publish_hint(self):
        if self.owner is None: return
        value = {'contract': 'retention_reuse_hint/v2', 'task_dir': str(self.task),
                 'archive': str(self.destination), 'retention_sha256': digest(raw_file(self.destination / 'retention.json')),
                 'manifest_sha256': self.stats['manifest_sha256']}
        with tempfile.NamedTemporaryFile(mode='w', dir=self.owner, prefix='.retention-reuse-', suffix='.tmp', delete=False) as stream:
            json.dump(value, stream); stream.write('\n'); stream.flush(); os.fsync(stream.fileno()); temporary = Path(stream.name)
        os.replace(temporary, self.hint)
        fd = open_path(self.owner, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def close(self):
        try:
            if self.stream is not None and not self.stream.closed:
                try: self.stream.flush(); os.fsync(self.stream.fileno())
                finally: self.stream.close()
        finally:
            if self.lock is not None: os.close(self.lock); self.lock = None
