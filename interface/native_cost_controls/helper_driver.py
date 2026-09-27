# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate an approved helper through the caller's existing native tool hooks.

This module executes no command and sends no request. A trusted runtime supplies
an exact source-bound task. The model request cannot register its own binding.
The runtime must retain its original permission decision before calling
``before_bash`` and must supply the actual native result to ``capture_bash``.
"""
import fcntl
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path

from .native_envelope import qualified_initial_messages, qualified_notice_text
from .system_envelope import normalized_system

ROLES = {'clock_reader', 'warm_start_resolver', 'storage_reclaim',
         'citation_writer', 'experience_writer'}
LOCAL_MODEL = 'local-deterministic-helper-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_messages(messages):
    """Ignore only the native transport's movable five-minute cache markers."""
    value = json.loads(canonical(messages))
    for message in value:
        content = message.get('content')
        # The qualified CLI serializes this notice as a text-block list on
        # the first request, then as the identical string on continuation.
        if message.get('role') == 'system' and qualified_notice_text(content):
            content = [{'type': 'text', 'text': content}]
            message['content'] = content
        if not isinstance(content, list):
            continue
        for block in content:
            if 'cache_control' in block:
                marker = block.pop('cache_control')
                check(marker in ({'type': 'ephemeral'}, {'type': 'ephemeral', 'ttl': '5m'}),
                      'unsupported_native_cache_marker')
    return value


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Unsupported(Exception):
    def __init__(self, code, may_have_executed=False):
        self.code, self.may_have_executed = code, may_have_executed
        super().__init__(code)


def check(condition, code, executed=False):
    if not condition:
        raise Unsupported(code, executed)


def parse_object(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Non-finite JSON')
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    check(isinstance(value, dict), 'stdout_not_object', True)
    canonical(value)
    return value


def project(binding, native):
    check(isinstance(native, dict), 'missing_native_result', True)
    check(native.get('interrupted') is False, 'interrupted_result', True)
    check(not any(native.get(k) for k in ['backgroundTaskId', 'background_task_id', 'truncated', 'isImage']),
          'unsupported_native_result', True)
    text = native.get('stdout')
    check(isinstance(text, str) and len(text.encode()) <= 65536, 'stdout_missing_or_large', True)
    check(not re.search(r'(?i)(output truncated|output exceeded|persisted.output|running in background)', text),
          'incomplete_stdout', True)
    preamble = binding.get('native_preamble')
    if preamble and text.startswith(preamble):
        text = text[len(preamble):]
        check(not text.startswith(preamble), 'repeated_native_preamble', True)
    role = binding['role']
    if role == 'clock_reader':
        check(re.fullmatch(r'[0-9]+\s*', text) is not None, 'clock_not_integer', True)
        epoch = int(text)
        check(epoch <= 9007199254740991, 'clock_not_exact_JS_integer', True)
        return {'epoch': epoch}
    if role == 'storage_reclaim':
        check(text.rstrip().endswith(binding['completion_marker']), 'reclaim_marker_missing', True)
        return {'ok': True, 'note': 'reclaimed'}
    value = parse_object(text)
    if role == 'citation_writer':
        count = value.get('citations', 0)
        check(type(count) is int and count >= 0, 'citation_count_invalid', True)
        return {'filed': count}
    return value


def json_values_equal(left, right):
    """Compare finite JSON values without numeric spelling or boolean coercion."""
    numeric = (int, float)
    if type(left) in numeric and type(right) in numeric:
        if (type(left) is float and not math.isfinite(left)) or (type(right) is float and not math.isfinite(right)):
            return False
        return left == right
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (all(type(key) is str for key in left) and all(type(key) is str for key in right)
                and set(left) == set(right) and all(json_values_equal(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(json_values_equal(a, b) for a, b in zip(left, right))
    return type(left) in (str, bool, type(None)) and left == right


class LocalHelperDriver:
    """A trusted caller supplies bindings. No API endpoint can register one."""
    def __init__(self, directory, approved_binding):
        check(isinstance(approved_binding, dict), 'helper_binding_required')
        check(re.fullmatch(r'[0-9a-f]{64}', approved_binding.get('operation_id', '')) is not None,
              'invalid_operation_id')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.approved = approved_binding
        self.path = self.directory / (approved_binding['operation_id'] + '.json')
        self.lock = self.path.with_suffix('.lock')
        self.dispatch_attested = False
        self.local_records = []

    def attest_dispatch(self, actual_script, expected_script):
        check(actual_script == expected_script, 'untrusted_native_dispatch')
        self.dispatch_attested = True

    def verify(self, binding):
        check(binding == self.approved, 'binding_differs')
        check(binding['role'] in ROLES and binding.get('gate') is True, 'role_or_gate_unsupported')
        workspace = Path(binding['workspace'])
        check(not workspace.is_symlink() and str(workspace.resolve()) == binding['workspace'], 'workspace_changed')
        st = workspace.stat()
        check([st.st_dev, st.st_ino] == binding['workspace_identity'], 'workspace_identity_changed')
        for name, expected in binding['source_bindings'].items():
            check(not Path(name).is_symlink() and file_digest(name) == expected, 'script_changed')
        check(digest(binding['schema']) == binding['schema_sha256'], 'schema_binding_changed')
        check(hashlib.sha256(binding['command'].encode()).hexdigest() == binding['command_sha256'], 'command_binding_changed')

    def semantic_digest(self):
        return digest({k: v for k, v in self.approved.items() if k != 'native_session'})

    def bash_input(self):
        return {'command': self.approved['command'], 'timeout': 120000,
                'run_in_background': False, 'description': 'Execute a bound local administrative helper'}

    @contextmanager
    def state(self):
        with self.lock.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(self.path.read_bytes()) if self.path.exists() else {
                'stage': 'NEW', 'binding_sha256': self.semantic_digest(), 'emissions': 0,
                'bash_emissions': 0, 'accepted_values': [], 'accepted_ids': [], 'acknowledgement_errors': [], 'origin': 'local_deterministic_driver',
                'remote_model_inference': False, 'remote_request_id': None, 'remote_usage': None}
            check(value['binding_sha256'] == self.semantic_digest(), 'operation_binding_changed', value['stage'] != 'NEW')
            try:
                yield value
            finally:
                atomic(self.path, value)

    def response(self, body, binding):
        try:
            return self._response(body, binding)
        except Unsupported as error:
            # A later binding failure must never reopen an unrestricted fallback path.
            may_have_executed = error.may_have_executed
            if self.path.exists():
                with self.state() as state:
                    if state['stage'] not in ('NEW', 'PROVIDER_ONLY'):
                        may_have_executed = True
                        state.update(stage='UNKNOWN', continuation_conflict=error.code)
            raise Unsupported(error.code, may_have_executed) from None

    def fallback_decision(self, original_request_bytes, error):
        check(isinstance(original_request_bytes, bytes), 'fallback_request_bytes_required')
        with self.state() as state:
            if state['stage'] == 'NEW':
                state['stage'] = 'UNKNOWN' if error.may_have_executed else 'PROVIDER_ONLY'
            elif state['stage'] != 'PROVIDER_ONLY':
                state.update(stage='UNKNOWN', continuation_conflict=error.code)
            provider_only = state['stage'] == 'PROVIDER_ONLY'
        return {'route': 'unchanged_provider_passthrough' if provider_only else 'original_failure_path',
                'original_request_bytes': original_request_bytes,
                'remote_request_sent': False, 'reason': error.code,
                'caller_must_forward_unchanged': provider_only}

    def _response(self, body, binding):
        self.verify(binding)
        check(self.dispatch_attested, 'dispatch_not_attested')
        check(isinstance(body, dict), 'native_request_object_required')
        try:
            user_id = json.loads(body.get('metadata', {}).get('user_id', '{}'))
        except (ValueError, TypeError, AttributeError):
            raise Unsupported('native_metadata_invalid') from None
        check(isinstance(user_id, dict), 'native_metadata_invalid')
        check(isinstance(body.get('tools'), list)
              and all(isinstance(tool, dict) for tool in body['tools']), 'native_tools_invalid')
        check(isinstance(body.get('messages'), list)
              and all(isinstance(message, dict) and isinstance(message.get('content'), (str, list))
                      and (not isinstance(message.get('content'), list)
                           or all(isinstance(block, dict) for block in message['content']))
                      for message in body['messages']), 'native_messages_invalid')
        check(user_id.get('session_id') == binding['native_session'], 'native_session_changed')
        schemas = [t['input_schema'] for t in body.get('tools', []) if t.get('name') == 'StructuredOutput']
        check(len(schemas) == 1 and digest(schemas[0]) == binding['schema_sha256'], 'native_schema_changed')
        texts = [b.get('text') for m in body.get('messages', []) for b in
                 (m.get('content', []) if isinstance(m.get('content'), list) else []) if b.get('type') == 'text']
        check(binding['prompt'] in texts, 'native_prompt_changed')
        messages = normalized_messages(body['messages'])
        messages_sha256 = digest(messages)
        system = normalized_system(body.get('system', []), workspace=binding['workspace'],
            shell=binding.get('native_shell') or os.environ.get('SHELL') or 'unknown')
        check(system is not None, 'unqualified_system_context')
        system_sha256 = digest(system)
        with self.state() as state:
            check(state['stage'] in ['NEW', 'RESULT', 'COMPLETE'], 'outcome_unknown_or_response_pending', True)
            if state['stage'] == 'NEW':
                check(qualified_initial_messages(messages, binding['prompt']), 'unqualified_initial_context')
            else:
                check(state['initial_system_sha256'] == system_sha256, 'native_system_context_changed', True)
            if state['stage'] == 'RESULT':
                count = state['initial_message_count']
                native = state['native_result']
                expected_suffix = [
                    {'role': 'assistant', 'content': [state['bash_block']]},
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': state['bash_id'],
                        'content': native.get('stdout'), 'is_error': False}]},
                ]
                valid = (native.get('stderr', '') == '' and len(messages) == count + 2
                         and digest(messages[:count]) == state['initial_messages_sha256']
                         and canonical(messages[count:]) == canonical(expected_suffix))
                if not valid:
                    state.update(stage='UNKNOWN', continuation_conflict='native_continuation_changed')
                    raise Unsupported('native_continuation_changed', True)
                state['result_messages_sha256'] = messages_sha256
            elif state['stage'] == 'COMPLETE':
                if messages_sha256 not in {state['initial_messages_sha256'], state.get('result_messages_sha256')}:
                    state.update(stage='UNKNOWN', continuation_conflict='native_replay_continuation_changed')
                    raise Unsupported('native_replay_continuation_changed', True)
            state['emissions'] += 1
            token = 'local_helper_' + binding['operation_id'] + '_' + str(state['emissions'])
            if state['stage'] == 'NEW':
                state.update(stage='MAY_HAVE_EXECUTED', bash_id=token, bash_emissions=state['bash_emissions'] + 1,
                             bash_native_session=binding['native_session'])
                block = {'type': 'tool_use', 'id': token, 'name': 'Bash',
                         'input': self.bash_input()}
                state.update(initial_message_count=len(messages), initial_messages_sha256=messages_sha256,
                             initial_system_sha256=system_sha256, bash_block=block)
            else:
                state['structured_id'] = token
                state.setdefault('structured_ids', []).append(token)
                state['stage'] = 'STRUCTURED_PENDING'
                block = {'type': 'tool_use', 'id': token, 'name': 'StructuredOutput', 'input': state['value']}
            self.local_records.append({'response_id': token, 'tool_use_id': token, 'origin': state['origin'],
                'producer': LOCAL_MODEL, 'remote_model_inference': False, 'remote_request_id': None,
                'remote_usage': None, 'request_sha256': digest(body), 'operation_id': binding['operation_id']})
        return block

    def before_bash(self, tool_id, tool_input):
        self.verify(self.approved)
        with self.state() as state:
            check(state['stage'] == 'MAY_HAVE_EXECUTED' and state['bash_id'] == tool_id,
                  'unregistered_bash_call', True)
            check(digest(tool_input) == digest(self.bash_input()), 'bash_arguments_changed', True)
            check(not state.get('bash_started'), 'duplicate_bash_execution', True)
            state['bash_started'] = True

    def capture_bash(self, tool_id, native_result, failed=False):
        with self.state() as state:
            check(state.get('bash_id') == tool_id and state.get('bash_started'), 'result_identity_changed', True)
            if 'native_result_sha256' in state:
                if state['native_result_sha256'] != digest(native_result) or state['native_result_failed'] != failed:
                    state.update(stage='UNKNOWN', continuation_conflict='conflicting_native_result')
                    raise Unsupported('conflicting_native_result', True)
                return
            check(state['stage'] == 'MAY_HAVE_EXECUTED', 'result_state_changed', True)
            state['native_result'] = native_result
            state['native_result_sha256'] = digest(native_result)
            state['native_result_failed'] = failed
            preamble = self.approved.get('native_preamble')
            state['recognized_native_preamble'] = bool(preamble and isinstance(native_result, dict)
                and isinstance(native_result.get('stdout'), str) and native_result['stdout'].startswith(preamble))
            if failed:
                state['stage'] = 'UNKNOWN'
                return
            try:
                state['value'] = project(self.approved, native_result)
                state['stage'] = 'RESULT'
            except (ValueError, Unsupported) as error:
                state['stage'] = 'UNKNOWN'
                state['unsupported_code'] = error.code if isinstance(error, Unsupported) else 'invalid_stdout_JSON'

    def accept(self, tool_id, value):
        with self.state() as state:
            errors = state.setdefault('acknowledgement_errors', [])
            check(isinstance(errors, list) and not errors and state['stage'] != 'UNKNOWN',
                  'acknowledgement_error_latched', True)
            accepted_ids = state.setdefault('accepted_ids', [])
            accepted_values = state.get('accepted_values', [])
            valid = (isinstance(accepted_ids, list) and isinstance(accepted_values, list)
                     and len(accepted_ids) == len(accepted_values)
                     and tool_id in state.get('structured_ids', [])
                     and json_values_equal(value, state.get('value')))
            if not valid:
                state['stage'] = 'UNKNOWN'
                errors.append({'code': 'accepted_value_changed',
                    'tool_id': tool_id if isinstance(tool_id, str) else type(tool_id).__name__})
                raise Unsupported('accepted_value_changed', True)
            if tool_id in accepted_ids:
                index = accepted_ids.index(tool_id)
                if not json_values_equal(value, accepted_values[index]):
                    state['stage'] = 'UNKNOWN'
                    errors.append({'code': 'accepted_value_changed', 'tool_id': tool_id})
                    raise Unsupported('accepted_value_changed', True)
            else:
                accepted_ids.append(tool_id)
                accepted_values.append(value)
            state['accepted_values'] = accepted_values
            state['stage'] = ('ACKNOWLEDGED' if set(state.get('structured_ids', [])) <= set(accepted_ids)
                              else 'STRUCTURED_PENDING')

    def confirm_completion(self, value, evidence):
        """Require a full native terminal value before a result becomes replayable."""
        with self.state() as state:
            valid = (state['stage'] in ('ACKNOWLEDGED', 'COMPLETE')
                     and json_values_equal(value, state.get('value'))
                     and isinstance(evidence, dict)
                     and evidence.get('status') == 'done'
                     and evidence.get('native_session') == self.approved['native_session']
                     and evidence.get('operation_id') == self.approved['operation_id']
                     and evidence.get('source') == 'native_workflow_journal'
                     and isinstance(evidence.get('entry_sha256'), str)
                     and re.fullmatch(r'[0-9a-f]{64}', evidence['entry_sha256']) is not None)
            if not valid:
                state.update(stage='UNKNOWN', continuation_conflict='native_terminal_outcome_changed')
                raise Unsupported('native_terminal_outcome_changed', True)
            state.setdefault('native_completion_evidence', []).append(evidence)
            state['stage'] = 'COMPLETE'

    def reject_completion(self, reason):
        with self.state() as state:
            state.update(stage='UNKNOWN', continuation_conflict=reason)

    def reject_structured(self, tool_id, error):
        with self.state() as state:
            check(tool_id in state.get('structured_ids', []), 'rejection_identity_changed', True)
            state.update(stage='UNKNOWN', native_schema_error=error)
