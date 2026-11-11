const EVO_SOURCE_PATHS = A.evolution_source_paths;
if (A.optimizer != null && typeof A.optimizer !== 'string') {
  throw new Error('optimizer must be greedy or evo (a string)');
}
if (String(A.optimizer || '').trim().toLowerCase() !== 'evo') {
  throw new Error('The EVO worker requires optimizer=evo');
}
if (A.kernel_lane_script) {
  throw new Error('optimizer=evo cannot override kernel_lane_script');
}
if (!['optimize', 'author'].includes(String(A.mode || 'optimize').trim().toLowerCase() || 'optimize')) {
  throw new Error('optimizer=evo currently supports optimize and author lanes, not bakeoff');
}
if (!Array.isArray(EVO_SOURCE_PATHS) || !EVO_SOURCE_PATHS.length ||
    EVO_SOURCE_PATHS.length > 256 || new Set(EVO_SOURCE_PATHS).size !== EVO_SOURCE_PATHS.length ||
    EVO_SOURCE_PATHS.some(sourcePath => typeof sourcePath !== 'string' || !sourcePath ||
      sourcePath.length > 1024 || sourcePath.includes('\\') || sourcePath.startsWith('/') ||
      sourcePath.split('/').some(part => !part || part === '.' || part === '..' || part.toLowerCase() === '.git') ||
      /[^\x20-\x7e]/.test(sourcePath) || EVO_SOURCE_PATHS.some(other => sourcePath.startsWith(other + '/')))) {
  throw new Error('evolution_source_paths must list safe relative source files, not directories');
}
const EVO_HELPER = `${WORKFLOW_DIR}/../evo/archive.py`;
const EVO_ERRORS = [];
let evoArchivePath = '';

function evoQuote(value) {
  return "'" + String(value).replace(/'/g, "'\"'\"'") + "'";
}

function evoCommand(operation, values) {
  return ['python3', EVO_HELPER, operation, '--archive', evoArchivePath, ...values]
    .map(evoQuote).join(' ');
}

async function evoRun(operation, values, prelude = '') {
  const result = await agentT(
    'Execute the following deterministic archive command with Bash exactly as supplied. ' +
    'Do not edit source, repair failures, rerun a successful command, or invent output. ' +
    'This operation is source bookkeeping, not correctness or performance verification. ' +
    'Return its actual exit_code and the complete stdout as a string; put diagnostics in stderr.\n' +
    '```bash\nset -euo pipefail\n' + prelude + evoCommand(operation, values) + '\n```',
    { phase: 'Optimize', label: `evo:${operation}`,
      schema: obj({ exit_code: { type: 'number' }, stdout: { type: 'string' }, stderr: { type: 'string' } },
        ['exit_code', 'stdout']) });
  if (!result || result.exit_code !== 0 || typeof result.stdout !== 'string') {
    throw new Error(`EVO ${operation} did not complete: ${result ? result.stderr || 'command failed' : 'no return'}`);
  }
  const receipt = JSON.parse(result.stdout);
  if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt) ||
      receipt.ok !== true || receipt.command !== operation || receipt.archive !== evoArchivePath) {
    throw new Error(`EVO ${operation} returned an invalid receipt`);
  }
  return receipt;
}

async function evoInitialize() {
  evoArchivePath = `${String(EVAL_DIR).replace(/\/+$/, '')}/evolution_archive`;
  return evoRun('init', ['--canonical', CANONICAL,
    ...EVO_SOURCE_PATHS.flatMap(sourcePath => ['--source-path', sourcePath])]);
}

function evoPlannerInput() {
  return {
    archive: evoArchivePath,
    query_command: evoCommand('query', []),
    source_paths: EVO_SOURCE_PATHS,
    policy: 'Read the archive before selecting sources. Choose parent_id=INCUMBENT, FOUNDER, or an archived entry ID. ' +
      'FOUNDER is the post-preparation starting source, not necessarily ORIGINAL. ' +
      'Retained unfinished/slower code may seed further engineering but is not a qualified result. ' +
      'Use donor_ids only for intended source contributions; explain the contribution and applicability in prompt. ' +
      'Choose sources and investigations from evidence, including weighted and zero-weight cases. ' +
      'Measurements are reported records, not transferable qualifications. Missing evidence stays unknown. ' +
      'The current objective, frozen ORIGINAL, ordinary verifier and final validation remain authoritative.',
  };
}

async function evoPrepare(direction) {
  const workspace = `${direction.out_dir}/workspace`;
  const prelude = `mkdir -p ${evoQuote(direction.out_dir)}\n` +
    ['bash', `${WORKFLOW_DIR}/scripts/materialize_workspace.sh`, '--src', CANONICAL,
      '--dst', workspace, '--shared-root', `${EVAL_DIR}/_shared`, '--link-aiter'].map(evoQuote).join(' ') +
      ` > ${evoQuote(`${direction.out_dir}/materialize.json`)}\n`;
  const donorIds = direction.donor_ids == null ? [] : direction.donor_ids;
  if (!Array.isArray(donorIds) || donorIds.some(donorId => typeof donorId !== 'string')) {
    throw new Error('donor_ids must be a list of archive entry IDs');
  }
  if (direction.parent_id != null && typeof direction.parent_id !== 'string') {
    throw new Error('parent_id must be an archive entry ID');
  }
  const parentId = direction.parent_id || 'INCUMBENT';
  const prepared = await evoRun('prepare', ['--workspace', workspace, '--canonical', CANONICAL,
    '--initialize-index', '--parent', parentId,
    ...donorIds.flatMap(donorId => ['--donor', donorId])], prelude);
  if (!prepared.receipt || prepared.receipt.workspace !== workspace ||
      !/^[a-f0-9]{64}$/.test(prepared.receipt.materialized_canonical_source_id) ||
      prepared.receipt.requested_parent_id !== parentId || !Array.isArray(prepared.receipt.donors) ||
      JSON.stringify(prepared.receipt.donors.map(donor => donor.id)) !== JSON.stringify(donorIds)) {
    throw new Error('prepare receipt does not match the selected workspace, parent and donors');
  }
  return prepared;
}

function evoEngineerInput(direction) {
  return {
    ...evoPlannerInput(),
    parent_id: direction.parent_id || 'INCUMBENT',
    donor_ids: direction.donor_ids || [],
    prepared: direction.evo_prepared,
    custody: 'The workspace is already initialized from the selected parent. Do not rematerialize it, ' +
      'reset its index, or commit a new baseline. The index remains relative to this round\'s canonical. ' +
      'Modify only evolution_source_paths. Include declared new source files with git add -N before git diff. ' +
      'Never change the oracle, benchmark, objective or verification rules. Donor IDs do not prove a combination.',
  };
}

async function evoRecordBatch(clean) {
  const reports = clean.map(result => ({
    id: `r${round}_d${result.d.idx}`,
    parent_id: result.d.parent_id || 'INCUMBENT', donor_ids: result.d.donor_ids || [],
    title: result.d.title, workspace: `${result.d.out_dir}/workspace`,
    patch: result.patch || `${result.d.out_dir}/best_patch.diff`, engineer: result.eng, verifier: result.ver,
  }));
  try {
    const receipt = await evoRun('record-batch', ['--canonical', CANONICAL, '--report-json', JSON.stringify(reports)]);
    if (receipt.has_failures) {
      EVO_ERRORS.push({ round, operation: 'record-batch', results: receipt.results });
    }
    return receipt;
  } catch (error) {
    const message = String(error && error.message || error);
    EVO_ERRORS.push({ round, operation: 'record-batch', message });
    log(message);
    return null;
  }
}

async function evoCheckPatch(direction, patch) {
  try {
    const receipt = await evoRun('check-patch', ['--canonical', CANONICAL, '--patch', patch]);
    if (typeof receipt.patch_object_path !== 'string' || !/^[a-f0-9]{64}$/.test(receipt.patch_sha256) ||
        receipt.patch_object_path !== `${evoArchivePath}/objects/${receipt.patch_sha256}`) {
      throw new Error('check-patch did not return a bound immutable patch');
    }
    return receipt;
  } catch (error) {
    const message = String(error && error.message || error);
    EVO_ERRORS.push({ round, operation: 'check-patch', id: direction.id, message });
    log(message);
    return null;
  }
}
