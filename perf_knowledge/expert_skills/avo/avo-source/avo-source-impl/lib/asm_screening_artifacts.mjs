import {
  copyFileSync, mkdirSync, readdirSync, readFileSync, realpathSync, writeFileSync,
} from 'node:fs';
import {
  extname, join, relative, resolve, sep,
} from 'node:path';

import { sha256File, verifyFormalArtifacts } from './formal_artifacts.mjs';

const finitePositive = (value) => Number.isFinite(value) && value > 0;
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
};
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const populationCv = (values) => {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const variance = mean(values.map((value) => (value - avg) ** 2));
  return avg > 0 ? Math.sqrt(variance) / avg : 0;
};
const medianLogEstimate = (values) => Math.exp(median(values.map(Math.log)));

function screeningLabel(languageMode) {
  const language = String(languageMode || '').toUpperCase();
  return language ? `${language} screening` : 'generated-persistent screening';
}

function assertInsideWorkspace(filePath, workspace, languageMode) {
  const root = realpathSync(workspace);
  const file = realpathSync(filePath);
  const rel = relative(root, file);
  if (rel === '..' || rel.startsWith(`..${sep}`)) {
    throw new Error(`${screeningLabel(languageMode)} report is outside its verify workspace: ${file}`);
  }
  return file;
}

// The verify workspace is torn down once the run moves on, so an archive holding
// only the report JSONs points at a candidate that no longer exists and the evidence
// can never be re-checked after the fact. Copying the measured artifact in beside its
// reports makes the archive self-contained. The baseline and the capture are left
// out on purpose: both live outside the workspace and outlive the run.
function archiveMeasuredCandidates(copiedReports, {
  evidenceDir, workspace, languageMode,
}) {
  const archived = [];
  const errors = [];
  const seenPaths = new Set();
  const seenDigests = new Set();
  for (const reportPath of copiedReports) {
    let candidatePath = '';
    try {
      candidatePath = String(JSON.parse(
        readFileSync(reportPath, 'utf8')).artifact?.candidate_path || '');
    } catch {
      continue; // A malformed report is the validator's problem, not the archiver's.
    }
    if (!candidatePath || seenPaths.has(candidatePath)) continue;
    seenPaths.add(candidatePath);
    try {
      // This path comes from the agent and nothing has validated it yet, so it is
      // confined to the workspace before a single byte is read or copied.
      const source = assertInsideWorkspace(candidatePath, workspace, languageMode);
      const sha256 = sha256File(source);
      if (seenDigests.has(sha256)) continue;
      const destination = join(evidenceDir, `candidate_${sha256}${extname(source)}`);
      copyFileSync(source, destination);
      // An archive holding the wrong bytes is worse than no archive at all: it would
      // let a later audit "confirm" evidence that was never actually measured.
      const stored = sha256File(destination);
      if (stored !== sha256) {
        throw new Error(
          `archived copy hashes ${stored} but the source hashes ${sha256}`);
      }
      seenDigests.add(sha256);
      archived.push({ source_path: source, archived_path: destination, sha256 });
    } catch (error) {
      errors.push({ candidate_path: candidatePath, error: error.message });
    }
  }
  return { archived, errors };
}

function archiveScreeningInputs(reportPaths, {
  archiveDir, sessionPrefix, workspace, languageMode,
} = {}) {
  if (!archiveDir || !sessionPrefix) return null;
  const safeSession = String(sessionPrefix).replace(/[^a-zA-Z0-9._-]/g, '_');
  const evidenceDir = join(resolve(archiveDir), safeSession);
  mkdirSync(evidenceDir, { recursive: true });
  const submittedPaths = Array.isArray(reportPaths) ? reportPaths.map(String) : [];
  const copiedReports = [];
  const copyErrors = [];
  submittedPaths.forEach((reportPath, index) => {
    try {
      const source = assertInsideWorkspace(reportPath, workspace, languageMode);
      const destination = join(evidenceDir, `report_r${index + 1}.json`);
      copyFileSync(source, destination);
      copiedReports.push(destination);
    } catch (error) {
      copyErrors.push({ report_path: reportPath, error: error.message });
    }
  });
  const candidates = archiveMeasuredCandidates(copiedReports, {
    evidenceDir, workspace, languageMode,
  });
  return {
    evidenceDir,
    submittedPaths,
    copiedReports,
    copyErrors,
    archivedCandidates: candidates.archived,
    candidateCopyErrors: candidates.errors,
  };
}

// The manifest is a DIAGNOSTIC artifact. It must never influence the verdict, so
// a failure to write it can neither reject valid evidence nor escape into the
// orchestrator's fan-out (an escaping rejection tears down sibling worktrees).
function writeScreeningManifest(archive, payload) {
  if (!archive) return;
  try {
    writeFileSync(join(archive.evidenceDir, 'manifest.json'), `${JSON.stringify({
      schema_version: 1,
      submitted_report_paths: archive.submittedPaths,
      archived_report_paths: archive.copiedReports,
      copy_errors: archive.copyErrors,
      archived_candidates: archive.archivedCandidates,
      candidate_copy_errors: archive.candidateCopyErrors,
      ...payload,
    }, null, 2)}\n`);
  } catch { /* diagnostic only */ }
}

export function recoverPersistentScreeningReports(workspace, sessionPrefix) {
  if (!workspace || !sessionPrefix) return [];
  const escaped = String(sessionPrefix).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`^\\.avo_screening_${escaped}_r(\\d+)\\.json$`);
  try {
    return readdirSync(workspace)
      .map((name) => ({ name, match: name.match(pattern) }))
      .filter(({ match }) => !!match)
      .sort((left, right) => Number(left.match[1]) - Number(right.match[1]))
      .map(({ name }) => join(resolve(workspace), name));
  } catch {
    return [];
  }
}

function sameIdentity(left, right) {
  return left.baseline_sha256 === right.baseline_sha256
    && left.candidate_sha256 === right.candidate_sha256
    && left.capture_identity_sha256 === right.capture_identity_sha256
    && resolve(left.baseline_path) === resolve(right.baseline_path)
    && resolve(left.candidate_path) === resolve(right.candidate_path)
    && resolve(left.capture_path) === resolve(right.capture_path);
}

function parseReport(reportPath, expectedSessionId, workspace, languageMode) {
  const path = assertInsideWorkspace(reportPath, workspace, languageMode);
  const report = JSON.parse(readFileSync(path, 'utf8'));
  if (report.schema_version !== 1
      || report.session_id !== expectedSessionId
      || report.protocol !== 'generated_persistent_reciprocal_role_swap'
      || report.screening_only !== true
      || report.status !== 'measured'
      || report.measurement_contract?.performance_backend !== 'generated-persistent'
      || report.statistics?.contaminated !== false) {
    throw new Error(`invalid generated-persistent screening report: ${path}`);
  }
  if (!Array.isArray(report.blocks) || report.blocks.length < 2) {
    throw new Error(`${screeningLabel(languageMode)} report has insufficient paired blocks: ${path}`);
  }
  const speedups = report.blocks.map((block) => Number(block.speedup));
  if (!speedups.every(finitePositive)) {
    throw new Error(`${screeningLabel(languageMode)} report contains invalid paired speedup: ${path}`);
  }
  const estimate = medianLogEstimate(speedups);
  const reported = Number(report.statistics.speedup_estimate);
  if (!finitePositive(reported)
      || Math.abs(estimate - reported) > Math.max(1e-12, Math.abs(estimate) * 1e-9)) {
    throw new Error(`${screeningLabel(languageMode)} report statistic does not match its paired blocks: ${path}`);
  }
  return { path, report, speedups };
}

export function ingestGeneratedPersistentScreeningReports(reportPaths, {
  sessionPrefix,
  expectedRepeats,
  workspace,
  languageMode,
} = {}) {
  if (!sessionPrefix || !workspace || !Number.isInteger(expectedRepeats)
      || expectedRepeats < 1 || !Array.isArray(reportPaths)
      || reportPaths.length !== expectedRepeats) {
    throw new Error(`${screeningLabel(languageMode)} evidence count/session/workspace is invalid`);
  }
  const parsed = reportPaths.map((path, index) => parseReport(
    path, `${sessionPrefix}_r${index + 1}`, workspace, languageMode));
  if (new Set(parsed.map(({ path }) => path)).size !== parsed.length) {
    throw new Error(`${screeningLabel(languageMode)} report path was reused across repeats`);
  }

  const first = parsed[0].report;
  const verifiedArtifact = verifyFormalArtifacts(first.artifact, parsed[0].path);
  for (const { report, path } of parsed.slice(1)) {
    if (!sameIdentity(first.artifact, report.artifact)) {
      throw new Error(`${screeningLabel(languageMode)} artifacts changed across repeats: ${path}`);
    }
    if (report.device?.physical_gpu_uuid !== first.device?.physical_gpu_uuid
        || report.device?.physical_gpu_bdf !== first.device?.physical_gpu_bdf) {
      throw new Error(`${screeningLabel(languageMode)} device changed across repeats: ${path}`);
    }
  }

  const speedups = parsed.flatMap(({ speedups: values }) => values);
  const baselineUs = parsed.map(({ report }) => Number(
    report.statistics.baseline_us_median));
  const candidateUs = parsed.map(({ report }) => Number(
    report.statistics.candidate_us_median));
  if (![...baselineUs, ...candidateUs].every(finitePositive)) {
    throw new Error(`${screeningLabel(languageMode)} report contains invalid paired latency`);
  }
  const verifiedGeomean = medianLogEstimate(speedups);
  return {
    verified_geomean: verifiedGeomean,
    verified_arithmetic: mean(speedups),
    cv: populationCv(speedups),
    protocol: 'generated_persistent_reciprocal_role_swap',
    infrastructure_status: 'healthy',
    screening_only: true,
    benchmark_report: parsed[0].path,
    benchmark_reports: parsed.map(({ path }) => path),
    per_case: [{
      name: 'captured_dispatch',
      baseline_ms: median(baselineUs) / 1000,
      optimized_ms: median(candidateUs) / 1000,
      speedup: verifiedGeomean,
    }],
    artifact: verifiedArtifact,
  };
}

export function authoritativeGeneratedPersistentVerification(agentResult, options) {
  if (!agentResult) {
    const recovered = recoverPersistentScreeningReports(
      options?.workspace, options?.sessionPrefix);
    let partialArchive = null;
    try {
      partialArchive = archiveScreeningInputs(recovered, options);
    } catch { /* diagnostic preservation is best-effort */ }
    writeScreeningManifest(partialArchive, {
      status: 'agent_outage',
      language_mode: options?.languageMode || '',
      session_prefix: options?.sessionPrefix || '',
      expected_repeats: options?.expectedRepeats || 0,
      recovered_repeats: recovered.length,
    });
    return {
      status: 'agent_outage',
      correctness: 'unknown',
      infrastructure_status: 'unreported',
      screening_only: true,
      benchmark_reports: recovered,
      partial_report_count: recovered.length,
      expected_report_count: options?.expectedRepeats || 0,
      agent_outage: true,
      screening_evidence_dir: partialArchive?.evidenceDir,
      notes: 'Verifier agent returned no contract; partial generated-persistent evidence was preserved.',
    };
  }
  let archive = null;
  try {
    archive = archiveScreeningInputs(agentResult?.benchmark_reports, options);
  } catch {
    // Evidence validation remains fail-closed even if the diagnostic archive is
    // temporarily unwritable. The validation error is still returned below.
  }
  if (!agentResult || agentResult.status !== 'ok'
      || agentResult.correctness !== 'pass') {
    writeScreeningManifest(archive, {
      status: 'agent_rejected',
      language_mode: options?.languageMode || '',
      session_prefix: options?.sessionPrefix || '',
      agent_status: agentResult?.status || 'missing',
      correctness: agentResult?.correctness || 'missing',
    });
    return archive
      ? { ...agentResult, screening_evidence_dir: archive.evidenceDir }
      : agentResult;
  }
  let evidence = null;
  try {
    evidence = ingestGeneratedPersistentScreeningReports(
      agentResult.benchmark_reports, options);
  } catch (error) {
    // The harness's OWN verdict, before this rejection overwrites it. A rejected
    // report set means "unusable evidence", which is most often the worker
    // submitting the wrong session/count/artifact rather than a broken rig; only
    // the harness can say the rig itself could not measure. Callers need the two
    // apart, because "abort the run" is the right answer to only one of them.
    const reported = String(agentResult.infrastructure_status || '').toLowerCase();
    const component = String(error.artifactComponent || '');
    writeScreeningManifest(archive, {
      status: 'rejected',
      language_mode: options?.languageMode || '',
      session_prefix: options?.sessionPrefix || '',
      rejection_reason: error.message,
      rejected_artifact: component || 'unclassified',
      reported_infrastructure_status: reported || 'unreported',
    });
    return {
      ...agentResult,
      status: 'failed',
      verified_geomean: 0,
      verified_arithmetic: 0,
      infrastructure_status: 'infrastructure_unhealthy',
      reported_infrastructure_status: reported || 'unreported',
      screening_only: true,
      evidence_rejection: error.message,
      evidence_rejection_component: component,
      screening_evidence_dir: archive?.evidenceDir,
      notes: `${screeningLabel(options?.languageMode)} evidence rejected: ${error.message}`,
    };
  }
  writeScreeningManifest(archive, {
    status: 'accepted',
    language_mode: options?.languageMode || '',
    session_prefix: options?.sessionPrefix || '',
    verified_geomean: evidence.verified_geomean,
  });
  return {
    ...agentResult,
    ...evidence,
    screening_evidence_dir: archive?.evidenceDir,
    variance_note: 'Recomputed from generated-persistent reciprocal paired blocks.',
    notes: `${agentResult.notes || ''} Authoritative ${
      String(options?.languageMode || '').toUpperCase() || 'generated-persistent'
    } speedup ignores BASELINE_PER_CASE.`.trim(),
  };
}

// Backward-compatible ASM API retained for callers and tests outside avo-workflow.
export const ingestAsmScreeningReports = ingestGeneratedPersistentScreeningReports;
export const authoritativeAsmVerification = authoritativeGeneratedPersistentVerification;
