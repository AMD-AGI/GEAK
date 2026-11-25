import {
  copyFileSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync,
} from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { verifyFormalArtifacts } from './formal_artifacts.mjs';

/**
 * Preserve the final formal report and its candidate outside the ephemeral
 * validation workspace. The archived copy remains independently verifiable
 * after cleanup removes validation_workspace/.
 */
export function archiveFormalReport(reportPath, archiveDirectory) {
  const sourceReportPath = resolve(reportPath);
  const archiveDir = resolve(archiveDirectory);
  const sourceReport = JSON.parse(readFileSync(sourceReportPath, 'utf8'));
  if (Number(sourceReport.schema_version) !== 2
      || !sourceReport.artifact || !sourceReport.artifact.candidate_path) {
    throw new Error('cannot archive incomplete formal report');
  }
  const policySha256 = sourceReport.correctness?.validation_policy_sha256 || '';
  const sourceArtifact = verifyFormalArtifacts(
    sourceReport.artifact, sourceReportPath, policySha256);

  mkdirSync(archiveDir, { recursive: true });
  const candidateSuffix = extname(sourceArtifact.candidate_path) || '.bin';
  const candidatePath = join(
    archiveDir, `candidate-${sourceArtifact.candidate_sha256}${candidateSuffix}`);
  const reportArchivePath = join(archiveDir, 'realcap_paired_result.json');
  const nonce = `${process.pid}-${Date.now()}`;
  const candidateTemp = `${candidatePath}.tmp-${nonce}`;
  const reportTemp = `${reportArchivePath}.tmp-${nonce}`;

  try {
    copyFileSync(sourceArtifact.candidate_path, candidateTemp);
    verifyFormalArtifacts({
      ...sourceArtifact,
      candidate_path: candidateTemp,
    }, '', policySha256);
    renameSync(candidateTemp, candidatePath);

    const artifact = verifyFormalArtifacts({
      ...sourceArtifact,
      candidate_path: candidatePath,
    }, reportArchivePath, policySha256);
    const archivedReport = {
      ...sourceReport,
      artifact,
      archive: {
        source_report_path: sourceReportPath,
        archived_at: new Date().toISOString(),
      },
    };
    writeFileSync(reportTemp, `${JSON.stringify(archivedReport, null, 2)}\n`, 'utf8');
    renameSync(reportTemp, reportArchivePath);
    return { reportPath: reportArchivePath, report: archivedReport };
  } finally {
    rmSync(candidateTemp, { force: true });
    rmSync(reportTemp, { force: true });
  }
}
