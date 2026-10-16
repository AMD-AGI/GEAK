const EXPECTED_GFX = String(A.expected_gfx || '').trim().toLowerCase();
const EXPECTED_TARGET = String(A.expected_target || '').trim().toLowerCase();
const EXPECTED_DEVICE_NAME = String(A.expected_device_name || '').trim();
const EXPECTED_PHYSICAL_CU_COUNT = Number(A.expected_physical_cu_count || 0);
if (EXPECTED_TARGET === 'r9700' && EXPECTED_GFX !== 'gfx1201') {
  throw new Error('expected_target=r9700 requires expected_gfx=gfx1201');
}
if (['mi350x', 'mi355x'].includes(EXPECTED_TARGET) && EXPECTED_GFX !== 'gfx950') {
  throw new Error(`expected_target=${EXPECTED_TARGET} requires expected_gfx=gfx950`);
}
if (EXPECTED_GFX === 'gfx1200' || EXPECTED_TARGET === 'gfx1200') {
  throw new Error('gfx1200 is not supported: this workflow is hardware-validated only on R9700 / gfx1201');
}
