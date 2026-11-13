import { build } from 'esbuild';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const rootDir = process.cwd();
const outFile = path.join(rootDir, '.vite', 'api-server.mjs');
fs.mkdirSync(path.dirname(outFile), { recursive: true });

await build({
  entryPoints: [path.join(rootDir, 'apps', 'api', 'src', 'index.ts')],
  bundle: true,
  platform: 'node',
  format: 'esm',
  outfile: outFile,
  external: ['express', 'cors', 'helmet', 'express-rate-limit', 'zod', 'pino'],
});

const isWindows = process.platform === 'win32';
const npxCmd = isWindows ? 'npx.cmd' : 'npx';
const nodeCmd = process.execPath;

const apiProc = spawn(nodeCmd, [outFile], {
  stdio: 'inherit',
  env: { ...process.env, PORT: process.env.PORT ?? '8787', API_DIRECT_RUN: '1' },
});

const viteProc = spawn(
  npxCmd,
  ['vite', '--port', '5173', '--host', '0.0.0.0'],
  {
    stdio: 'inherit',
    shell: isWindows,
    env: process.env,
  },
);

let shuttingDown = false;
function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  if (!apiProc.killed) apiProc.kill();
  if (!viteProc.killed) viteProc.kill();
  process.exit(code);
}

apiProc.on('exit', (code) => {
  if (!shuttingDown) {
    console.error(`[dev-all] API exited unexpectedly with code ${code ?? 1}`);
    shutdown(code ?? 1);
  }
});

viteProc.on('exit', (code) => {
  if (!shuttingDown) {
    console.error(`[dev-all] Vite exited unexpectedly with code ${code ?? 1}`);
    shutdown(code ?? 1);
  }
});

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
