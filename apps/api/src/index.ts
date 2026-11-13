import pino from 'pino';
import { createDrTechCaresApiApp } from './app';
import { ClinicalDatabase } from './db';

export { createDrTechCaresApiApp } from './app';
export * from './db';

const isDirectRun =
  process.env.API_DIRECT_RUN === '1' ||
  (typeof process !== 'undefined' &&
    process.argv[1] &&
    (process.argv[1].includes('apps\\api\\src\\index') || process.argv[1].includes('apps/api/src/index')));

if (isDirectRun) {
  const logger = pino({ name: 'drtechcares-api' });
  const port = Number(process.env.PORT ?? 8787);
  const db = new ClinicalDatabase({ persistToDisk: true });
  const app = createDrTechCaresApiApp(db);
  app.listen(port, () => {
    logger.info({ port }, 'DRTECHCARES Production API server listening');
  });
}
