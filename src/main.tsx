import React from 'react';
import { createRoot } from 'react-dom/client';
import { ClinicalI18nProvider } from '@drtechcares/clinical-i18n';
import App from './App';
import './index.css';
import { installGlobalTactileMotion } from './lib/globalTactileMotion';

installGlobalTactileMotion();

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ClinicalI18nProvider initialLang="th">
      <App />
    </ClinicalI18nProvider>
  </React.StrictMode>,
);

