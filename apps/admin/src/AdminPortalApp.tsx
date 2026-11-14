import React from 'react';
import type { DrTechCaresApiClient, UserAccount } from '@drtechcares/api-client';
import { ClinicalStaffPortalShell } from '../../../src/components/ClinicalStaffPortalShell';

export interface AdminPortalAppProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => void;
  onReturnToKiosk?: () => void;
  autoOpenEnroll?: boolean;
  onAutoOpenEnrollConsumed?: () => void;
}

export function AdminPortalApp({
  apiClient,
  currentUser,
  onLogout,
  onReturnToKiosk,
  autoOpenEnroll,
  onAutoOpenEnrollConsumed,
}: AdminPortalAppProps) {
  return (
    <ClinicalStaffPortalShell
      apiClient={apiClient}
      currentUser={currentUser}
      onLogout={onLogout}
      onReturnToKiosk={onReturnToKiosk}
      autoOpenEnroll={autoOpenEnroll}
      onAutoOpenEnrollConsumed={onAutoOpenEnrollConsumed}
    />
  );
}
