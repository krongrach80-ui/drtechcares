import React from 'react';
import type { DrTechCaresApiClient, UserAccount } from '@drtechcares/api-client';
import { ClinicalStaffPortalShell } from '../../../src/components/ClinicalStaffPortalShell';

export interface AdminPortalAppProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => void;
  onReturnToKiosk?: () => void;
  onOpenPoseTestMode?: (initialExerciseCode?: string) => void;
  autoOpenEnroll?: boolean;
  onAutoOpenEnrollConsumed?: () => void;
}

export function AdminPortalApp({
  apiClient,
  currentUser,
  onLogout,
  onReturnToKiosk,
  onOpenPoseTestMode,
  autoOpenEnroll,
  onAutoOpenEnrollConsumed,
}: AdminPortalAppProps) {
  return (
    <ClinicalStaffPortalShell
      apiClient={apiClient}
      currentUser={currentUser}
      onLogout={onLogout}
      onReturnToKiosk={onReturnToKiosk}
      onOpenPoseTestMode={onOpenPoseTestMode}
      autoOpenEnroll={autoOpenEnroll}
      onAutoOpenEnrollConsumed={onAutoOpenEnrollConsumed}
    />
  );
}
