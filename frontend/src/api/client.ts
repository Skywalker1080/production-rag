import type { HealthResponse, UploadResponse, JobStatus, AskRequest, AskResponse, StatsResponse } from '../types';

const API_BASE = '/api';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>(`${API_BASE.replace('/api', '')}/health`);
}

export async function uploadPDF(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return fetchJSON<JobStatus>(`${API_BASE}/jobs/${jobId}`);
}

export async function askQuestion(request: AskRequest): Promise<AskResponse> {
  return fetchJSON<AskResponse>(`${API_BASE}/ask`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getStats(): Promise<StatsResponse> {
  return fetchJSON<StatsResponse>(`${API_BASE}/stats`);
}

export async function deleteDocuments(): Promise<{ cleared: boolean }> {
  return fetchJSON<{ cleared: boolean }>(`${API_BASE}/documents`, {
    method: 'DELETE',
  });
}
