export interface HealthResponse {
  status: string;
  qdrant_ok: boolean;
  qdrant_url: string;
  llm_model: string;
  embed_model: string;
  chunks: number;
}

export interface UploadResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  filename: string;
}

export interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  filename: string;
  progress?: number;
  error?: string;
}

export interface Source {
  content: string;
  metadata: {
    source?: string;
    page?: number;
    [key: string]: unknown;
  };
}

export interface AskRequest {
  question: string;
  top_k?: number;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
}

export interface StatsResponse {
  chunks: number;
  files: string[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: Date;
  isLoading?: boolean;
}

export interface UploadedDocument {
  name: string;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  progress?: number;
}
