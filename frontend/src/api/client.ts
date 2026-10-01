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

// --- S3 multipart path for big PDFs (>50MB). Direct-to-S3, 16MB parts. ---

export const MULTIPART_THRESHOLD_BYTES = 50 * 1024 * 1024;

export interface MultipartInit {
  upload_id: string;
  doc_id: string;
  part_size: number;
  total_parts: number;
  urls: { part_number: number; url: string }[];
}

export interface MultipartPart {
  part_number: number;
  etag: string;
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

/** Chunked SHA256 (no whole-file buffering → safe for multi-GB). */
export async function sha256File(file: File, onProgress?: (frac: number) => void): Promise<string> {
  // Incremental SHA-256 (compact, public-domain style) over 16MB slices.
  const K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
  let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
  const buf = new Uint8Array(64);
  let bufLen = 0;
  let total = 0;
  const w = new Int32Array(64);
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));
  function compress() {
    for (let i = 0; i < 16; i++) w[i] = (buf[i * 4] << 24) | (buf[i * 4 + 1] << 16) | (buf[i * 4 + 2] << 8) | buf[i * 4 + 3];
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, hh = h7;
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + K[i] + w[i]) | 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) | 0;
      hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + hh) | 0;
  }
  function feed(chunk: Uint8Array) {
    let off = 0;
    while (off < chunk.length) {
      const take = Math.min(64 - bufLen, chunk.length - off);
      buf.set(chunk.subarray(off, off + take), bufLen);
      bufLen += take; off += take;
      if (bufLen === 64) { compress(); bufLen = 0; }
    }
  }
  const SLICE = 16 * 1024 * 1024;
  for (let off = 0; off < file.size; off += SLICE) {
    const ab = await file.slice(off, off + SLICE).arrayBuffer();
    feed(new Uint8Array(ab));
    total += ab.byteLength;
    onProgress?.(total / file.size);
  }
  const bitLenHi = Math.floor((total * 8) / 0x100000000);
  const bitLenLo = (total * 8) >>> 0;
  feed(new Uint8Array([0x80]));
  while (bufLen !== 56) feed(new Uint8Array([0]));
  const lenBytes = new Uint8Array(8);
  new DataView(lenBytes.buffer).setUint32(0, bitLenHi);
  new DataView(lenBytes.buffer).setUint32(4, bitLenLo);
  feed(lenBytes);
  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map(x => (x >>> 0).toString(16).padStart(8, '0')).join('');
}

async function putPart(url: string, blob: Blob, tries = 4): Promise<string> {
  let lastErr: unknown;
  for (let a = 0; a < tries; a++) {
    try {
      const res = await fetch(url, { method: 'PUT', body: blob, headers: { 'Content-Type': 'application/pdf' } });
      if (!res.ok) throw new Error(`Part PUT failed: ${res.status}`);
      const etag = res.headers.get('ETag') ?? res.headers.get('etag') ?? '';
      if (!etag) throw new Error('S3 missing ETag (bucket CORS must expose ETag).');
      return etag;
    } catch (e) {
      lastErr = e;
      await sleep(500 * 2 ** a);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Part PUT failed');
}

export async function uploadPDFBig(
  file: File,
  onProgress?: (uploaded: number, total: number) => void,
): Promise<UploadResponse> {
  const sha256 = await sha256File(file);
  const initRes = await fetch(`${API_BASE}/uploads/init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: file.name, size_bytes: file.size, sha256 }),
  });
  if (!initRes.ok) {
    const e = await initRes.json().catch(() => ({ detail: 'Init failed' }));
    throw new Error(e.detail || `Init failed: ${initRes.status}`);
  }
  const init = (await initRes.json()) as MultipartInit;
  const etags = new Array<string>(init.total_parts);
  let done = 0;
  // Sequential PUTs: simplest + thrifty on flaky nets; parallelize later if needed.
  for (let i = 0; i < init.total_parts; i++) {
    const { url } = init.urls[i];
    const blob = file.slice(i * init.part_size, (i + 1) * init.part_size);
    etags[i] = await putPart(url, blob);
    done++;
    onProgress?.(done, init.total_parts);
  }
  const compRes = await fetch(`${API_BASE}/uploads/${init.upload_id}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parts: etags.map((etag, i) => ({ part_number: i + 1, etag })),
      sha256,
    }),
  });
  if (!compRes.ok) {
    const e = await compRes.json().catch(() => ({ detail: 'Complete failed' }));
    throw new Error(e.detail || `Complete failed: ${compRes.status}`);
  }
  return compRes.json();
}

export function shouldUseMultipart(file: File): boolean {
  return file.size >= MULTIPART_THRESHOLD_BYTES;
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
