import { useState, useRef, useCallback, useEffect } from 'react';
import type { UploadedDocument } from '../types';
import { uploadPDF, getJobStatus, getStats, deleteDocuments } from '../api/client';

export default function DocumentPanel() {
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [stats, setStats] = useState<{ chunks: number } | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadStats = useCallback(async () => {
    try {
      const s = await getStats();
      setStats({ chunks: s.chunks });
    } catch {
      // Backend not ready
    }
  }, []);

  useEffect(() => { void loadStats(); }, [loadStats]);

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Only PDF files are supported');
      return;
    }

    const doc: UploadedDocument = {
      name: file.name,
      status: 'processing',
    };

    setDocuments(prev => [doc, ...prev]);

    try {
      const result = await uploadPDF(file);
      
      const pollStatus = async () => {
        const job = await getJobStatus(result.job_id);
        
        if (job.status === 'completed') {
          setDocuments(prev =>
            prev.map(d => (d.name === file.name ? { ...d, status: 'completed' as const } : d))
          );
          loadStats();
        } else if (job.status === 'failed') {
          setDocuments(prev =>
            prev.map(d => (d.name === file.name ? { ...d, status: 'failed' as const } : d))
          );
        } else {
          setTimeout(pollStatus, 1000);
        }
      };

      pollStatus();
    } catch (error) {
      setDocuments(prev =>
        prev.map(d => (d.name === file.name ? { ...d, status: 'failed' as const } : d))
      );
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    files.forEach(handleFile);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    files.forEach(handleFile);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDeleteAll = async () => {
    if (!confirm('Are you sure you want to delete all documents? This will clear the entire knowledge base.')) {
      return;
    }

    setIsDeleting(true);
    try {
      await deleteDocuments();
      setDocuments([]);
      setStats(null);
    } catch (error) {
      alert('Failed to delete documents');
    } finally {
      setIsDeleting(false);
    }
  };

  const getStatusIcon = (status: UploadedDocument['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return (
          <svg className="w-4 h-4 animate-spin text-yellow-500" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
        );
      case 'completed':
        return (
          <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case 'failed':
        return (
          <svg className="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start justify-between border-b border-[#29303d] px-5 py-5">
        <div>
          <p className="mb-1 text-[10px] font-bold uppercase tracking-[0.15em] text-[#788399]">Retrieval context</p>
          <h2 className="text-base font-semibold text-[#eff2f8]">Document library</h2>
          <p className="mt-2 text-xs text-[#8d97aa]">{stats ? `${stats.chunks.toLocaleString()} chunks available` : 'Index status unavailable'}</p>
        </div>
        {documents.length > 0 && (
          <button
            onClick={handleDeleteAll}
            disabled={isDeleting}
            className="flex items-center gap-1 border border-[#59363d] px-2 py-1.5 text-[11px] font-medium text-red-300 transition-colors hover:border-[#95505b] hover:bg-[#281619] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isDeleting ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            )}
            Clear All
          </button>
        )}
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`m-5 flex min-h-[220px] flex-1 flex-col items-center justify-center border border-dashed p-6 transition-colors ${
          isDragging
            ? 'border-[#779eec] bg-[#12203c]'
            : 'border-[#354052] bg-[#10141d] hover:border-[#52627d]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          onChange={handleInputChange}
          className="hidden"
        />
        
        <div className="text-center">
          <svg
            className={`mx-auto mb-4 h-10 w-10 ${isDragging ? 'text-[#a9c4ff]' : 'text-[#64718a]'}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="mb-1 text-sm font-medium text-[#c6cedc]">Drop PDFs to enrich Atlas</p>
          <p className="mb-4 text-xs text-[#778399]">Your documents are indexed for grounded answers.</p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="border border-[#405f9b] bg-[#172647] px-3 py-2 text-xs font-semibold text-[#b6ccff] transition-colors hover:bg-[#203864]"
          >
            Select PDF files
          </button>
        </div>
      </div>

      {documents.length > 0 && (
        <div className="max-h-[220px] overflow-y-auto border-t border-[#29303d] px-5 py-4">
          <p className="mb-3 text-[10px] font-bold uppercase tracking-[0.15em] text-[#778399]">Upload queue</p>
          <ul className="space-y-1">
            {documents.map((doc, idx) => (
              <li key={idx} className="flex items-center gap-2 border border-[#29303d] bg-[#10141d] px-2.5 py-2">
                <svg className="h-4 w-4 shrink-0 text-[#70809b]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                <span className="flex-1 truncate text-xs text-[#c6cedc]">{doc.name}</span>
                {getStatusIcon(doc.status)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
