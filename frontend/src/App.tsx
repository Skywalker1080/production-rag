import { useEffect, useState } from 'react';
import ChatPanel from './components/ChatPanel';
import DocumentPanel from './components/DocumentPanel';
import { getHealth } from './api/client';

type ConnectionStatus = 'checking' | 'connected' | 'disconnected';

function StatusIndicator({ status }: Readonly<{ status: ConnectionStatus }>) {
  const labels = { checking: 'Checking connection', connected: 'System online', disconnected: 'System offline' };

  return (
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-slate-400">
      <span className={`status-dot ${status}`} aria-hidden="true" />
      <span>{labels[status]}</span>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>('checking');
  const [isLibraryOpen, setIsLibraryOpen] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try { await getHealth(); setStatus('connected'); } catch { setStatus('disconnected'); }
    };
    void checkHealth();
    const interval = window.setInterval(() => void checkHealth(), 30_000);
    return () => window.clearInterval(interval);
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand"><div className="brand-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M5 5.75h14v12.5H9.2L5 21V5.75Z" /><path d="M8.5 10h7M8.5 13.5h4.5" /></svg></div><div><p className="eyebrow">Knowledge workspace</p><h1>Atlas RAG</h1></div></div>
        <div className="header-actions"><StatusIndicator status={status} /><button type="button" className="library-toggle" onClick={() => setIsLibraryOpen((open) => !open)} aria-expanded={isLibraryOpen} aria-controls="document-library"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M4 5h16M4 12h16M4 19h16" /></svg><span>Library</span></button></div>
      </header>
      <main className={`workspace ${isLibraryOpen ? '' : 'library-closed'}`}>
        <section className="chat-workspace" aria-label="RAG conversation"><ChatPanel /></section>
        <aside id="document-library" className="document-library" aria-label="Document library"><DocumentPanel /></aside>
      </main>
    </div>
  );
}
