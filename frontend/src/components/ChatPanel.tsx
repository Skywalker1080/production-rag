import { useEffect, useRef, useState, type ComponentPropsWithoutRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { askQuestion } from '../api/client';
import type { Message, Source } from '../types';

const suggestedQuestions = ['What are the main findings?', 'Summarize the documents', 'Compare the key figures'];

function citationMarkdown(content: string, sources: Source[] = []) {
  return content.replace(/\[([^\]\n]+?),\s*p(?:age)?\.?\s*(\d+)\]/gi, (match, name: string, page: string) => {
    const normalizedName = name.trim().toLowerCase();
    const sourceIndex = sources.findIndex((source) => {
      const sourceName = String(source.metadata?.source ?? '').trim().toLowerCase();
      return Number(source.metadata?.page) === Number(page) && (sourceName === normalizedName || sourceName.includes(normalizedName) || normalizedName.includes(sourceName));
    });
    return sourceIndex === -1 ? match : `[${match}](#citation-${sourceIndex})`;
  });
}

function CitationPill({ href, children, ...props }: Readonly<ComponentPropsWithoutRef<'a'> & { source?: Source }>) {
  const [isOpen, setIsOpen] = useState(false);
  const source = props.source;
  if (!href?.startsWith('#citation-') || !source) return <a href={href} {...props}>{children}</a>;
  const citationText = String(children);
  const pageLabel = citationText.match(/p(?:age)?\.?\s*(\d+)/i)?.[0] ?? 'Source';

  return <span className="inline-citation" onMouseEnter={() => setIsOpen(true)} onMouseLeave={() => setIsOpen(false)}>
    <button type="button" className="citation-pill" aria-label={`View citation: ${citationText}`} aria-expanded={isOpen} onFocus={() => setIsOpen(true)} onBlur={() => setIsOpen(false)} onClick={() => setIsOpen((open) => !open)}>{pageLabel}</button>
    {isOpen && <span role="tooltip" className="citation-tooltip"><span className="citation-tooltip-label">{source.metadata?.source ?? 'Source'} · {pageLabel}</span>{source.content.slice(0, 350)}{source.content.length > 350 && '…'}</span>}
  </span>;
}

function Citation({ source, index }: Readonly<{ source: Source; index: number }>) {
  const [expanded, setExpanded] = useState(false);
  const sourceName = source.metadata?.source || 'Unknown source';
  const page = source.metadata?.page;

  return <div className="border-t border-[#2d3442] first:border-t-0">
    <button type="button" onClick={() => setExpanded((value) => !value)} className="flex w-full items-center gap-2 py-2 text-left text-xs text-slate-300 transition-colors hover:text-white">
      <span className="grid h-5 w-5 shrink-0 place-items-center border border-[#4163a6] bg-[#142242] text-[10px] font-bold text-[#adc7ff]">{String(index + 1).padStart(2, '0')}</span>
      <span className="min-w-0 flex-1 truncate">{sourceName}</span>
      {page !== undefined && <span className="text-slate-500">p. {page}</span>}
      <svg className={`h-3.5 w-3.5 text-slate-500 transition-transform ${expanded ? 'rotate-180' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m6 9 6 6 6-6" /></svg>
    </button>
    {expanded && <p className="mb-2 border-l-2 border-[#4163a6] bg-[#101722] px-3 py-2 text-xs leading-5 text-slate-400">{source.content.slice(0, 350)}{source.content.length > 350 && '…'}</p>}
  </div>;
}

function MessageItem({ message }: Readonly<{ message: Message }>) {
  const isUser = message.role === 'user';
  const markdown = citationMarkdown(message.content, message.sources);
  const citationSources = message.sources ?? [];
  return <article className={`message-row ${isUser ? 'message-user' : 'message-assistant'}`}>
    {!isUser && <div className="message-avatar">AI</div>}
    <div className={`message-content ${isUser ? 'user-content' : 'assistant-content'}`}>
      <p className="message-label">{isUser ? 'You' : 'Atlas'}</p>
      {message.isLoading ? <div className="loading-line"><span /><span /><span /></div> : isUser ? <p className="whitespace-pre-wrap text-[15px] leading-7 text-slate-200">{message.content}</p> : <div className="markdown-output"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ href, children, ...props }) => { const index = Number(href?.replace('#citation-', '')); return <CitationPill href={href} source={Number.isInteger(index) ? citationSources[index] : undefined} {...props}>{children}</CitationPill>; } }}>{markdown}</ReactMarkdown></div>}
      {!isUser && message.sources && message.sources.length > 0 && <section className="sources"><p className="sources-title">Grounding sources</p>{message.sources.map((source, index) => <Citation key={`${source.metadata?.source ?? 'source'}-${index}`} source={source} index={index} />)}</section>}
    </div>
  </article>;
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const submitQuestion = async () => {
    const question = input.trim();
    if (!question || isLoading) return;
    const timestamp = Date.now();
    const userMessage: Message = { id: String(timestamp), role: 'user', content: question, timestamp: new Date() };
    const loadingMessage: Message = { id: String(timestamp + 1), role: 'assistant', content: '', timestamp: new Date(), isLoading: true };
    setMessages((current) => [...current, userMessage, loadingMessage]);
    setInput('');
    setIsLoading(true);
    try {
      const response = await askQuestion({ question });
      setMessages((current) => current.map((message) => message.id === loadingMessage.id ? { ...loadingMessage, content: response.answer, sources: response.sources, isLoading: false } : message));
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to retrieve an answer.';
      setMessages((current) => current.map((message) => message.id === loadingMessage.id ? { ...loadingMessage, content: `Request failed: ${detail}`, isLoading: false } : message));
    } finally { setIsLoading(false); }
  };

  return <div className="flex h-full flex-col">
    <div className="chat-scroll flex-1 overflow-y-auto">
      {messages.length === 0 ? <div className="empty-state">
        <div className="empty-grid" aria-hidden="true"><span /><span /><span /><span /></div>
        <p className="eyebrow">Ready when you are</p>
        <h2>Ask your knowledge base.</h2>
        <p className="empty-copy">Atlas finds the most relevant passages across your library and returns answers with traceable sources.</p>
        <div className="suggestion-list">{suggestedQuestions.map((question) => <button key={question} type="button" onClick={() => { setInput(question); inputRef.current?.focus(); }}><span>{question}</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M5 12h13M13 6l6 6-6 6" /></svg></button>)}</div>
      </div> : <div className="messages">{messages.map((message) => <MessageItem key={message.id} message={message} />)}<div ref={messagesEndRef} /></div>}
    </div>
    <form onSubmit={(event) => { event.preventDefault(); void submitQuestion(); }} className="composer-shell">
      <div className="composer">
        <textarea ref={inputRef} value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submitQuestion(); } }} placeholder="Ask anything about your documents" disabled={isLoading} rows={1} />
        <button type="submit" disabled={!input.trim() || isLoading} aria-label="Send question">{isLoading ? <span className="send-spinner" /> : <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m5 12 14-7-5 14-2.5-5L5 12Z" /><path d="m11.5 14 3-3" /></svg>}</button>
      </div>
      <p>Enter to send <span>·</span> Shift + Enter for a new line</p>
    </form>
  </div>;
}
