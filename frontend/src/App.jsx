import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const ROUTE_COLORS = {
  index: '#4ade80',
  web: '#60a5fa',
  general: '#f59e0b',
}

function QueryPanel() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const res = await fetch('/agents/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let answer = ''

      setResult({ route: null, plan: null, answer: '', context: [] })

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6)
          if (raw === '[DONE]') break

          let parsed
          try { parsed = JSON.parse(raw) } catch { continue }

          if (parsed.type === 'meta') {
            setResult(prev => ({ ...prev, route: parsed.route, plan: parsed.plan }))
          } else if (parsed.type === 'token') {
            answer += parsed.token
            setResult(prev => ({ ...prev, answer }))
          } else if (parsed.type === 'done') {
            if (!answer && parsed.answer) answer = parsed.answer
            setResult(prev => ({ ...prev, answer, context: parsed.context || [], _done: true }))
          } else if (parsed.type === 'error') {
            throw new Error(parsed.message)
          }
        }
      }
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <h2>Query</h2>
      <p className="subtitle">Ask anything about your ingested documents or get web-sourced answers.</p>
      <form onSubmit={handleSubmit} className="query-form">
        <textarea
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g. Why is the payment service failing in production?"
          rows={4}
          disabled={loading}
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit(e)
          }}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? 'Thinking...' : 'Ask →'}
        </button>
      </form>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-box">
          {(result.route || result.plan) && (
            <div className="result-meta">
              {result.route && (
                <span
                  className="route-badge"
                  style={{ background: ROUTE_COLORS[result.route] ?? '#6b7280' }}
                >
                  {result.route}
                </span>
              )}
              {result.plan && <span className="plan-text">Plan: {result.plan}</span>}
            </div>
          )}
          <div className={`answer${loading && !result._done ? ' streaming' : ''}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.answer}
            </ReactMarkdown>
          </div>
          {result.context?.length > 0 && (
            <details className="context-details">
              <summary>Sources ({result.context.length})</summary>
              {result.context.map((c, i) => (
                <p key={i} className="context-chunk">{c}</p>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  )
}

function IngestPanel() {
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [documents, setDocuments] = useState([])
  const [deletingSource, setDeletingSource] = useState(null)
  const inputRef = useRef(null)

  async function fetchDocuments() {
    try {
      const res = await fetch('/rag/documents')
      if (res.ok) {
        const data = await res.json()
        setDocuments(data.documents || [])
      }
    } catch {}
  }

  useEffect(() => { fetchDocuments() }, [])

  async function handleDelete(source) {
    setDeletingSource(source)
    try {
      const res = await fetch(`/rag/documents?source=${encodeURIComponent(source)}`, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Delete failed')
      }
      await fetchDocuments()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeletingSource(null)
    }
  }

  function handleFiles(incoming) {
    const valid = Array.from(incoming).filter(f =>
      ['.pdf', '.txt', '.md'].some(ext => f.name.endsWith(ext))
    )
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...valid.filter(f => !names.has(f.name))]
    })
  }

  function handleDrop(e) {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  function removeFile(name) {
    setFiles(prev => prev.filter(f => f.name !== name))
  }

  async function handleUpload() {
    if (!files.length) return
    setUploading(true)
    setStatus(null)
    setError(null)
    try {
      const form = new FormData()
      files.forEach(f => form.append('files', f))
      const res = await fetch('/rag/ingest', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      setStatus(data.message)
      setFiles([])
      await fetchDocuments()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="panel">
      <h2>Ingest</h2>
      <p className="subtitle">Upload PDF, .txt, or .md files to index them for querying.</p>

      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md"
          style={{ display: 'none' }}
          onChange={e => handleFiles(e.target.files)}
        />
        <span>Drop files here or click to browse</span>
        <small>.pdf · .txt · .md</small>
      </div>

      {files.length > 0 && (
        <ul className="file-list">
          {files.map(f => (
            <li key={f.name}>
              <span>{f.name}</span>
              <span className="file-size">{(f.size / 1024).toFixed(1)} KB</span>
              <button className="remove-btn" onClick={() => removeFile(f.name)}>×</button>
            </li>
          ))}
        </ul>
      )}

      <button
        className="upload-btn"
        onClick={handleUpload}
        disabled={!files.length || uploading}
      >
        {uploading ? 'Uploading...' : `Upload${files.length ? ` (${files.length})` : ''}`}
      </button>

      {status && <div className="success-box">{status}</div>}
      {error && <div className="error-box">{error}</div>}

      <div className="doc-index">
        <h3>Indexed documents {documents.length > 0 && <span className="doc-count">{documents.length}</span>}</h3>
        {documents.length === 0 ? (
          <p className="doc-empty">No documents indexed yet.</p>
        ) : (
          <ul className="doc-list">
            {documents.map(doc => (
              <li key={doc.source}>
                <span className="doc-name">{doc.source}</span>
                <span className="doc-chunks">{doc.chunk_count} chunks</span>
                <button
                  className="doc-delete"
                  onClick={() => handleDelete(doc.source)}
                  disabled={deletingSource === doc.source}
                  title="Remove from index"
                >
                  {deletingSource === doc.source ? '…' : '×'}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState('query')

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <span className="logo-mark">◈</span>
          <span>Synapse</span>
        </div>
        <nav>
          <button className={tab === 'query' ? 'active' : ''} onClick={() => setTab('query')}>
            Query
          </button>
          <button className={tab === 'ingest' ? 'active' : ''} onClick={() => setTab('ingest')}>
            Ingest
          </button>
        </nav>
      </aside>
      <main className="content">
        {tab === 'query' ? <QueryPanel /> : <IngestPanel />}
      </main>
    </div>
  )
}
