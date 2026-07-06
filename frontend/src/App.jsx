import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Search, Settings, Send, Scale, ChevronRight, X, Loader2, Book, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './index.css';

const LinkifyCitations = ({ content, onResolve }) => {
  if (!content) return null;
  
  const citRegex = /\b(\d+)\s+((?:U\.S\.|F\.(?:2d|3d|4th)?|F\.\s*Supp\.(?:2d|3d)?|S\.\s*Ct\.|L\.\s*Ed\.(?:2d)?|A\.(?:2d|3d)?|P\.(?:2d|3d)?|N\.\s*E\.(?:2d)?|N\.\s*W\.(?:2d)?|S\.\s*E\.(?:2d)?|S\.\s*W\.(?:2d)?|So\.(?:2d|3d)?))\s+(\d+)\b/gi;
  
  // Split the string by existing markdown citation links (both legacy citation:// and new hash #citation- link formats)
  const parts = content.split(/(\[[^\]]+\]\((?:citation:\/\/|#citation-)[^\)]+\))/gi);
  const linkifiedParts = parts.map((part) => {
    if (part.startsWith('[') && (part.includes('](citation://') || part.includes('](#citation-'))) {
      // Normalize legacy custom protocols to safe hash links to bypass ReactMarkdown URL sanitization
      if (part.includes('](citation://')) {
        return part.replace('](citation://', '](#citation-');
      }
      return part;
    }
    // Search/replace plain text segments with safe hash links
    return part.replace(citRegex, (match) => {
      return `[${match}](#citation-${encodeURIComponent(match)})`;
    });
  });
  
  const linkifiedContent = linkifiedParts.join('');
  
  return (
    <ReactMarkdown 
      className="markdown-body"
      components={{
        a: ({ href, children }) => {
          const isCitation = href && (
            href.startsWith('#citation-') || 
            href.includes('#citation-') ||
            href.startsWith('citation://') || 
            href.includes('citation://')
          );
          
          if (isCitation) {
            let citationText = '';
            if (href.includes('citation://')) {
              const citationPart = href.substring(href.indexOf('citation://') + 11);
              citationText = decodeURIComponent(citationPart);
            } else {
              const citationPart = href.substring(href.indexOf('#citation-') + 10);
              citationText = decodeURIComponent(citationPart);
            }
            
            return (
              <span 
                className="citation-link" 
                style={{ cursor: 'pointer', textDecoration: 'underline', color: 'var(--accent-hover)', fontWeight: 500 }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onResolve(citationText);
                }}
              >
                {children}
              </span>
            );
          }
          return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
        }
      }}
    >
      {linkifiedContent}
    </ReactMarkdown>
  );
};

const getApiUrl = (path) => {
  if (window.location.port === '8000' || window.location.port === '' || window.location.port === '80' || window.location.port === '443') {
    return path;
  }
  return `http://${window.location.hostname}:8000${path}`;
};

const PIPELINE_STEPS = [
  "🔍 Checking query signature in search cache...",
  "🔄 Expanding query for semantic search terms...",
  "🧠 Generating query embeddings...",
  "📚 Executing PostgreSQL hybrid vector + FTS search...",
  "🎯 Reranking documents with Cohere...",
  "⚖️ Simulating Attorney Agent advocacy framing...",
  "👨‍⚖️ Simulating Judge Agent objective guidance..."
];

function App() {
  const [query, setQuery] = useState('');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome to Judge Read. How can I assist you with your legal research today?', sources: null }
  ]);
  const messagesEndRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  
  // Auth & Session History State
  const [username, setUsername] = useState(() => localStorage.getItem('username') || '');
  const [tempUsername, setTempUsername] = useState('');
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [userSessions, setUserSessions] = useState([]);
  const [isSessionsLoading, setIsSessionsLoading] = useState(false);
  
  // Full Case Modal State
  const [selectedCase, setSelectedCase] = useState(null);
  const [isCaseLoading, setIsCaseLoading] = useState(false);

  // Settings state
  const [embeddingModel, setEmbeddingModel] = useState('OpenAI:text-embedding-3-small');
  const [embeddingKey, setEmbeddingKey] = useState('');
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState([]);
  const [llmEngine, setLlmEngine] = useState('claude');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [anthropicApiKey, setAnthropicApiKey] = useState('');
  const [ollamaHost, setOllamaHost] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  
  // Metadata Filters
  const [filterYear, setFilterYear] = useState('');
  const [filterCourt, setFilterCourt] = useState('');
  const [filterSystem, setFilterSystem] = useState('');
  const [filterState, setFilterState] = useState('');
  const [filterStatus, setFilterStatus] = useState('good_law');
  const [filterJudge, setFilterJudge] = useState('');
  const [filterTopic, setFilterTopic] = useState('');
  
  // Tracing
  const [langsmithKey, setLangsmithKey] = useState('');
  const [cohereKey, setCohereKey] = useState('');

  // Postgres Settings
  const [pgHost, setPgHost] = useState('localhost');
  const [pgPort, setPgPort] = useState('5432');
  const [pgUser, setPgUser] = useState('user');
  const [pgPassword, setPgPassword] = useState('password');
  const [pgDb, setPgDb] = useState('judgeread');
  
  const [isConfigLoaded, setIsConfigLoaded] = useState(false);

  // Case Explorer State
  const [isExplorerOpen, setIsExplorerOpen] = useState(false);
  const [explorerSearch, setExplorerSearch] = useState('');
  const [explorerCases, setExplorerCases] = useState([]);
  const [isExplorerLoading, setIsExplorerLoading] = useState(false);

  // Advanced Suite State
  const [activeTab, setActiveTab] = useState('chat'); // chat, explorer, dashboard, benchmark
  const [isSplitScreen, setIsSplitScreen] = useState(false);
  const [expandedStepsIdx, setExpandedStepsIdx] = useState(null);
  const [isBriefUploading, setIsBriefUploading] = useState(false);
  const [uploadedBriefName, setUploadedBriefName] = useState('');
  const [expandQuery, setExpandQuery] = useState(false);
  const [annotations, setAnnotations] = useState([]);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isAnalyticsLoading, setIsAnalyticsLoading] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [highlightPopover, setHighlightPopover] = useState(null); // { x, y, text }
  const [annotationText, setAnnotationText] = useState('');

  const [loadingStep, setLoadingStep] = useState(0);
  const [showLiveProgress, setShowLiveProgress] = useState(false);

  useEffect(() => {
    if (!isLoading) {
      setLoadingStep(0);
    }
  }, [isLoading]);

  useEffect(() => {
    // Load config on mount
    axios.get(getApiUrl('/api/config')).then((response) => {
      const data = response.data;
      if (data.embeddingModel) setEmbeddingModel(data.embeddingModel);
      if (data.embeddingKey) setEmbeddingKey(data.embeddingKey);
      if (data.llmEngine) setLlmEngine(data.llmEngine);
      if (data.openaiApiKey) setOpenaiApiKey(data.openaiApiKey);
      if (data.anthropicApiKey) setAnthropicApiKey(data.anthropicApiKey);
      if (data.ollamaHost) setOllamaHost(data.ollamaHost);
      if (data.langsmithKey) setLangsmithKey(data.langsmithKey);
      if (data.cohereKey) setCohereKey(data.cohereKey);
      if (data.pgHost) setPgHost(data.pgHost);
      if (data.pgPort) setPgPort(data.pgPort);
      if (data.pgUser) setPgUser(data.pgUser);
      if (data.pgPassword) setPgPassword(data.pgPassword);
      if (data.pgDb) setPgDb(data.pgDb);
      if (data.availableModels) setAvailableModels(data.availableModels);
      if (data.availableEmbeddingModels) setAvailableEmbeddingModels(data.availableEmbeddingModels);
      setIsConfigLoaded(true);
    }).catch((err) => {
      console.error("Could not load config", err);
      setIsConfigLoaded(true);
    });
  }, []);

  // Save config whenever it changes
  useEffect(() => {
    if (isConfigLoaded) {
      axios.post(getApiUrl('/api/config'), {
        embeddingModel,
        embeddingKey,
        llmEngine,
        openaiApiKey,
        anthropicApiKey,
        ollamaHost,
        langsmithKey,
        cohereKey,
        pgHost,
        pgPort,
        pgUser,
        pgPassword,
        pgDb,
        availableModels,
        availableEmbeddingModels
      }).catch(err => console.error("Failed to save config", err));
    }
  }, [embeddingModel, embeddingKey, llmEngine, openaiApiKey, anthropicApiKey, ollamaHost, langsmithKey, cohereKey, pgHost, pgPort, pgUser, pgPassword, pgDb, availableModels, availableEmbeddingModels, isConfigLoaded]);

  const fetchCases = async () => {
    setIsExplorerLoading(true);
    try {
      const params = new URLSearchParams();
      if (explorerSearch) params.append('search', explorerSearch);
      if (filterYear) params.append('year', filterYear);
      if (filterCourt) params.append('court', filterCourt);
      if (filterSystem === 'Federal') params.append('system', 'Federal');
      if (filterSystem === 'State') params.append('system', filterState || 'State');
      if (filterStatus) params.append('status', filterStatus);

      const response = await axios.get(getApiUrl(`/api/cases?${params.toString()}`));
      setExplorerCases(response.data.cases || []);
    } catch (err) {
      console.error("Failed to fetch cases:", err);
    } finally {
      setIsExplorerLoading(false);
    }
  };

  useEffect(() => {
    if (isExplorerOpen) {
      fetchCases();
    }
  }, [isExplorerOpen, explorerSearch, filterYear, filterCourt, filterSystem, filterState, filterStatus]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (username) {
      fetchUserSessions();
    }
  }, [username]);

  const fetchUserSessions = async () => {
    if (!username) return;
    setIsSessionsLoading(true);
    try {
      const response = await axios.get(getApiUrl(`/api/users/${username}/sessions`));
      setUserSessions(response.data.sessions || []);
    } catch (err) {
      console.error("Failed to fetch sessions:", err);
    } finally {
      setIsSessionsLoading(false);
    }
  };

  const loadChatHistory = async (id) => {
    try {
      const response = await axios.get(getApiUrl(`/api/sessions/${id}/history`));
      setMessages(response.data.messages || []);
      setSessionId(id);
    } catch (err) {
      console.error("Failed to load chat history:", err);
    }
  };

  const fetchFullCase = async (caseId) => {
    if (!caseId) return;
    setIsCaseLoading(true);
    setSelectedCase(null);
    try {
      const response = await axios.get(getApiUrl(`/api/cases/${caseId}`));
      setSelectedCase(response.data);
    } catch (error) {
      console.error("Failed to fetch full case", error);
      alert("Sorry, could not load the full text for this case.");
    } finally {
      setIsCaseLoading(false);
    }
  };

  const resolveAndOpenCitation = async (citationText) => {
    try {
      const res = await axios.post(getApiUrl('/api/citations/resolve'), {
        text: citationText
      });
      if (res.data.citations && res.data.citations.length > 0) {
        fetchFullCase(res.data.citations[0].case_id);
      } else {
        alert(`Citation "${citationText}" was recognized, but it was not found in the local database.`);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleBriefUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadedBriefName(file.name);
    setIsBriefUploading(true);
    setIsLoading(true);
    
    const userMessage = { role: 'user', content: `Uploaded Legal Brief: ${file.name}` };
    setMessages(prev => [...prev, userMessage]);
    
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) formData.append('session_id', sessionId);
    if (username) formData.append('username', username);
    formData.append('embedding_model', embeddingModel);
    formData.append('embedding_key', embeddingKey);
    formData.append('llm_engine', llmEngine);
    formData.append('openai_api_key', openaiApiKey);
    formData.append('anthropic_api_key', anthropicApiKey);
    formData.append('ollama_host', ollamaHost);
    if (cohereKey) formData.append('cohere_key', cohereKey);
    formData.append('expand_query', expandQuery);
    
    try {
      const response = await axios.post(getApiUrl('/api/upload_brief'), formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      if (!sessionId && response.data.session_id) {
        setSessionId(response.data.session_id);
      }
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources
      }]);
    } catch (err) {
      console.error("Brief upload failed", err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Failed to process uploaded brief: ${err.response?.data?.detail || err.message}`,
        error: true
      }]);
    } finally {
      setIsBriefUploading(false);
      setIsLoading(false);
      setUploadedBriefName('');
    }
  };

  const fetchAnnotations = async () => {
    if (!sessionId) return;
    try {
      const res = await axios.get(getApiUrl(`/api/sessions/${sessionId}/annotations`));
      setAnnotations(res.data.annotations || []);
    } catch (err) {
      console.error("Failed to fetch annotations", err);
    }
  };

  useEffect(() => {
    if (sessionId) {
      fetchAnnotations();
    }
  }, [sessionId, selectedCase]);

  const handleTextSelection = () => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text.length > 5) {
      try {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setHighlightPopover({
          x: rect.left + window.scrollX,
          y: rect.bottom + window.scrollY + 10,
          selectedText: text
        });
      } catch (err) {
        console.error(err);
      }
    } else {
      setHighlightPopover(null);
    }
  };

  const saveHighlight = async () => {
    if (!sessionId) {
      alert("Please start a search query or conversation first to create a session for annotations.");
      return;
    }
    if (!highlightPopover || !selectedCase) return;
    
    try {
      const res = await axios.post(getApiUrl(`/api/sessions/${sessionId}/annotations`), {
        case_id: selectedCase.case_id,
        highlighted_text: highlightPopover.selectedText,
        note: annotationText
      });
      setAnnotations(prev => [res.data, ...prev]);
      setAnnotationText('');
      setHighlightPopover(null);
      window.getSelection().removeAllRanges();
    } catch (err) {
      console.error("Failed to save annotation", err);
      alert("Error saving highlight annotation.");
    }
  };

  const deleteAnnotation = async (annoId) => {
    try {
      await axios.delete(getApiUrl(`/api/annotations/${annoId}`));
      setAnnotations(prev => prev.filter(a => a.id !== annoId));
    } catch (err) {
      console.error("Failed to delete annotation", err);
    }
  };

  const handleExportMemo = () => {
    if (!sessionId) {
      alert("No active session to export.");
      return;
    }
    window.open(getApiUrl(`/api/sessions/${sessionId}/export_memo`));
  };

  const fetchAnalytics = async () => {
    setIsAnalyticsLoading(true);
    try {
      const res = await axios.get(getApiUrl('/api/analytics/dashboard'));
      setAnalyticsData(res.data);
    } catch (err) {
      console.error("Failed to fetch analytics", err);
    } finally {
      setIsAnalyticsLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchAnalytics();
    }
  }, [activeTab]);

  const runBenchmarkSuite = async () => {
    setIsBenchmarking(true);
    try {
      const res = await axios.post(getApiUrl('/api/benchmark/run'), null, {
        params: {
          embedding_model: embeddingModel,
          embedding_key: embeddingKey,
          cohere_key: cohereKey
        }
      });
      setBenchmarkData(res.data);
    } catch (err) {
      console.error("Failed to run benchmarks", err);
      alert("Benchmark execution failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setIsBenchmarking(false);
    }
  };

  const renderCaseReader = (scase, loading, isSplit) => {
    if (loading) {
      return (
        <div style={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center', padding: '40px', height: '100%' }}>
          <Loader2 className="spinner" size={32} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
        </div>
      );
    }
    if (!scase) return null;

    let parsed = null;
    if (scase.full_text) {
      try {
        let rawText = scase.full_text;
        rawText = rawText.replace(/[\u0000-\u001F]+/g, (match) => {
          return match.split('').map(char => {
            if (char === '\n') return '\\n';
            if (char === '\r') return '\\r';
            if (char === '\t') return '\\t';
            if (char === '\f') return '\\f';
            return '';
          }).join('');
        });
        parsed = JSON.parse(rawText);
      } catch (e) {
        // not JSON
      }
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', flex: 1, overflow: 'hidden' }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(20,25,35,0.4)' }}>
          <div>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '350px' }}>
              {parsed && parsed.case_name_full ? parsed.case_name_full : scase.name}
            </h2>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              {scase.reporter} • {scase.court} • {scase.year}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              className="button-icon" 
              onClick={() => setIsSplitScreen(!isSplitScreen)} 
              title={isSplitScreen ? "Maximize View" : "Split View Mode"}
              style={{ color: isSplitScreen ? 'var(--accent-hover)' : 'inherit', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              {isSplitScreen ? "Full Screen" : "Split Screen"}
            </button>
            <button className="button-icon" onClick={() => { setSelectedCase(null); if (!isSplit) setIsCaseLoading(false); }}>
              <X size={20} />
            </button>
          </div>
        </div>

        <div 
          className="scroll-smooth" 
          onMouseUp={handleTextSelection}
          style={{ flex: 1, overflowY: 'auto', padding: '24px', fontSize: '0.95rem', lineHeight: '1.7', color: 'var(--text-main)', position: 'relative' }}
        >
          {highlightPopover && (
            <div 
              className="highlight-popover" 
              style={{ top: `${highlightPopover.y - 120}px`, left: `${Math.min(highlightPopover.x - 100, window.innerWidth - 300)}px` }}
              onMouseUp={(e) => e.stopPropagation()}
            >
              <div style={{ fontSize: '0.8rem', fontWeight: 'bold', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '4px', color: 'var(--text-main)' }}>
                Annotate Highlight
              </div>
              <textarea 
                placeholder="Add your attorney note here..."
                value={annotationText}
                onChange={(e) => setAnnotationText(e.target.value)}
                rows={3}
              />
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '6px' }}>
                <button 
                  className="button-icon" 
                  onClick={() => setHighlightPopover(null)} 
                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                >
                  Cancel
                </button>
                <button 
                  className="button-primary" 
                  onClick={saveHighlight}
                  style={{ padding: '4px 10px', fontSize: '0.75rem', borderRadius: '4px' }}
                >
                  Save
                </button>
              </div>
            </div>
          )}

          {parsed ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <h3 style={{ marginTop: 0, color: 'var(--accent)', fontSize: '1rem', marginBottom: '8px' }}>Metadata Details</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr', gap: '8px 12px', fontSize: '0.85rem' }}>
                  {parsed.case_name_full && <><span style={{ color: 'var(--text-muted)' }}>Name:</span><span>{parsed.case_name_full}</span></>}
                  {parsed.date_filed && <><span style={{ color: 'var(--text-muted)' }}>Filed:</span><span>{parsed.date_filed}</span></>}
                  {parsed.court_full_name && <><span style={{ color: 'var(--text-muted)' }}>Court:</span><span>{parsed.court_full_name}</span></>}
                  {parsed.judges && <><span style={{ color: 'var(--text-muted)' }}>Judges:</span><span>{parsed.judges}</span></>}
                  {parsed.attorneys && <><span style={{ color: 'var(--text-muted)' }}>Attorneys:</span><span>{parsed.attorneys}</span></>}
                  {parsed.citations && parsed.citations.length > 0 && <><span style={{ color: 'var(--text-muted)' }}>Citations:</span><span>{parsed.citations.join(', ')}</span></>}
                </div>
              </div>
              
              {parsed.summary && (
                <div>
                  <h3 style={{ color: 'var(--accent)', fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px' }}>Summary</h3>
                  <LinkifyCitations content={parsed.summary} onResolve={resolveAndOpenCitation} />
                </div>
              )}
              
              {parsed.syllabus && (
                <div>
                  <h3 style={{ color: 'var(--accent)', fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px' }}>Syllabus</h3>
                  <LinkifyCitations content={parsed.syllabus} onResolve={resolveAndOpenCitation} />
                </div>
              )}

              {parsed.headnotes && (
                <div>
                  <h3 style={{ color: 'var(--accent)', fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px' }}>Headnotes</h3>
                  <LinkifyCitations content={parsed.headnotes} onResolve={resolveAndOpenCitation} />
                </div>
              )}

              {parsed.opinions && parsed.opinions.length > 0 && (
                <div>
                  <h3 style={{ color: 'var(--accent)', fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px' }}>Opinions</h3>
                  {parsed.opinions.map((o, idx) => (
                    <div key={idx} style={{ marginTop: '12px' }}>
                      {o.author_str && <h4 style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Author: {o.author_str}</h4>}
                      <LinkifyCitations content={o.opinion_text} onResolve={resolveAndOpenCitation} />
                      {idx < parsed.opinions.length - 1 && <hr style={{ borderColor: 'rgba(255,255,255,0.05)', margin: '20px 0' }}/>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <LinkifyCitations content={scase.full_text} onResolve={resolveAndOpenCitation} />
          )}

          {annotations.length > 0 && (
            <div className="annotations-sidebar-section">
              <h3 style={{ color: 'var(--accent)', fontSize: '1.05rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', marginTop: '24px' }}>
                Attorney Highlights & Notes ({annotations.filter(a => a.case_id === scase.case_id).length})
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px' }}>
                {annotations.filter(a => a.case_id === scase.case_id).map(anno => (
                  <div key={anno.id} className="annotation-item-card animate-fade-in">
                    <div className="annotation-item-text">
                      "{anno.highlighted_text}"
                    </div>
                    {anno.note && (
                      <div style={{ color: 'var(--text-main)', fontWeight: 'normal', marginTop: '4px' }}>
                        Note: {anno.note}
                      </div>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {new Date(anno.created_at).toLocaleDateString()}
                      </span>
                      <button 
                        onClick={() => deleteAnnotation(anno.id)} 
                        style={{ color: '#ef4444', background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '0.75rem' }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderHeader = () => (
    <header style={{ 
      padding: '20px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      borderBottom: '1px solid var(--border-color)', background: 'rgba(11, 15, 25, 0.8)', backdropFilter: 'blur(10px)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ background: 'var(--accent)', padding: '8px', borderRadius: '12px', boxShadow: '0 0 15px var(--accent-glow)' }}>
          <Scale size={24} color="white" />
        </div>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 600, letterSpacing: '-0.5px' }}>Judge Read</h1>
        <div className="tabs-navigation">
          <button className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
            Research Chat
          </button>
          <button className={`tab-button ${activeTab === 'explorer' ? 'active' : ''}`} onClick={() => setActiveTab('explorer')}>
            Case Explorer
          </button>
          <button className={`tab-button ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            DB Analytics
          </button>
          <button className={`tab-button ${activeTab === 'benchmark' ? 'active' : ''}`} onClick={() => setActiveTab('benchmark')}>
            Performance
          </button>
        </div>
      </div>
      <div style={{ display: 'flex', gap: '12px' }}>
        <button className="button-icon" onClick={() => setIsLoginModalOpen(true)} title={username ? "Profile" : "Login"}>
          <User size={22} />
        </button>
        <button className="button-icon" onClick={() => setActiveTab('explorer')} title="Case Explorer">
          <Book size={22} />
        </button>
        <button className="button-icon" onClick={() => setIsSettingsOpen(true)}>
          <Settings size={22} />
        </button>
      </div>
    </header>
  );

  const renderRightSidebar = () => (
    <div style={{ width: '320px', borderLeft: '1px solid var(--border-color)', background: 'rgba(255,255,255,0.01)', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '20px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Chat History</h3>
        <button 
          className="button-icon" 
          style={{ padding: '4px 8px', fontSize: '0.8rem', color: '#ef4444' }}
          onClick={() => {
            setUsername('');
            localStorage.removeItem('username');
            setUserSessions([]);
          }}
        >
          Sign Out
        </button>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {isSessionsLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}><Loader2 size={24} className="animate-spin text-muted" /></div>
        ) : userSessions.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem', textAlign: 'center', marginTop: '20px' }}>No saved sessions.</p>
        ) : (
          userSessions.map(session => (
            <div 
              key={session.id}
              style={{
                padding: '12px', background: session.id === sessionId ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.03)', 
                borderRadius: '8px', cursor: 'pointer',
                border: session.id === sessionId ? '1px solid var(--accent)' : '1px solid var(--border-color)', 
                transition: 'all 0.2s'
              }}
              onClick={() => loadChatHistory(session.id)}
              onMouseEnter={(e) => e.currentTarget.style.background = session.id === sessionId ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.08)'}
              onMouseLeave={(e) => e.currentTarget.style.background = session.id === sessionId ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.03)'}
            >
              <div style={{ fontSize: '0.75rem', color: 'var(--accent)', marginBottom: '4px' }}>
                {new Date(session.created_at).toLocaleString()}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-main)', lineHeight: '1.4' }}>
                {session.preview || "Empty session"}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderTabContent = () => {
    if (activeTab === 'chat') {
      return (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div className="scroll-smooth chat-container" style={{ flex: 1, overflowY: 'auto', padding: '32px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`animate-fade-in chat-bubble ${msg.role === 'user' ? 'user-msg' : 'assistant-msg'}`} style={{
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: '8px'
              }}>
                {msg.role === 'assistant' && msg.cached && (
                  <div style={{ 
                    display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', 
                    color: 'var(--accent-hover)', fontWeight: 600, alignSelf: 'flex-start', 
                    marginLeft: '8px', marginBottom: '-4px' 
                  }}>
                    <span>⚡ Cached</span>
                  </div>
                )}
                <div style={{ 
                  padding: '16px 20px', 
                  borderRadius: '16px',
                  background: msg.role === 'user' ? 'var(--accent)' : 'var(--panel-bg)',
                  border: msg.role === 'user' ? 'none' : '1px solid var(--border-color)',
                  color: msg.role === 'user' ? 'white' : 'var(--text-main)',
                  boxShadow: msg.role === 'user' ? '0 4px 15px var(--accent-glow)' : '0 4px 15px rgba(0,0,0,0.2)',
                  borderBottomRightRadius: msg.role === 'user' ? '4px' : '16px',
                  borderTopLeftRadius: msg.role === 'assistant' ? '4px' : '16px',
                  lineHeight: '1.6'
                }}>
                  <LinkifyCitations content={msg.content} onResolve={resolveAndOpenCitation} />
                  
                  {msg.role === 'assistant' && msg.steps && msg.steps.length > 0 && (
                    <div style={{ marginTop: '12px', borderTop: '1px dashed var(--border-color)', paddingTop: '12px' }}>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setExpandedStepsIdx(expandedStepsIdx === idx ? null : idx);
                        }}
                        style={{
                          background: 'none', border: 'none', color: 'var(--accent-hover)',
                          fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center',
                          gap: '6px', padding: 0, fontWeight: 500, transition: 'color 0.2s'
                        }}
                        className="hover-underline"
                      >
                        <span>{expandedStepsIdx === idx ? '▼ Hide Backend Pipeline Trace' : '▶ Show Backend Pipeline Trace'}</span>
                      </button>
                      
                      {expandedStepsIdx === idx && (
                        <div className="animate-fade-in" style={{
                          marginTop: '10px', padding: '12px', background: 'rgba(0,0,0,0.2)',
                          borderRadius: '8px', border: '1px solid var(--border-color)',
                          fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)',
                          display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left'
                        }}>
                          {msg.steps.map((step, sIdx) => (
                            <div key={sIdx} style={{ display: 'flex', gap: '8px' }}>
                              <span style={{ color: 'var(--accent)' }}>•</span>
                              <span>{step}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {msg.sources && msg.sources.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '4px' }}>
                    {msg.sources.map((src, i) => (
                      <div 
                        key={i} 
                        onClick={() => fetchFullCase(src.case_id)}
                        style={{ 
                          fontSize: '0.8rem', padding: '6px 12px', background: 'rgba(255,255,255,0.05)', 
                          borderRadius: '20px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '4px',
                          color: 'var(--text-muted)', cursor: src.case_id ? 'pointer' : 'default',
                          transition: 'all 0.2s ease'
                        }}
                        className={src.case_id ? 'hover-pill' : ''}
                      >
                        <ChevronRight size={14} /> {src.name} ({src.reporter})
                        {src.overruled && (
                          <span style={{ 
                            background: 'rgba(255, 50, 50, 0.2)', color: '#ff6b6b', 
                            padding: '2px 6px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 'bold', marginLeft: '4px' 
                          }}>
                            OVERRULED
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              showLiveProgress ? (
                <div className="animate-fade-in animate-pulse-subtle" style={{ 
                  alignSelf: 'flex-start', padding: '20px', borderRadius: '16px', 
                  background: 'var(--panel-bg)', border: '1px solid var(--border-color)',
                  width: '320px', display: 'flex', flexDirection: 'column', gap: '12px',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Loader2 className="spinner" size={16} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', letterSpacing: '0.5px' }}>Analyzing Case Precedents...</span>
                  </div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', borderTop: '1px solid var(--border-color)', paddingTop: '10px' }}>
                    {PIPELINE_STEPS.map((step, idx) => {
                      let statusColor = 'var(--text-muted)';
                      let icon = '○';
                      let opacity = 0.4;
                      if (idx < loadingStep) {
                        statusColor = '#10B981'; // Completed (Green)
                        icon = '✓';
                        opacity = 1;
                      } else if (idx === loadingStep) {
                        statusColor = 'var(--accent-hover)'; // Active
                        icon = '●';
                        opacity = 1;
                      }
                      return (
                        <div key={idx} style={{ 
                          fontSize: '0.75rem', color: statusColor, display: 'flex', 
                          alignItems: 'center', gap: '8px', fontWeight: idx === loadingStep ? '600' : 'normal',
                          opacity: opacity, transition: 'all 0.3s'
                        }}>
                          <span style={{ fontSize: '0.8rem', width: '12px' }}>{icon}</span>
                          <span>{step}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="animate-fade-in" style={{ alignSelf: 'flex-start', padding: '16px 20px', borderRadius: '16px', background: 'var(--panel-bg)', border: '1px solid var(--border-color)' }}>
                  <Loader2 className="spinner" size={20} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                </div>
              )
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container" style={{ padding: '24px 32px', background: 'linear-gradient(to top, rgba(11,15,25,1) 50%, rgba(11,15,25,0))' }}>
            <div style={{
              display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '20px',
              background: 'rgba(255, 255, 255, 0.02)', padding: '16px', borderRadius: '16px',
              border: '1px solid rgba(255, 255, 255, 0.08)', width: '100%'
            }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>
                Search Filters
              </div>
              <FilterControls />
            </div>

            <form onSubmit={handleSearch} className="glass-panel" style={{ 
              display: 'flex', alignItems: 'center', padding: '8px 16px', borderRadius: '24px',
              border: '1px solid rgba(255,255,255,0.15)'
            }}>
              <Search size={20} color="var(--text-muted)" style={{ margin: '0 8px' }} />
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search case law (e.g., 'Texas cases on implied warranty...')" 
                style={{
                  flex: 1, background: 'transparent', border: 'none', color: 'var(--text-main)',
                  padding: '12px 8px', fontSize: '1rem', outline: 'none', fontFamily: 'Outfit'
                }}
                disabled={isLoading}
              />
              <button type="submit" className="button-primary" style={{ padding: '10px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '40px', height: '40px' }} disabled={isLoading || !query.trim()}>
                <Send size={18} style={{ marginLeft: '2px' }} />
              </button>
            </form>
            
            {isBriefUploading && (
              <div className="file-chip animate-fade-in" style={{ marginTop: '8px' }}>
                <Loader2 className="spinner" size={14} style={{ animation: 'spin 1s linear infinite' }} /> Processing Brief...
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: 'var(--text-muted)' }}>
                  <input 
                    type="file" 
                    accept=".txt,.pdf" 
                    style={{ display: 'none' }} 
                    onChange={handleBriefUpload}
                    disabled={isBriefUploading || isLoading}
                  />
                  <Book size={16} /> {isBriefUploading ? "Parsing Brief..." : "Analyze Legal Brief (PDF/TXT)"}
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={expandQuery} 
                    onChange={(e) => setExpandQuery(e.target.checked)} 
                    style={{ cursor: 'pointer' }}
                  />
                  LLM Query Expansion
                </label>
              </div>
              {sessionId && (
                <button 
                  className="button-icon" 
                  onClick={handleExportMemo} 
                  style={{ fontSize: '0.8rem', padding: '4px 10px', display: 'flex', alignItems: 'center', gap: '6px', border: '1px solid var(--border-color)', borderRadius: '6px' }}
                >
                  Export Memo
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }
    
    if (activeTab === 'explorer') {
      return (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '24px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', background: 'var(--panel-bg)', borderRadius: '16px', border: '1px solid var(--border-color)', padding: '24px', height: '100%' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px' }}>
              <Book size={24} color="var(--accent)" />
              <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Case Precedent Explorer</h2>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
                <Search size={20} color="var(--text-muted)" style={{ position: 'absolute', marginLeft: '16px' }} />
                <input 
                  type="text" 
                  value={explorerSearch}
                  onChange={(e) => setExplorerSearch(e.target.value)}
                  placeholder="Filter case names..."
                  className="input-glass"
                  style={{ width: '100%', padding: '10px 16px 10px 44px', borderRadius: '8px' }}
                />
              </div>
              <FilterControls />
            </div>

            <div style={{ flex: 1, overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
              {isExplorerLoading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}>
                  <Loader2 className="spinner" size={32} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                </div>
              ) : explorerCases.length === 0 ? (
                <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No cases found. Try adjusting filters.
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-color)', zIndex: 1 }}>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Case Name</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Year</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Court</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explorerCases.map(c => (
                      <tr 
                        key={c.case_id} 
                        onClick={() => fetchFullCase(c.case_id)}
                        style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', cursor: 'pointer', transition: 'background 0.2s' }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                      >
                        <td style={{ padding: '12px 16px', color: 'var(--accent-hover)', fontWeight: 500 }}>{c.name}</td>
                        <td style={{ padding: '12px 16px' }}>{c.year}</td>
                        <td style={{ padding: '12px 16px' }}>{c.court} ({c.jurisdiction})</td>
                        <td style={{ padding: '12px 16px' }}>
                          <span style={{ 
                            color: c.status === 'good_law' ? '#51cf66' : (c.status === 'overruled' ? '#ff6b6b' : '#fcc419'),
                            background: c.status === 'good_law' ? 'rgba(81,207,102,0.1)' : (c.status === 'overruled' ? 'rgba(255,107,107,0.1)' : 'rgba(252,196,25,0.1)'),
                            padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600
                          }}>
                            {c.status === 'good_law' ? 'Good Law' : (c.status === 'overruled' ? 'Overruled' : 'Caution')}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      );
    }
    
    if (activeTab === 'dashboard') {
      return (
        <div className="dashboard-grid scroll-smooth">
          {isAnalyticsLoading ? (
            <div style={{ gridColumn: '1/-1', display: 'flex', justifyContent: 'center', padding: '100px' }}>
              <Loader2 className="spinner" size={32} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
            </div>
          ) : !analyticsData ? (
            <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '100px', color: 'var(--text-muted)' }}>
              No database statistics available. Check connection.
            </div>
          ) : (
            <>
              <div className="glass-panel dashboard-card">
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Total Cases Ingested</div>
                <div className="dashboard-stat-num">{analyticsData.total_cases.toLocaleString()}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>US State & Federal opinions index</div>
              </div>

              <div className="glass-panel dashboard-card">
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Queries Processed</div>
                <div className="dashboard-stat-num">{analyticsData.total_queries.toLocaleString()}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Total LLM & Hybrid searches run</div>
              </div>

              <div className="glass-panel dashboard-card" style={{ gridColumn: 'span 2' }}>
                <h3 style={{ fontSize: '1rem', color: 'var(--accent)', margin: 0 }}>Precedent Citator Statuses</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '12px' }}>
                  {analyticsData.status_distribution.map(status => {
                    const pct = ((status.count / analyticsData.total_cases) * 100).toFixed(1);
                    return (
                      <div key={status.status} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                          <span style={{ textTransform: 'uppercase', fontWeight: 500 }}>
                            {status.status === 'good_law' ? '✅ Good Law' : (status.status === 'overruled' ? '❌ Overruled' : '⚠️ Caution')}
                          </span>
                          <span>{status.count.toLocaleString()} cases ({pct}%)</span>
                        </div>
                        <div className="progress-bar-container">
                          <div className="progress-bar-fill" style={{ 
                            width: `${pct}%`,
                            background: status.status === 'good_law' ? '#51cf66' : (status.status === 'overruled' ? '#ff6b6b' : '#fcc419')
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="glass-panel dashboard-card">
                <h3 style={{ fontSize: '1rem', color: 'var(--accent)', margin: 0 }}>Top Courts Represented</h3>
                <ul className="dashboard-list" style={{ marginTop: '12px' }}>
                  {analyticsData.court_distribution.map(court => (
                    <li key={court.court} className="dashboard-list-item" style={{ fontSize: '0.85rem' }}>
                      <span style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={court.court}>{court.court}</span>
                      <span style={{ color: 'var(--text-muted)' }}>{court.count} cases</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="glass-panel dashboard-card">
                <h3 style={{ fontSize: '1rem', color: 'var(--accent)', margin: 0 }}>Top Legal Categories</h3>
                <ul className="dashboard-list" style={{ marginTop: '12px' }}>
                  {analyticsData.topic_distribution.map(topic => (
                    <li key={topic.topic} className="dashboard-list-item" style={{ fontSize: '0.85rem' }}>
                      <span>{topic.topic}</span>
                      <span style={{ color: 'var(--text-muted)' }}>{topic.count} tags</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="glass-panel dashboard-card">
                <h3 style={{ fontSize: '1rem', color: 'var(--accent)', margin: 0 }}>Top Search Queries</h3>
                <ul className="dashboard-list" style={{ marginTop: '12px' }}>
                  {analyticsData.top_queries.map(q => (
                    <li key={q.query} className="dashboard-list-item" style={{ fontSize: '0.85rem' }}>
                      <span style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={q.query}>"{q.query}"</span>
                      <span style={{ color: 'var(--text-muted)' }}>{q.count} times</span>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      );
    }
    
    if (activeTab === 'benchmark') {
      return (
        <div className="benchmark-container scroll-smooth">
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px' }}>
              <div>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-main)' }}>Search Latency Benchmark Suite</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Runs controlled searches using standard legal questions to measure step-by-step performance.
                </p>
              </div>
              <button className="button-primary" onClick={runBenchmarkSuite} disabled={isBenchmarking}>
                {isBenchmarking ? "Running Benchmarks..." : "Run Benchmark Test"}
              </button>
            </div>

            {isBenchmarking ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', padding: '40px' }}>
                <Loader2 className="spinner" size={40} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Executing 5 hybrid queries & reranking runs...</p>
              </div>
            ) : benchmarkData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <div>
                  <h3 style={{ fontSize: '1rem', color: 'var(--accent)', marginBottom: '16px' }}>Average Execution Times (ms)</h3>
                  <div className="benchmark-chart">
                    <div className="chart-bar-row">
                      <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>1. Embedding Generation</span>
                      <div className="chart-bar-container">
                        <div 
                          className="chart-bar-fill chart-bar-embedding" 
                          style={{ width: `${Math.max(5, Math.min(100, (benchmarkData.averages.embedding_ms / benchmarkData.averages.total_ms) * 100))}%` }}
                        >
                          {benchmarkData.averages.embedding_ms} ms
                        </div>
                      </div>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                        {((benchmarkData.averages.embedding_ms / benchmarkData.averages.total_ms) * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="chart-bar-row">
                      <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>2. pgvector + FTS Search</span>
                      <div className="chart-bar-container">
                        <div 
                          className="chart-bar-fill chart-bar-database" 
                          style={{ width: `${Math.max(5, Math.min(100, (benchmarkData.averages.database_ms / benchmarkData.averages.total_ms) * 100))}%` }}
                        >
                          {benchmarkData.averages.database_ms} ms
                        </div>
                      </div>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                        {((benchmarkData.averages.database_ms / benchmarkData.averages.total_ms) * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="chart-bar-row">
                      <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>3. Cohere Reranking</span>
                      <div className="chart-bar-container">
                        <div 
                          className="chart-bar-fill chart-bar-rerank" 
                          style={{ width: `${Math.max(5, Math.min(100, (benchmarkData.averages.rerank_ms / benchmarkData.averages.total_ms) * 100))}%` }}
                        >
                          {benchmarkData.averages.rerank_ms} ms
                        </div>
                      </div>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                        {((benchmarkData.averages.rerank_ms / benchmarkData.averages.total_ms) * 100).toFixed(0)}%
                      </span>
                    </div>

                    <hr style={{ borderColor: 'rgba(255,255,255,0.05)', margin: '8px 0' }} />

                    <div className="chart-bar-row" style={{ fontWeight: 'bold' }}>
                      <span>Average Query Latency</span>
                      <span></span>
                      <span style={{ color: 'var(--accent-hover)', textAlign: 'right' }}>{benchmarkData.averages.total_ms} ms</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 style={{ fontSize: '1rem', color: 'var(--accent)', marginBottom: '12px' }}>Query Latency Breakdowns</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {benchmarkData.queries.map((q, idx) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontSize: '0.9rem', fontWeight: 500 }}>"{q.query}"</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                            Embed: {q.latency.embedding_ms}ms | DB Search: {q.latency.database_ms}ms | Rerank: {q.latency.rerank_ms}ms
                          </div>
                        </div>
                        <div style={{ fontWeight: 'bold', color: 'var(--text-main)', fontSize: '0.9rem' }}>
                          {q.latency.total_ms} ms
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
                Benchmark has not been run in this workspace session yet.
              </div>
            )}
          </div>
        </div>
      );
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);
    setLoadingStep(0);

    try {
      const response = await fetch(getApiUrl('/api/search'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: userMessage.content,
          session_id: sessionId,
          username: username || null,
          embedding_model: embeddingModel,
          embedding_key: embeddingKey,
          llm_engine: llmEngine,
          openai_api_key: openaiApiKey,
          anthropic_api_key: anthropicApiKey,
          ollama_host: ollamaHost,
          filter_year: filterYear ? parseInt(filterYear) : null,
          filter_court: filterCourt ? filterCourt : null,
          filter_jurisdiction: filterSystem === 'Federal' ? 'Federal' : (filterSystem === 'State' ? (filterState || 'State') : null),
          filter_status: filterStatus ? filterStatus : null,
          filter_judge: filterJudge ? filterJudge : null,
          filter_topic: filterTopic ? filterTopic : null,
          langsmith_key: langsmithKey ? langsmithKey : null,
          cohere_key: cohereKey ? cohereKey : null,
          expand_query: expandQuery
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.type === 'step') {
                const stepText = data.step;
                const matchStr = stepText.substring(0, 8);
                const idx = PIPELINE_STEPS.findIndex(s => s.startsWith(matchStr));
                if (idx !== -1) {
                  setLoadingStep(idx);
                }
              } else if (data.type === 'result') {
                if (!sessionId && data.session_id) {
                  setSessionId(data.session_id);
                }
                setMessages(prev => [...prev, {
                  role: 'assistant',
                  content: data.answer,
                  sources: data.sources,
                  cached: data.cached,
                  steps: data.steps
                }]);
              }
            } catch (err) {
              console.error("Failed to parse stream chunk", err);
            }
          }
        }
      }
    } catch (error) {
      console.error("Search error:", error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'I encountered an error connecting to the retrieval system. Please check your backend.',
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const FilterControls = () => (
    <div className="filter-controls" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', width: '100%' }}>
      <select 
        className="input-glass" 
        value={filterSystem} 
        onChange={(e) => {
          setFilterSystem(e.target.value);
          if (e.target.value !== 'State') setFilterState('');
        }}
        style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', flex: 1, minWidth: '130px' }}
      >
        <option value="">Any System</option>
        <option value="Federal">Federal Only</option>
        <option value="State">State Only</option>
      </select>

      {filterSystem === 'State' && (
        <select 
          className="input-glass animate-fade-in" 
          value={filterState} 
          onChange={(e) => setFilterState(e.target.value)}
          style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', flex: 1, minWidth: '150px' }}
        >
          <option value="">All States</option>
          {["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"].map(state => (
            <option key={state} value={state}>{state}</option>
          ))}
        </select>
      )}

      <select 
        className="input-glass" 
        value={filterCourt} 
        onChange={(e) => setFilterCourt(e.target.value)}
        style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', flex: 1, minWidth: '140px' }}
      >
        <option value="">Any Court Level</option>
        {(filterSystem === '' || filterSystem === 'Federal') && (
          <optgroup label="Federal Courts">
            <option value="US Supreme Court">US Supreme Court</option>
            <option value="US Court of Appeals (1st Circuit)">US Court of Appeals (1st Circuit)</option>
            <option value="US Court of Appeals (2nd Circuit)">US Court of Appeals (2nd Circuit)</option>
            <option value="US Court of Appeals (3rd Circuit)">US Court of Appeals (3rd Circuit)</option>
            <option value="US Court of Appeals (4th Circuit)">US Court of Appeals (4th Circuit)</option>
            <option value="US Court of Appeals (5th Circuit)">US Court of Appeals (5th Circuit)</option>
            <option value="US Court of Appeals (6th Circuit)">US Court of Appeals (6th Circuit)</option>
            <option value="US Court of Appeals (7th Circuit)">US Court of Appeals (7th Circuit)</option>
            <option value="US Court of Appeals (8th Circuit)">US Court of Appeals (8th Circuit)</option>
            <option value="US Court of Appeals (9th Circuit)">US Court of Appeals (9th Circuit)</option>
            <option value="US Court of Appeals (10th Circuit)">US Court of Appeals (10th Circuit)</option>
            <option value="US Court of Appeals (11th Circuit)">US Court of Appeals (11th Circuit)</option>
            <option value="US Court of Appeals (DC Circuit)">US Court of Appeals (DC Circuit)</option>
            <option value="US Court of Appeals (Federal Circuit)">US Court of Appeals (Federal Circuit)</option>
            <option value="US District Court">US District Court</option>
            <option value="US Bankruptcy Court">US Bankruptcy Court</option>
            <option value="US Tax Court">US Tax Court</option>
            <option value="US Court of Federal Claims">US Court of Federal Claims</option>
            <option value="US Court of International Trade">US Court of International Trade</option>
            <option value="US Court of Appeals for Veterans Claims">US Court of Appeals for Veterans Claims</option>
            <option value="US Court of Appeals for the Armed Forces">US Court of Appeals for the Armed Forces</option>
          </optgroup>
        )}
        {(filterSystem === '' || filterSystem === 'State') && (
          <optgroup label="State Courts">
            <option value="State Supreme Court">State Supreme Court</option>
            <option value="State Court of Appeals">State Court of Appeals</option>
            <option value="Superior Court">Superior Court</option>
            <option value="Circuit Court">Circuit Court</option>
            <option value="District Court">District Court</option>
            <option value="Municipal Court">Municipal Court</option>
            <option value="Justice Court">Justice Court</option>
            <option value="Magistrate Court">Magistrate Court</option>
            <option value="Family Court">Family Court</option>
            <option value="Probate Court">Probate Court</option>
            <option value="Juvenile Court">Juvenile Court</option>
            <option value="Small Claims Court">Small Claims Court</option>
            <option value="Traffic Court">Traffic Court</option>
            <option value="Workers' Compensation Court">Workers' Compensation Court</option>
          </optgroup>
        )}
      </select>

      <select 
        className="input-glass" 
        value={filterTopic} 
        onChange={(e) => setFilterTopic(e.target.value)}
        style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', flex: 1, minWidth: '140px' }}
      >
        <option value="">Any Topic</option>
        <option value="Criminal">Criminal</option>
        <option value="Civil">Civil</option>
        <option value="Tax">Tax</option>
        <option value="Intellectual Property">Intellectual Property</option>
        <option value="Constitutional">Constitutional</option>
      </select>

      <input 
        type="text"
        className="input-glass" 
        placeholder="Judge (e.g. Sotomayor)"
        value={filterJudge} 
        onChange={(e) => setFilterJudge(e.target.value)} 
        style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', flex: 1, minWidth: '150px' }}
      />

      <input 
        type="number"
        className="input-glass" 
        placeholder="Year (e.g. 2015)"
        value={filterYear} 
        onChange={(e) => setFilterYear(e.target.value)} 
        style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', flex: 1, minWidth: '120px' }}
      />

      <label className="input-glass" style={{ 
        display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', 
        fontSize: '0.85rem', color: 'var(--text-main)', userSelect: 'none',
        background: 'rgba(255,255,255,0.05)', padding: '8px 16px', borderRadius: '20px',
        flex: 1, minWidth: '150px'
      }}>
        <input 
          type="checkbox" 
          checked={filterStatus === 'good_law'} 
          onChange={(e) => setFilterStatus(e.target.checked ? 'good_law' : '')}
          style={{ cursor: 'pointer', accentColor: 'var(--accent)' }}
        />
        <span>Good Law Only</span>
      </label>

      <label className="input-glass" style={{ 
        display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', 
        fontSize: '0.85rem', color: 'var(--text-main)', userSelect: 'none',
        background: 'rgba(255,255,255,0.05)', padding: '8px 16px', borderRadius: '20px',
        flex: 1, minWidth: '150px'
      }}>
        <input 
          type="checkbox" 
          checked={showLiveProgress} 
          onChange={(e) => setShowLiveProgress(e.target.checked)}
          style={{ cursor: 'pointer', accentColor: 'var(--accent)' }}
        />
        <span>Show Live Trace</span>
      </label>
    </div>
  );

  return (
    <div className="app-container" style={{ display: 'flex', height: '100vh', position: 'relative' }}>
      
      {/* Sidebar / Settings */}
      <div className={`glass-panel settings-panel ${isSettingsOpen ? 'open' : ''}`} style={{
        position: 'fixed', left: 0, top: 0, bottom: 0, width: '300px', zIndex: 50,
        transform: isSettingsOpen ? 'translateX(0)' : 'translateX(-100%)',
        transition: 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
        borderLeft: 'none', borderTop: 'none', borderBottom: 'none', borderRadius: 0,
        padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={20} /> Configuration
          </h2>
          <button className="button-icon" onClick={() => setIsSettingsOpen(false)}>
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>Embedding Model</label>
            <select className="input-glass" value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)} style={{ marginBottom: '8px' }}>
              {availableEmbeddingModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
            <input 
              type={embeddingModel && embeddingModel.startsWith('Ollama:') ? 'text' : 'password'}
              className="input-glass" 
              placeholder={embeddingModel && embeddingModel.startsWith('Ollama:') ? 'Host (e.g. http://localhost:11434)' : 'API Key'}
              value={embeddingKey} 
              onChange={(e) => setEmbeddingKey(e.target.value)} 
              autoComplete="new-password"
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>LLM Engine</label>
            <select className="input-glass" value={llmEngine} onChange={(e) => setLlmEngine(e.target.value)}>
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>OpenAI API Key</label>
            <input 
              type="password"
              className="input-glass" 
              placeholder="sk-proj-..."
              value={openaiApiKey} 
              onChange={(e) => setOpenaiApiKey(e.target.value)} 
              autoComplete="new-password"
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>Anthropic API Key</label>
            <input 
              type="password"
              className="input-glass" 
              placeholder="sk-ant-..."
              value={anthropicApiKey} 
              onChange={(e) => setAnthropicApiKey(e.target.value)} 
              autoComplete="new-password"
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>Ollama Host URL</label>
            <input 
              type="text"
              className="input-glass" 
              placeholder="http://localhost:11434"
              value={ollamaHost} 
              onChange={(e) => setOllamaHost(e.target.value)} 
            />
          </div>

          <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 500, marginBottom: '12px' }}>PostgreSQL Configuration</h3>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>Host</label>
                <input type="text" className="input-glass" value={pgHost} onChange={(e) => setPgHost(e.target.value)} />
              </div>
              <div style={{ width: '80px' }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>Port</label>
                <input type="text" className="input-glass" value={pgPort} onChange={(e) => setPgPort(e.target.value)} />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>User</label>
                <input type="text" className="input-glass" value={pgUser} onChange={(e) => setPgUser(e.target.value)} />
              </div>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>Password</label>
                <input type="password" className="input-glass" value={pgPassword} onChange={(e) => setPgPassword(e.target.value)} autoComplete="new-password" />
              </div>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>Database Name</label>
              <input type="text" className="input-glass" value={pgDb} onChange={(e) => setPgDb(e.target.value)} />
            </div>
          </div>

          <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 500, marginBottom: '12px' }}>Reranking</h3>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>Cohere API Key</label>
            <input 
              type="password"
              className="input-glass" 
              placeholder="Cohere API Key..."
              value={cohereKey} 
              onChange={(e) => setCohereKey(e.target.value)} 
              style={{ marginBottom: '16px' }}
              autoComplete="new-password"
            />

            <h3 style={{ fontSize: '1rem', fontWeight: 500, marginBottom: '12px' }}>LangSmith Tracing</h3>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>LangSmith API Key</label>
            <input 
              type="password"
              className="input-glass" 
              placeholder="lsv2_pt_..."
              value={langsmithKey} 
              onChange={(e) => setLangsmithKey(e.target.value)} 
              autoComplete="new-password"
            />
          </div>
        </div>
      </div>

      {/* Dynamic View container supporting Split Screen and standard modes */}
      {isSplitScreen && (selectedCase || isCaseLoading) ? (
        <div className="split-workspace animate-fade-in" style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          {/* Left Column: Interactive Tab Workspace */}
          <div className="split-chat-pane">
            {renderHeader()}
            {renderTabContent()}
          </div>
          
          {/* Right Column: Case Precedent Reading Panel */}
          <div className="split-reading-pane" style={{ borderLeft: '1px solid var(--border-color)' }}>
            {renderCaseReader(selectedCase, isCaseLoading, true)}
          </div>
        </div>
      ) : (
        <div className="app-container-inner" style={{ display: 'flex', flex: 1, width: '100%', maxWidth: '1600px', margin: '0 auto' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
            {renderHeader()}
            {renderTabContent()}
          </div>
          {username && renderRightSidebar()}
        </div>
      )}

      {/* Full Case Modal Overlay (when not in Split Screen mode) */}
      {!isSplitScreen && (selectedCase || isCaseLoading) && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.7)', backdropFilter: 'blur(4px)',
          zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '40px'
        }}>
          <div className="glass-panel animate-fade-in" style={{
            width: '100%', maxWidth: '900px', maxHeight: '90vh', display: 'flex', flexDirection: 'column',
            background: 'var(--panel-bg)', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
          }}>
            {renderCaseReader(selectedCase, isCaseLoading, false)}
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .hover-pill:hover { background: rgba(255,255,255,0.1) !important; color: white !important; transform: translateY(-1px); }
      `}</style>

      {/* Profile & History Modal */}
      {isLoginModalOpen && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)', backdropFilter: 'blur(8px)',
          zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '20px'
        }}>
          <div className="glass-panel animate-fade-in" style={{
            width: '100%', maxWidth: '500px', display: 'flex', flexDirection: 'column',
            background: 'var(--panel-bg)', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
            maxHeight: '80vh'
          }}>
            <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>{username ? `Welcome, ${username}` : 'Login'}</h2>
              <button className="button-icon" onClick={() => setIsLoginModalOpen(false)}><X size={24} /></button>
            </div>
            <div style={{ padding: '24px', overflowY: 'auto' }}>
              {!username ? (
                <div>
                  <p style={{ color: 'var(--text-muted)', marginBottom: '16px', lineHeight: '1.5' }}>
                    Enter a username to save your chat sessions and review them later. No password required.
                  </p>
                  <input 
                    type="text" 
                    className="input-glass" 
                    placeholder="Username" 
                    value={tempUsername}
                    onChange={(e) => setTempUsername(e.target.value)}
                    style={{ marginBottom: '16px' }}
                  />
                  <button className="button-primary" style={{ width: '100%' }} onClick={() => {
                    if (tempUsername.trim()) {
                      setUsername(tempUsername.trim());
                      localStorage.setItem('username', tempUsername.trim());
                    }
                  }}>
                    Save Profile
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', alignItems: 'center' }}>
                  <p style={{ color: 'var(--text-main)', fontSize: '1.1rem' }}>You are logged in as <strong>{username}</strong>.</p>
                  <p style={{ color: 'var(--text-muted)' }}>Your chat history is displayed in the right sidebar.</p>
                  <button 
                    className="button-icon" 
                    style={{ marginTop: '16px', color: '#ef4444', border: '1px solid #ef4444', padding: '8px 16px', borderRadius: '8px' }}
                    onClick={() => {
                      setUsername('');
                      localStorage.removeItem('username');
                      setUserSessions([]);
                      setIsLoginModalOpen(false);
                    }}
                  >
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
