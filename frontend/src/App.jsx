import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Search, Settings, Send, Scale, ChevronRight, X, Loader2, Book } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './index.css';

function App() {
  const [query, setQuery] = useState('');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome to Judge Read. How can I assist you with your legal research today?', sources: null }
  ]);
  const messagesEndRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  
  // Full Case Modal State
  const [selectedCase, setSelectedCase] = useState(null);
  const [isCaseLoading, setIsCaseLoading] = useState(false);

  // Settings state
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-3-small');
  const [embeddingKey, setEmbeddingKey] = useState('');
  const [llmEngine, setLlmEngine] = useState('claude');
  const [apiKey, setApiKey] = useState('');
  
  // Metadata Filters
  const [filterYear, setFilterYear] = useState('');
  const [filterCourt, setFilterCourt] = useState('');
  const [filterSystem, setFilterSystem] = useState('');
  const [filterState, setFilterState] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
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

  useEffect(() => {
    // Load config on mount
    axios.get(`http://${window.location.hostname}:8000/api/config`).then((response) => {
      const data = response.data;
      if (data.embeddingModel) setEmbeddingModel(data.embeddingModel);
      if (data.embeddingKey) setEmbeddingKey(data.embeddingKey);
      if (data.llmEngine) setLlmEngine(data.llmEngine);
      if (data.apiKey) setApiKey(data.apiKey);
      if (data.langsmithKey) setLangsmithKey(data.langsmithKey);
      if (data.cohereKey) setCohereKey(data.cohereKey);
      if (data.pgHost) setPgHost(data.pgHost);
      if (data.pgPort) setPgPort(data.pgPort);
      if (data.pgUser) setPgUser(data.pgUser);
      if (data.pgPassword) setPgPassword(data.pgPassword);
      if (data.pgDb) setPgDb(data.pgDb);
      setIsConfigLoaded(true);
    }).catch((err) => {
      console.error("Could not load config", err);
      setIsConfigLoaded(true);
    });
  }, []);

  // Save config whenever it changes
  useEffect(() => {
    if (isConfigLoaded) {
      axios.post(`http://${window.location.hostname}:8000/api/config`, {
        embeddingModel,
        embeddingKey,
        llmEngine,
        apiKey,
        langsmithKey,
        cohereKey,
        pgHost,
        pgPort,
        pgUser,
        pgPassword,
        pgDb
      }).catch(err => console.error("Failed to save config", err));
    }
  }, [embeddingModel, embeddingKey, llmEngine, apiKey, langsmithKey, cohereKey, pgHost, pgPort, pgUser, pgPassword, pgDb, isConfigLoaded]);

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

      const response = await axios.get(`http://${window.location.hostname}:8000/api/cases?${params.toString()}`);
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

  const fetchFullCase = async (caseId) => {
    if (!caseId) return;
    setIsCaseLoading(true);
    setSelectedCase(null);
    try {
      const response = await axios.get(`http://${window.location.hostname}:8000/api/cases/${caseId}`);
      setSelectedCase(response.data);
    } catch (error) {
      console.error("Failed to fetch full case", error);
      alert("Sorry, could not load the full text for this case.");
    } finally {
      setIsCaseLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      const response = await axios.post(`http://${window.location.hostname}:8000/api/search`, {
        query: userMessage.content,
        session_id: sessionId,
        embedding_model: embeddingModel,
        embedding_key: embeddingKey,
        llm_engine: llmEngine,
        api_key: apiKey,
        filter_year: filterYear ? parseInt(filterYear) : null,
        filter_court: filterCourt ? filterCourt : null,
        filter_jurisdiction: filterSystem === 'Federal' ? 'Federal' : (filterSystem === 'State' ? (filterState || 'State') : null),
        filter_status: filterStatus ? filterStatus : null,
        filter_judge: filterJudge ? filterJudge : null,
        filter_topic: filterTopic ? filterTopic : null,
        langsmith_key: langsmithKey ? langsmithKey : null,
        cohere_key: cohereKey ? cohereKey : null
      });

      if (!sessionId && response.data.session_id) {
        setSessionId(response.data.session_id);
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources
      }]);
    } catch (error) {
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
    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', width: '100%' }}>
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

      <select 
        className="input-glass" 
        value={filterStatus} 
        onChange={(e) => setFilterStatus(e.target.value)}
        style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', color: filterStatus === 'good_law' ? '#51cf66' : 'inherit', flex: 1, minWidth: '150px' }}
      >
        <option value="">All Precedent</option>
        <option value="good_law">Good Law Only</option>
      </select>
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
              <option value="text-embedding-3-small">text-embedding-3-small (OpenAI)</option>
              <option value="text-embedding-3-large">text-embedding-3-large (OpenAI)</option>
              <option value="claude">Claude Embeddings</option>
              <option value="ollama">Ollama Embeddings</option>
            </select>
            <input 
              type={embeddingModel === 'ollama' ? 'text' : 'password'}
              className="input-glass" 
              placeholder={embeddingModel === 'ollama' ? 'Host (e.g. http://localhost:11434)' : 'API Key'}
              value={embeddingKey} 
              onChange={(e) => setEmbeddingKey(e.target.value)} 
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>LLM Engine</label>
            <select className="input-glass" value={llmEngine} onChange={(e) => setLlmEngine(e.target.value)}>
              <option value="gpt-5.5-pro">GPT-5.5 Pro</option>
              <option value="gpt-5.5">GPT-5.5</option>
              <option value="chat-latest">Chat Latest (OpenAI)</option>
              <option value="claude-sonnet-5">Claude Sonnet 5</option>
              <option value="claude-fable-5">Claude Fable 5</option>
              <option value="claude-opus-4-8">Claude Opus 4.8</option>
              <option value="o1">OpenAI o1</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              {llmEngine === 'ollama' ? 'Host URL' : 'API Key'}
            </label>
            <input 
              type={llmEngine === 'ollama' ? 'text' : 'password'}
              className="input-glass" 
              placeholder={llmEngine === 'ollama' ? 'http://localhost:11434' : 'sk-...'}
              value={apiKey} 
              onChange={(e) => setApiKey(e.target.value)} 
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
                <input type="password" className="input-glass" value={pgPassword} onChange={(e) => setPgPassword(e.target.value)} />
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
            />

            <h3 style={{ fontSize: '1rem', fontWeight: 500, marginBottom: '12px' }}>LangSmith Tracing</h3>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>LangSmith API Key</label>
            <input 
              type="password"
              className="input-glass" 
              placeholder="lsv2_pt_..."
              value={langsmithKey} 
              onChange={(e) => setLangsmithKey(e.target.value)} 
            />
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', width: '100%', maxWidth: '1200px', margin: '0 auto' }}>
        
        {/* Header */}
        <header style={{ 
          padding: '20px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          borderBottom: '1px solid var(--border-color)', background: 'rgba(11, 15, 25, 0.8)', backdropFilter: 'blur(10px)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ background: 'var(--accent)', padding: '8px', borderRadius: '12px', boxShadow: '0 0 15px var(--accent-glow)' }}>
              <Scale size={24} color="white" />
            </div>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 600, letterSpacing: '-0.5px' }}>Judge Read</h1>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button className="button-icon" onClick={() => setIsExplorerOpen(true)} title="Case Explorer">
              <Book size={22} />
            </button>
            <button className="button-icon" onClick={() => setIsSettingsOpen(true)}>
              <Settings size={22} />
            </button>
          </div>
        </header>

        {/* Chat History */}
        <div className="scroll-smooth" style={{ flex: 1, overflowY: 'auto', padding: '32px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {messages.map((msg, idx) => (
            <div key={idx} className={`animate-fade-in ${msg.role === 'user' ? 'user-msg' : 'assistant-msg'}`} style={{
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: '8px'
            }}>
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
                <ReactMarkdown className="markdown-body">
                  {msg.content}
                </ReactMarkdown>
              </div>
              
              {/* Sources render if assistant and has sources */}
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
            <div className="animate-fade-in" style={{ alignSelf: 'flex-start', padding: '16px 20px', borderRadius: '16px', background: 'var(--panel-bg)', border: '1px solid var(--border-color)' }}>
              <Loader2 className="spinner" size={20} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{ padding: '24px 32px', background: 'linear-gradient(to top, rgba(11,15,25,1) 50%, rgba(11,15,25,0))' }}>
          
          {/* External Search Filters */}
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
          <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Retrieval-Augmented Generation relies on localized case repository to prevent hallucinations.
          </div>
        </div>

      </div>

      {/* Case Explorer Modal */}
      {isExplorerOpen && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)', backdropFilter: 'blur(8px)',
          zIndex: 90, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '20px'
        }}>
          <div className="glass-panel animate-fade-in" style={{
            width: '100%', maxWidth: '1200px', height: '90vh', display: 'flex', flexDirection: 'column',
            background: 'var(--panel-bg)', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
          }}>
            <div style={{ padding: '24px 32px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Book size={24} color="var(--accent)" />
                <h2 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-main)' }}>Case Explorer</h2>
              </div>
              <button className="button-icon" onClick={() => setIsExplorerOpen(false)}>
                <X size={24} />
              </button>
            </div>
            
            <div style={{ padding: '24px 32px', borderBottom: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
                <Search size={20} color="var(--text-muted)" style={{ position: 'absolute', marginLeft: '16px' }} />
                <input 
                  type="text" 
                  value={explorerSearch}
                  onChange={(e) => setExplorerSearch(e.target.value)}
                  placeholder="Search case name..."
                  className="input-glass"
                  style={{ width: '100%', padding: '10px 16px 10px 44px', borderRadius: '8px' }}
                />
              </div>
              
              <FilterControls />
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '0' }}>
              {isExplorerLoading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}>
                  <Loader2 className="spinner" size={32} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                </div>
              ) : explorerCases.length === 0 ? (
                <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No cases found. Try adjusting your filters.
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-main)', zIndex: 1 }}>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <th style={{ padding: '16px 24px', color: 'var(--text-muted)', fontWeight: 500, fontSize: '0.9rem' }}>Case Name</th>
                      <th style={{ padding: '16px 24px', color: 'var(--text-muted)', fontWeight: 500, fontSize: '0.9rem' }}>Year</th>
                      <th style={{ padding: '16px 24px', color: 'var(--text-muted)', fontWeight: 500, fontSize: '0.9rem' }}>Court</th>
                      <th style={{ padding: '16px 24px', color: 'var(--text-muted)', fontWeight: 500, fontSize: '0.9rem' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explorerCases.map(c => (
                      <tr 
                        key={c.case_id} 
                        onClick={() => fetchFullCase(c.case_id)}
                        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', cursor: 'pointer', transition: 'background 0.2s ease' }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                      >
                        <td style={{ padding: '16px 24px', color: 'var(--accent)', fontWeight: 500 }}>{c.name}</td>
                        <td style={{ padding: '16px 24px' }}>{c.year}</td>
                        <td style={{ padding: '16px 24px' }}>
                          <div style={{ fontSize: '0.95rem' }}>{c.court}</div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{c.jurisdiction}</div>
                        </td>
                        <td style={{ padding: '16px 24px' }}>
                          {c.status === 'good_law' ? (
                            <span style={{ color: '#51cf66', background: 'rgba(81, 207, 102, 0.1)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>Good Law</span>
                          ) : c.status === 'overruled' ? (
                            <span style={{ color: '#ff6b6b', background: 'rgba(255, 107, 107, 0.1)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>Overruled</span>
                          ) : (
                            <span style={{ color: '#fcc419', background: 'rgba(252, 196, 25, 0.1)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>Caution</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Full Case Modal */}
      {(selectedCase || isCaseLoading) && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.7)', backdropFilter: 'blur(4px)',
          zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '40px'
        }}>
          <div className="glass-panel animate-fade-in" style={{
            width: '100%', maxWidth: '900px', maxHeight: '100%', display: 'flex', flexDirection: 'column',
            background: 'var(--panel-bg)', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
          }}>
            <div style={{ padding: '24px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {selectedCase ? selectedCase.name : 'Loading Document...'}
                </h2>
                {selectedCase && (
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    {selectedCase.reporter} • {selectedCase.court} • {selectedCase.year}
                  </div>
                )}
              </div>
              <button className="button-icon" onClick={() => setSelectedCase(null) || setIsCaseLoading(false)}>
                <X size={24} />
              </button>
            </div>
            
            <div style={{ flex: 1, overflowY: 'auto', padding: '32px', fontSize: '1rem', lineHeight: '1.8', color: 'var(--text-main)' }}>
              {isCaseLoading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}>
                  <Loader2 className="spinner" size={32} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                </div>
              ) : (
                <div className="markdown-body" style={{ overflowWrap: 'anywhere' }}>
                  {(() => {
                    let parsed = null;
                    try {
                      let rawText = selectedCase.full_text;
                      // Clean up unescaped control characters from postgres json dumps
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
                      // Fallback if not JSON
                    }

                    if (parsed) {
                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '24px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <h3 style={{ marginTop: 0, color: 'var(--accent)', fontSize: '1.1rem' }}>Metadata</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '12px 16px', fontSize: '0.95rem' }}>
                              {parsed.case_name_full && <><span style={{ color: 'var(--text-muted)' }}>Case Name:</span><span>{parsed.case_name_full}</span></>}
                              {parsed.date_filed && <><span style={{ color: 'var(--text-muted)' }}>Date Filed:</span><span>{parsed.date_filed}</span></>}
                              {parsed.court_full_name && <><span style={{ color: 'var(--text-muted)' }}>Court:</span><span>{parsed.court_full_name}</span></>}
                              {parsed.judges && <><span style={{ color: 'var(--text-muted)' }}>Judges:</span><span>{parsed.judges}</span></>}
                              {parsed.attorneys && <><span style={{ color: 'var(--text-muted)' }}>Attorneys:</span><span>{parsed.attorneys}</span></>}
                              {parsed.citations && parsed.citations.length > 0 && <><span style={{ color: 'var(--text-muted)' }}>Citations:</span><span>{parsed.citations.join(', ')}</span></>}
                            </div>
                          </div>
                          
                          {parsed.summary && (
                            <div>
                              <h3 style={{ color: 'var(--accent)' }}>Summary</h3>
                              <ReactMarkdown>{parsed.summary}</ReactMarkdown>
                            </div>
                          )}
                          
                          {parsed.syllabus && (
                            <div>
                              <h3 style={{ color: 'var(--accent)' }}>Syllabus</h3>
                              <ReactMarkdown>{parsed.syllabus}</ReactMarkdown>
                            </div>
                          )}

                          {parsed.headnotes && (
                            <div>
                              <h3 style={{ color: 'var(--accent)' }}>Headnotes</h3>
                              <ReactMarkdown>{parsed.headnotes}</ReactMarkdown>
                            </div>
                          )}

                          {parsed.headmatter && (
                            <div>
                              <h3 style={{ color: 'var(--accent)' }}>Headmatter</h3>
                              <ReactMarkdown>{parsed.headmatter}</ReactMarkdown>
                            </div>
                          )}
                          
                          {parsed.opinions && parsed.opinions.length > 0 ? (
                            <div>
                              <h3 style={{ color: 'var(--accent)' }}>Opinions</h3>
                              {parsed.opinions.map((o, idx) => (
                                <div key={idx} style={{ marginTop: '16px' }}>
                                  {o.author_str && <h4 style={{ color: 'var(--text-muted)' }}>Author: {o.author_str}</h4>}
                                  <ReactMarkdown>{o.opinion_text}</ReactMarkdown>
                                  {idx < parsed.opinions.length - 1 && <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '32px 0' }}/>}
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      );
                    }

                    return <ReactMarkdown>{selectedCase.full_text}</ReactMarkdown>;
                  })()}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .hover-pill:hover { background: rgba(255,255,255,0.1) !important; color: white !important; transform: translateY(-1px); }
      `}</style>
    </div>
  );
}

export default App;
