import { useState, useEffect } from 'react';
import { Sparkles, Download, RefreshCw, Loader2, AlertCircle, Check, ChevronDown, Terminal, Eye, EyeOff } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiFetch } from '../lib/api';
import SegmentedControl from './ui/SegmentedControl';

const ASPECT_OPTIONS = [
  { value: '9:16', label: 'Vertical (9:16)' },
  { value: '16:9', label: 'Landscape (16:9)' },
  { value: '1:1', label: 'Square (1:1)' },
];

const LANGUAGE_OPTIONS = [
  { value: '', label: 'Auto' },
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Español' },
  { value: 'pt', label: 'Português' },
  { value: 'fr', label: 'Français' },
  { value: 'de', label: 'Deutsch' },
  { value: 'it', label: 'Italiano' },
];

// Small self-contained BYOK key field. topic_video doesn't have a dedicated
// Settings section (v1 scope is one tab), so both keys live inline here —
// caching still goes through the same localStorage pattern App.jsx already
// uses for falKey/elevenLabsKey (see openaiKey_v1/pexelsKey_v1 there).
function InlineKeyField({ label, value, onChange, placeholder, helpUrl, helpLabel }) {
  const [visible, setVisible] = useState(false);
  return (
    <div>
      <label className="eyebrow block mb-2">{label}</label>
      <div className="relative">
        <input
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="input-field pr-10 font-mono text-sm"
        />
        <button
          type="button"
          onClick={() => setVisible((v) => !v)}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
        >
          {visible ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
      {helpUrl && (
        <a href={helpUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-brass hover:underline mt-1 inline-block">
          {helpLabel || 'Get an API key →'}
        </a>
      )}
    </div>
  );
}

export default function TopicToVideoTab({ openaiKey, setOpenaiKey, pexelsKey, setPexelsKey }) {
  const [topic, setTopic] = useState('');
  const [videoLanguage, setVideoLanguage] = useState('');
  const [videoAspect, setVideoAspect] = useState('9:16');
  const [subtitleEnabled, setSubtitleEnabled] = useState(true);
  const [voices, setVoices] = useState([]);
  const [voiceName, setVoiceName] = useState('');

  const [generating, setGenerating] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle'); // idle | processing | completed | failed
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [logsExpanded, setLogsExpanded] = useState(true);

  const hasKeys = !!openaiKey && !!pexelsKey;

  // Fetch the curated edge-tts voice list once.
  useEffect(() => {
    fetch(getApiUrl('/api/topicvideo/voices'))
      .then((res) => (res.ok ? res.json() : { voices: [] }))
      .then((data) => {
        const list = data.voices || [];
        setVoices(list);
        if (list.length > 0) setVoiceName((prev) => prev || list[0].voice_name);
      })
      .catch(() => {});
  }, []);

  // Poll generation status.
  useEffect(() => {
    let interval;
    if (jobId && status === 'processing') {
      interval = setInterval(async () => {
        try {
          const res = await apiFetch(`/api/topicvideo/status/${jobId}`);
          if (res.status === 404) {
            setStatus('failed');
            setGenerating(false);
            setLogs((prev) => [...prev, 'Job lost after server restart.']);
            clearInterval(interval);
            return;
          }
          if (!res.ok) return;
          const data = await res.json();
          if (data.logs) setLogs(data.logs);
          if (data.status === 'completed') {
            setStatus('completed');
            setResult(data.result);
            setGenerating(false);
            clearInterval(interval);
          } else if (data.status === 'failed') {
            setStatus('failed');
            setGenerating(false);
            clearInterval(interval);
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, status]);

  const filteredVoices = videoLanguage
    ? voices.filter((v) => v.language === videoLanguage)
    : voices;

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    if (!openaiKey || !pexelsKey) {
      setError('OpenAI and Pexels API keys are both required.');
      return;
    }

    setGenerating(true);
    setStatus('processing');
    setLogs(['Topic-to-Video job started.']);
    setResult(null);
    setError('');

    try {
      const res = await apiFetch('/api/topicvideo/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-OpenAI-Key': openaiKey,
          'X-Pexels-Key': pexelsKey,
        },
        body: JSON.stringify({
          topic: topic.trim(),
          video_language: videoLanguage,
          voice_name: voiceName || undefined,
          video_aspect: videoAspect,
          subtitle_enabled: subtitleEnabled,
        }),
      });

      if (!res.ok) {
        let msg = 'Generation failed';
        try { const err = await res.json(); msg = err.detail || msg; } catch { msg = (await res.text()) || msg; }
        throw new Error(msg);
      }

      const data = await res.json();
      setJobId(data.job_id);
    } catch (e) {
      setStatus('failed');
      setError(e.message);
      setGenerating(false);
    }
  };

  const handleReset = () => {
    setTopic('');
    setJobId(null);
    setLogs([]);
    setStatus('idle');
    setResult(null);
    setError('');
    setGenerating(false);
  };

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="max-w-3xl mx-auto p-4 sm:p-6 lg:p-8">
        <div className="flex items-end justify-between mb-2">
          <div>
            <p className="eyebrow mb-2">04 · TOPIC TO VIDEO</p>
            <h1 className="font-display lowercase text-2xl text-ink">Topic to Video</h1>
          </div>
          {status !== 'idle' && (
            <button onClick={handleReset} className="text-xs lowercase text-muted hover:text-ink flex items-center gap-1 transition-colors">
              <RefreshCw size={12} /> Start over
            </button>
          )}
        </div>
        <p className="text-sm lowercase text-muted mb-6">
          Generate a narrated short video from a single topic — script, voiceover, stock footage and subtitles, fully automatic.
        </p>

        {(status === 'idle' || status === 'failed') && (
          <div className="animate-fade space-y-6">
            <div className="card p-4 sm:p-8 space-y-6">
              <div>
                <label className="eyebrow block mb-2">Topic</label>
                <textarea
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  rows={3}
                  className="input-field resize-none text-sm"
                  placeholder="e.g. the history of coffee, 5 tips for better sleep, why octopuses are so smart..."
                />
              </div>

              <div>
                <label className="eyebrow block mb-3">Language</label>
                <SegmentedControl options={LANGUAGE_OPTIONS} value={videoLanguage} onChange={setVideoLanguage} />
              </div>

              <div>
                <label className="eyebrow block mb-2">Voice</label>
                {filteredVoices.length > 0 ? (
                  <select value={voiceName} onChange={(e) => setVoiceName(e.target.value)} className="input-field">
                    {filteredVoices.map((v) => (
                      <option key={v.voice_name} value={v.voice_name}>{v.name}</option>
                    ))}
                  </select>
                ) : (
                  <p className="text-xs text-muted">Loading voices...</p>
                )}
              </div>

              <div>
                <label className="eyebrow block mb-3">Aspect Ratio</label>
                <SegmentedControl options={ASPECT_OPTIONS} value={videoAspect} onChange={setVideoAspect} />
              </div>

              <label className="flex items-center gap-2 text-sm text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={subtitleEnabled}
                  onChange={(e) => setSubtitleEnabled(e.target.checked)}
                  className="accent-brass"
                />
                Burn in subtitles
              </label>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-rule">
                <InlineKeyField
                  label="OpenAI API Key"
                  value={openaiKey}
                  onChange={setOpenaiKey}
                  placeholder="sk-..."
                  helpUrl="https://platform.openai.com/api-keys"
                  helpLabel="Get an OpenAI API key →"
                />
                <InlineKeyField
                  label="Pexels API Key"
                  value={pexelsKey}
                  onChange={setPexelsKey}
                  placeholder="563492ad6f9170..."
                  helpUrl="https://www.pexels.com/api/"
                  helpLabel="Get a free Pexels API key →"
                />
              </div>

              {(error || !hasKeys) && (
                <div className="flex items-center gap-2 text-sm text-danger bg-danger/10 rounded-input p-3">
                  <AlertCircle size={14} />
                  {error || 'Both API keys are required to generate a video.'}
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={generating || !topic.trim() || !hasKeys}
                className="btn-primary w-full"
              >
                {generating ? (
                  <><Loader2 size={16} className="animate-spin" /> Starting...</>
                ) : (
                  <><Sparkles size={16} className="hidden sm:block" /> Generate video</>
                )}
              </button>
            </div>
          </div>
        )}

        {status === 'processing' && (
          <div className="animate-fade space-y-6">
            <div className="card p-6">
              <div className="flex items-center justify-between mb-5">
                <h2 className="font-display lowercase text-xl text-ink">Generating Video</h2>
                <span className="badge-brass">PROCESSING</span>
              </div>

              <div className="border-y border-rule divide-y divide-rule mb-4">
                {['Script', 'Search terms', 'Narration (edge-tts)', 'Subtitles', 'Stock footage (Pexels)', 'Final render'].map((label, i) => {
                  const logStr = logs.join(' ').toLowerCase();
                  const marker = `[${i + 1}/6]`;
                  const nextMarker = `[${i + 2}/6]`;
                  const stepDone = logStr.includes(nextMarker) || (i === 5 && status !== 'processing');
                  const stepActive = logStr.includes(marker) && !stepDone;
                  return (
                    <div key={label} className="flex items-center gap-3 py-2.5 text-sm">
                      <span className="font-mono text-micro text-muted w-5 shrink-0">{String(i + 1).padStart(2, '0')}</span>
                      {stepDone ? (
                        <Check size={14} className="text-ok shrink-0" />
                      ) : stepActive ? (
                        <Loader2 size={14} className="text-brass animate-spin shrink-0" />
                      ) : (
                        <span className="w-3.5 h-3.5 rounded-full border border-rule shrink-0" />
                      )}
                      <span className={`lowercase ${stepDone ? 'text-muted' : stepActive ? 'text-ink' : 'text-muted/60'}`}>{label}</span>
                    </div>
                  );
                })}
              </div>

              <div className="bg-paper rounded-card border border-rule overflow-hidden">
                <div className="px-4 py-2 border-b border-rule flex items-center justify-between">
                  <span className="readout flex items-center gap-2"><Terminal size={12} /> Generation Logs</span>
                  <button onClick={() => setLogsExpanded(!logsExpanded)} className="text-muted hover:text-ink transition-colors">
                    <ChevronDown size={14} className={logsExpanded ? '' : 'rotate-180'} />
                  </button>
                </div>
                {logsExpanded && (
                  <div className="p-4 max-h-64 overflow-y-auto font-mono text-xs space-y-1 custom-scrollbar">
                    {logs.map((l, i) => (
                      <div key={i} className={l.toLowerCase().includes('error') ? 'text-danger' : 'text-muted'}>{l}</div>
                    ))}
                    <div className="animate-pulse text-brass">_</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {status === 'completed' && result && (
          <div className="animate-fade space-y-6">
            <div className="card p-6">
              <h2 className="font-display lowercase text-xl text-ink mb-4">Your Video is Ready</h2>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className={`card ${videoAspect === '9:16' ? 'aspect-[9/16]' : videoAspect === '1:1' ? 'aspect-square' : 'aspect-video'} max-h-[500px] bg-black overflow-hidden relative`}>
                  <video src={getApiUrl(result.video_url)} controls className="w-full h-full object-contain" autoPlay />
                </div>

                <div className="space-y-4">
                  <p className="readout">{result.duration ? `${result.duration}s` : ''} &middot; {videoAspect}</p>

                  {result.script && (
                    <div>
                      <span className="eyebrow block mb-1.5">Script</span>
                      <p className="text-xs text-ink2 bg-paper border border-rule rounded-input p-2.5 max-h-40 overflow-y-auto whitespace-pre-wrap">{result.script}</p>
                    </div>
                  )}

                  {result.terms && result.terms.length > 0 && (
                    <div>
                      <span className="eyebrow block mb-1.5">Search Terms</span>
                      <div className="flex flex-wrap gap-1.5">
                        {result.terms.map((t, i) => (
                          <span key={i} className="readout bg-paper3 px-2 py-0.5 rounded-full">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-3 pt-2">
                    <a href={getApiUrl(result.video_url)} download className="btn-primary px-4 py-2 text-sm">
                      <Download size={14} /> Download
                    </a>
                    <button onClick={handleReset} className="btn-ghost px-4 py-2 text-sm">
                      <RefreshCw size={14} /> New video
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
