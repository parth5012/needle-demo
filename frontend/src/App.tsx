import { useState, useEffect } from 'react';
import {
  Sparkles,
  Bot,
  CheckCircle2,
  AlertCircle,
  FileText,
  Clock,
  ShieldCheck,
  Send,
  RefreshCw,
  Award,
  MapPin,
  Phone,
  User,
  Hash,
  BookOpen,
  Code,
  Zap,
  SlidersHorizontal,
  IdCard,
} from 'lucide-react';
import type {
  ArtisanOnboardingForm,
  QuickArtisanOnboardingForm,
  ExtractionResponse,
  SamplePrompt,
  ExtractedFieldMeta
} from './types';

const INITIAL_FORM: ArtisanOnboardingForm = {
  name: '',
  phone: '',
  pin: '',
  primary_dialect: 'hindi',
  craft_category: '',
  state: '',
  district: '',
  cluster_name: '',
  trifed_id: '',
  pehchan_id: '',
  gi_tag_name: '',
  gi_certified: false,
  years_experience: null,
  skills_summary: ''
};

const INITIAL_QUICK_FORM: QuickArtisanOnboardingForm = {
  name: '',
  phone: '',
  pehchan_id: ''
};

export function App() {
  const [mode, setMode] = useState<'quick' | 'full'>('quick');
  const [engineMode, setEngineMode] = useState<'fast' | 'neural'>('fast');
  const [inputText, setInputText] = useState('');
  const [samplePrompts, setSamplePrompts] = useState<SamplePrompt[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionResult, setExtractionResult] = useState<ExtractionResponse | null>(null);
  
  // Full Form Data
  const [formData, setFormData] = useState<ArtisanOnboardingForm>(INITIAL_FORM);
  // Quick Form Data
  const [quickFormData, setQuickFormData] = useState<QuickArtisanOnboardingForm>(INITIAL_QUICK_FORM);

  const [fieldMetadata, setFieldMetadata] = useState<Record<string, ExtractedFieldMeta>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'form' | 'json' | 'entities'>('form');

  // Load sample prompts on mount
  useEffect(() => {
    fetch('/api/sample-prompts')
      .then((res) => res.json())
      .then((data) => {
        setSamplePrompts(data);
        if (data && data.length > 0) {
          handleSelectPrompt(data[0]);
        }
      })
      .catch((err) => {
        console.warn('Backend API not reachable directly, using offline presets', err);
        const fallback: SamplePrompt[] = [
          {
            id: "varanasi-weaver",
            title: "Varanasi Silk Weaver (Bhojpuri)",
            dialect: "bhojpuri",
            category: "Handloom Textile",
            text: "Pranam! Humar naam Gauri Devi ba. Hum Varanasi Handloom Cluster me pichhle 18 years se authentic Banarasi brocade aur pure silk saree bunat baani. Contact number 9876543210 ba aur humar security pin 1234. Humaar government Pehchan ID PEH-IND-88320 ha aur TRIFED registration number TRIFED-UP-VNS-1049 ba. Banarasi Brocade GI tagged craft."
          },
          {
            id: "channapatna-toys",
            title: "Channapatna Toy Crafter (Kannada/English)",
            dialect: "kannada",
            category: "Wooden Craft & Toys",
            text: "Namaskara, I am Ramesh Gowda, a traditional lacquerware artisan from Channapatna Craft Cluster, Ramanagara, Karnataka. 12 years of experience crafting Channapatna wooden toys. Phone: 9123456780, pin: 4321. Pehchan card is PEH-KA-44912, TRIFED ID is TRIFED-KA-CHN-8821. GI certified toys."
          }
        ];
        setSamplePrompts(fallback);
        handleSelectPrompt(fallback[0]);
      });
  }, []);

  const handleSelectPrompt = (p: SamplePrompt) => {
    setSelectedPromptId(p.id);
    setInputText(p.text);
    setSubmitSuccess(null);
    setError(null);
  };

  const handleExtract = async () => {
    if (!inputText.trim()) return;
    setIsExtracting(true);
    setError(null);
    setSubmitSuccess(null);

    try {
      const res = await fetch('/api/extract-artisan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          model_name: engineMode === 'neural' ? 'needle-3-local' : 'needle-3-heuristic'
        })
      });

      if (!res.ok) {
        throw new Error(`Extraction failed: ${res.statusText}`);
      }

      const data: ExtractionResponse = await res.json();
      setExtractionResult(data);
      
      // Update full form
      setFormData({
        name: data.form_data.name || '',
        phone: data.form_data.phone || '',
        pin: data.form_data.pin || '',
        primary_dialect: data.form_data.primary_dialect || 'hindi',
        craft_category: data.form_data.craft_category || '',
        state: data.form_data.state || '',
        district: data.form_data.district || '',
        cluster_name: data.form_data.cluster_name || '',
        trifed_id: data.form_data.trifed_id || '',
        pehchan_id: data.form_data.pehchan_id || '',
        gi_tag_name: data.form_data.gi_tag_name || '',
        gi_certified: data.form_data.gi_certified || false,
        years_experience: data.form_data.years_experience ?? null,
        skills_summary: data.form_data.skills_summary || ''
      });

      // Update quick form
      setQuickFormData({
        name: data.form_data.name || '',
        phone: data.form_data.phone || '',
        pehchan_id: data.form_data.pehchan_id || ''
      });

      setFieldMetadata(data.field_metadata || {});
    } catch (err: any) {
      setError(err.message || 'Error running Needle 3 extraction');
    } finally {
      setIsExtracting(false);
    }
  };

  const handleQuickInputChange = (field: keyof QuickArtisanOnboardingForm, value: any) => {
    setQuickFormData((prev) => ({ ...prev, [field]: value }));
    // Also sync to full form
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleFullInputChange = (field: keyof ArtisanOnboardingForm, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (field === 'name' || field === 'phone' || field === 'pehchan_id') {
      setQuickFormData((prev) => ({ ...prev, [field]: value }));
    }
  };

  const handleQuickSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSubmitSuccess(null);

    try {
      const res = await fetch('/api/submit-quick-onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(quickFormData)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || 'Submission failed');
      }

      const data = await res.json();
      setSubmitSuccess(`Quick profile created! ID: ${data.artisan_id} (${data.verification_status})`);
    } catch (err: any) {
      setError(err.message || 'Error submitting quick onboarding profile');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFullSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSubmitSuccess(null);

    try {
      const res = await fetch('/api/submit-onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || 'Submission failed');
      }

      const data = await res.json();
      setSubmitSuccess(`Full profile registered successfully! ID: ${data.artisan_id} (${data.verification_status})`);
    } catch (err: any) {
      setError(err.message || 'Error submitting full onboarding profile');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderConfidenceBadge = (fieldKey: string) => {
    const meta = fieldMetadata[fieldKey];
    if (!meta || meta.confidence === 0) return null;
    const pct = Math.round(meta.confidence * 100);
    const color =
      pct >= 90
        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
        : pct >= 75
        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
        : 'bg-slate-700/50 text-slate-300 border-slate-600';

    return (
      <span
        title={`Extracted from: ${meta.source_segment || 'context'}`}
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 text-[11px] font-mono font-semibold rounded border ${color}`}
      >
        <Sparkles className="w-2.5 h-2.5" />
        {pct}%
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-orange-500 via-amber-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-orange-500/20">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-lg tracking-tight text-white">Needle 3 Demo</h1>
              <span className="px-2 py-0.5 text-[11px] font-mono bg-orange-500/20 text-orange-400 border border-orange-500/30 rounded-full">
                Artify Bharat
              </span>
            </div>
            <p className="text-xs text-slate-400">Zero-Shot Unstructured Text to Structured Form Filling</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Mode Switcher */}
          <div className="bg-slate-900 p-1 rounded-xl border border-slate-800 flex items-center gap-1">
            <button
              onClick={() => setMode('quick')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition cursor-pointer ${
                mode === 'quick'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Quick Onboarding (3 Fields)
            </button>
            <button
              onClick={() => setMode('full')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition cursor-pointer ${
                mode === 'full'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-emerald-400" />
              Full Schema (All Fields)
            </button>
          </div>

          <a
            href="https://github.com/parth5012/needle-demo"
            target="_blank"
            rel="noreferrer"
            className="text-xs text-slate-400 hover:text-white transition px-2 py-1 rounded hover:bg-slate-800"
          >
            GitHub Repo →
          </a>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Natural Language Input & Prompt Presets */}
        <section className="lg:col-span-5 flex flex-col gap-4">
          
          {/* Preset Chips */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 shadow-sm">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5 text-amber-400" />
              Artisan Story Presets (Dialect / Craft)
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {samplePrompts.map((p) => (
                <button
                  key={p.id}
                  onClick={() => handleSelectPrompt(p)}
                  className={`text-left p-2.5 rounded-xl border text-xs transition flex flex-col justify-between cursor-pointer ${
                    selectedPromptId === p.id
                      ? 'bg-amber-500/10 border-amber-500/50 text-amber-200'
                      : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <span className="font-semibold">{p.title}</span>
                  <span className="text-[10px] text-slate-500 mt-1 uppercase tracking-wider">{p.category}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Text Input Panel */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 flex-1 flex flex-col shadow-sm">
            {/* Header: Label + Engine Mode Toggle */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-orange-400" />
                Raw Artisan Narrative / Voice Transcript
              </label>

              {/* Extraction Engine Toggle */}
              <div className="flex items-center gap-2">
                <div className="bg-slate-900 p-0.5 rounded-lg border border-slate-800 flex items-center">
                  <button
                    type="button"
                    onClick={() => setEngineMode('fast')}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium transition cursor-pointer flex items-center gap-1 ${
                      engineMode === 'fast'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                    title="Instant regex & domain cluster lookup (~50ms)"
                  >
                    <Zap className="w-3 h-3 text-emerald-400" />
                    ⚡ Fast Mode (~50ms)
                  </button>
                  <button
                    type="button"
                    onClick={() => setEngineMode('neural')}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium transition cursor-pointer flex items-center gap-1 ${
                      engineMode === 'neural'
                        ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                    title="Real on-device Cactus Needle 3 foundation model inference (~15-25s)"
                  >
                    <Bot className="w-3 h-3 text-orange-400" />
                    🧠 Needle 3 Neural
                  </button>
                </div>

                <button
                  onClick={() => setInputText('')}
                  className="text-[11px] text-slate-500 hover:text-slate-300 transition cursor-pointer px-1"
                >
                  Clear
                </button>
              </div>
            </div>

            <textarea
              rows={8}
              value={inputText}
              onChange={(e) => {
                setInputText(e.target.value);
                setSelectedPromptId(null);
              }}
              placeholder="Paste natural language artisan introduction, phone, cluster, Pehchan ID, craft background, and GI details here..."
              className="w-full flex-1 p-3.5 text-sm bg-slate-900 border border-slate-800 rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition resize-none"
            />

            <div className="mt-4 flex items-center justify-between gap-3">
              <div className="text-xs text-slate-500">
                {inputText.trim().split(/\s+/).filter(Boolean).length} words • {inputText.length} chars
                {engineMode === 'neural' && (
                  <span className="ml-2 text-amber-400/80 font-mono text-[11px] hidden sm:inline">
                    (Neural token generation takes ~15s on CPU)
                  </span>
                )}
              </div>
              <button
                disabled={isExtracting || !inputText.trim()}
                onClick={handleExtract}
                className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ${
                  engineMode === 'neural'
                    ? 'bg-gradient-to-r from-orange-500 to-amber-600 hover:from-orange-600 hover:to-amber-700 shadow-orange-500/25'
                    : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 shadow-emerald-500/25'
                }`}
              >
                {isExtracting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    {engineMode === 'neural' ? 'Running Needle 3 Neural...' : 'Extracting (Fast)...'}
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    {engineMode === 'neural' ? 'Extract with Needle 3 Neural' : 'Instant Extract (Fast)'}
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Model Stats / Extraction Meta */}
          {extractionResult && (
            <div className="bg-slate-950/40 border border-slate-800/80 rounded-2xl p-4 text-xs flex items-center justify-between text-slate-400">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-emerald-400" />
                <span>Latency: <strong className="text-slate-200">{extractionResult.processing_time_ms} ms</strong></span>
              </div>
              <div>
                Overall Confidence: <strong className="text-emerald-400">{Math.round(extractionResult.overall_confidence * 100)}%</strong>
              </div>
              <div>
                Model: <span className="font-mono text-slate-300">{extractionResult.model_used}</span>
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Structured Form (Quick vs Full) */}
        <section className="lg:col-span-7 flex flex-col gap-4">
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col flex-1">
            
            {/* Form Header & Tabs */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-5">
              <div>
                <h2 className="font-bold text-base text-white flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  {mode === 'quick' ? 'Quick Artisan Onboarding (3 Essential Fields)' : 'Artify Bharat: Full Artisan Onboarding Record'}
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  {mode === 'quick' 
                    ? 'Simplified zero-friction signup: Name, Phone Number, and Government Pehchan ID' 
                    : 'Complete heritage artisan registry profile with geographical cluster and GI details'}
                </p>
              </div>

              {/* View Switcher */}
              <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs">
                <button
                  onClick={() => setActiveTab('form')}
                  className={`px-3 py-1 rounded-md transition cursor-pointer ${activeTab === 'form' ? 'bg-slate-800 text-white font-medium shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Form UI
                </button>
                <button
                  onClick={() => setActiveTab('json')}
                  className={`px-3 py-1 rounded-md transition flex items-center gap-1 cursor-pointer ${activeTab === 'json' ? 'bg-slate-800 text-white font-medium shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  <Code className="w-3 h-3" />
                  JSON
                </button>
                <button
                  onClick={() => setActiveTab('entities')}
                  className={`px-3 py-1 rounded-md transition cursor-pointer ${activeTab === 'entities' ? 'bg-slate-800 text-white font-medium shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Entities ({extractionResult?.extracted_entities.length || 0})
                </button>
              </div>
            </div>

            {/* Notification messages */}
            {submitSuccess && (
              <div className="mb-4 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
                <span>{submitSuccess}</span>
              </div>
            )}
            {error && (
              <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{error}</span>
              </div>
            )}

            {/* Form Content */}
            {activeTab === 'form' && (
              <>
                {/* QUICK ONBOARDING (3 FIELDS) */}
                {mode === 'quick' ? (
                  <form onSubmit={handleQuickSubmit} className="flex flex-col gap-6 flex-1 justify-between py-2">
                    <div className="space-y-5">
                      <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-center gap-2">
                        <Zap className="w-4 h-4 text-amber-400 shrink-0" />
                        <span>Needle 3 extracted the 3 essential parameters required for lightning-fast artisan onboarding.</span>
                      </div>

                      {/* Field 1: Name */}
                      <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 hover:border-slate-700 transition">
                        <div className="flex items-center justify-between mb-1.5">
                          <label className="text-xs text-slate-200 font-semibold flex items-center gap-1.5">
                            <User className="w-3.5 h-3.5 text-amber-400" /> 
                            1. Artisan Full Name *
                          </label>
                          {renderConfidenceBadge('name')}
                        </div>
                        <input
                          type="text"
                          required
                          value={quickFormData.name || ''}
                          onChange={(e) => handleQuickInputChange('name', e.target.value)}
                          placeholder="e.g. Gauri Devi"
                          className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-100 focus:border-amber-500 focus:outline-none"
                        />
                      </div>

                      {/* Field 2: Phone */}
                      <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 hover:border-slate-700 transition">
                        <div className="flex items-center justify-between mb-1.5">
                          <label className="text-xs text-slate-200 font-semibold flex items-center gap-1.5">
                            <Phone className="w-3.5 h-3.5 text-emerald-400" /> 
                            2. Mobile Contact Number *
                          </label>
                          {renderConfidenceBadge('phone')}
                        </div>
                        <input
                          type="tel"
                          required
                          value={quickFormData.phone || ''}
                          onChange={(e) => handleQuickInputChange('phone', e.target.value)}
                          placeholder="e.g. 9876543210"
                          className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-100 focus:border-amber-500 focus:outline-none"
                        />
                      </div>

                      {/* Field 3: Pehchan ID */}
                      <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 hover:border-slate-700 transition">
                        <div className="flex items-center justify-between mb-1.5">
                          <label className="text-xs text-slate-200 font-semibold flex items-center gap-1.5">
                            <IdCard className="w-3.5 h-3.5 text-orange-400" /> 
                            3. Government Pehchan Card ID
                          </label>
                          {renderConfidenceBadge('pehchan_id')}
                        </div>
                        <input
                          type="text"
                          value={quickFormData.pehchan_id || ''}
                          onChange={(e) => handleQuickInputChange('pehchan_id', e.target.value)}
                          placeholder="e.g. PEH-IND-88320"
                          className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-lg text-sm font-mono text-amber-300 focus:border-amber-500 focus:outline-none"
                        />
                      </div>
                    </div>

                    {/* Quick Form Actions */}
                    <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                      <div className="text-xs text-slate-400 flex items-center gap-1.5">
                        <ShieldCheck className="w-4 h-4 text-emerald-400" />
                        <span>Instant artisan identity card registration</span>
                      </div>

                      <button
                        type="submit"
                        disabled={isSubmitting || !quickFormData.name || !quickFormData.phone}
                        className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-amber-500 to-emerald-600 hover:from-amber-600 hover:to-emerald-700 text-white transition disabled:opacity-50 disabled:cursor-not-allowed shadow-md cursor-pointer"
                      >
                        {isSubmitting ? (
                          <>
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            Registering...
                          </>
                        ) : (
                          <>
                            <Send className="w-3.5 h-3.5" />
                            Complete Quick Onboarding
                          </>
                        )}
                      </button>
                    </div>
                  </form>
                ) : (
                  /* FULL DETAILED FORM */
                  <form onSubmit={handleFullSubmit} className="flex flex-col gap-4 flex-1 justify-between">
                    <div className="space-y-4">
                      
                      {/* Personal & Contact Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium flex items-center gap-1">
                              <User className="w-3 h-3 text-slate-400" /> Full Name *
                            </label>
                            {renderConfidenceBadge('name')}
                          </div>
                          <input
                            type="text"
                            required
                            value={formData.name || ''}
                            onChange={(e) => handleFullInputChange('name', e.target.value)}
                            placeholder="e.g. Gauri Devi"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium flex items-center gap-1">
                              <Phone className="w-3 h-3 text-slate-400" /> Phone Number *
                            </label>
                            {renderConfidenceBadge('phone')}
                          </div>
                          <input
                            type="text"
                            required
                            value={formData.phone || ''}
                            onChange={(e) => handleFullInputChange('phone', e.target.value)}
                            placeholder="e.g. 9876543210"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium flex items-center gap-1">
                              <Hash className="w-3 h-3 text-slate-400" /> 4-Digit PIN
                            </label>
                            {renderConfidenceBadge('pin')}
                          </div>
                          <input
                            type="password"
                            maxLength={4}
                            value={formData.pin || ''}
                            onChange={(e) => handleFullInputChange('pin', e.target.value)}
                            placeholder="e.g. 1234"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>
                      </div>

                      {/* Location & Cluster Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium flex items-center gap-1">
                              <MapPin className="w-3 h-3 text-slate-400" /> State
                            </label>
                            {renderConfidenceBadge('state')}
                          </div>
                          <input
                            type="text"
                            value={formData.state || ''}
                            onChange={(e) => handleFullInputChange('state', e.target.value)}
                            placeholder="e.g. Uttar Pradesh"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">District</label>
                            {renderConfidenceBadge('district')}
                          </div>
                          <input
                            type="text"
                            value={formData.district || ''}
                            onChange={(e) => handleFullInputChange('district', e.target.value)}
                            placeholder="e.g. Varanasi"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">Craft Cluster</label>
                            {renderConfidenceBadge('cluster_name')}
                          </div>
                          <input
                            type="text"
                            value={formData.cluster_name || ''}
                            onChange={(e) => handleFullInputChange('cluster_name', e.target.value)}
                            placeholder="e.g. Varanasi Handloom Cluster"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>
                      </div>

                      {/* Government Identifiers Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium flex items-center gap-1">
                              <Award className="w-3 h-3 text-amber-400" /> Pehchan Card ID
                            </label>
                            {renderConfidenceBadge('pehchan_id')}
                          </div>
                          <input
                            type="text"
                            value={formData.pehchan_id || ''}
                            onChange={(e) => handleFullInputChange('pehchan_id', e.target.value)}
                            placeholder="PEH-IND-88320"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-amber-300 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">TRIFED ID</label>
                            {renderConfidenceBadge('trifed_id')}
                          </div>
                          <input
                            type="text"
                            value={formData.trifed_id || ''}
                            onChange={(e) => handleFullInputChange('trifed_id', e.target.value)}
                            placeholder="TRIFED-UP-VNS-1049"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-amber-300 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">Dialect / Language</label>
                            {renderConfidenceBadge('primary_dialect')}
                          </div>
                          <select
                            value={formData.primary_dialect}
                            onChange={(e) => handleFullInputChange('primary_dialect', e.target.value)}
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          >
                            <option value="hindi">Hindi</option>
                            <option value="bhojpuri">Bhojpuri</option>
                            <option value="maithili">Maithili</option>
                            <option value="kannada">Kannada</option>
                            <option value="telugu">Telugu</option>
                            <option value="gujarati">Gujarati</option>
                            <option value="rajasthani">Rajasthani</option>
                            <option value="marathi">Marathi</option>
                            <option value="bengali">Bengali</option>
                            <option value="odia">Odia</option>
                          </select>
                        </div>
                      </div>

                      {/* Craft & GI Tag Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">Craft Category</label>
                            {renderConfidenceBadge('craft_category')}
                          </div>
                          <input
                            type="text"
                            value={formData.craft_category || ''}
                            onChange={(e) => handleFullInputChange('craft_category', e.target.value)}
                            placeholder="e.g. Handloom Textile"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">GI Certification Name</label>
                            {renderConfidenceBadge('gi_tag_name')}
                          </div>
                          <input
                            type="text"
                            value={formData.gi_tag_name || ''}
                            onChange={(e) => handleFullInputChange('gi_tag_name', e.target.value)}
                            placeholder="e.g. Banarasi Brocade & Saree (GI Reg #39)"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-slate-300 font-medium">Experience (Years)</label>
                            {renderConfidenceBadge('years_experience')}
                          </div>
                          <input
                            type="number"
                            min={0}
                            max={80}
                            value={formData.years_experience ?? ''}
                            onChange={(e) => handleFullInputChange('years_experience', e.target.value ? parseInt(e.target.value) : null)}
                            placeholder="e.g. 18"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
                          />
                        </div>
                      </div>

                      {/* Summary Narrative */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-xs text-slate-300 font-medium">Artisan Profile Summary</label>
                          {renderConfidenceBadge('skills_summary')}
                        </div>
                        <textarea
                          rows={2}
                          value={formData.skills_summary || ''}
                          onChange={(e) => handleFullInputChange('skills_summary', e.target.value)}
                          placeholder="Auto-generated artisan heritage background..."
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-100 focus:border-amber-500 focus:outline-none resize-none"
                        />
                      </div>

                    </div>

                    {/* Form Actions */}
                    <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                      <div className="text-xs text-slate-400 flex items-center gap-1.5">
                        <ShieldCheck className="w-4 h-4 text-emerald-400" />
                        <span>Government registry verification active</span>
                      </div>

                      <button
                        type="submit"
                        disabled={isSubmitting || !formData.name || !formData.phone}
                        className="inline-flex items-center gap-2 px-6 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-emerald-900/30 cursor-pointer"
                      >
                        {isSubmitting ? (
                          <>
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            Saving Profile...
                          </>
                        ) : (
                          <>
                            <Send className="w-3.5 h-3.5" />
                            Register Full Profile
                          </>
                        )}
                      </button>
                    </div>
                  </form>
                )}
              </>
            )}

            {/* JSON Tab */}
            {activeTab === 'json' && (
              <div className="flex-1 flex flex-col">
                <pre className="p-4 bg-slate-900 border border-slate-800 rounded-xl text-xs font-mono text-emerald-400 overflow-auto max-h-[420px] leading-relaxed">
                  {JSON.stringify(mode === 'quick' ? quickFormData : formData, null, 2)}
                </pre>
              </div>
            )}

            {/* Extracted Entities Tab */}
            {activeTab === 'entities' && (
              <div className="flex-1 overflow-auto max-h-[420px]">
                {extractionResult?.extracted_entities && extractionResult.extracted_entities.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {extractionResult.extracted_entities.map((ent, idx) => (
                      <div key={idx} className="p-3 bg-slate-900 border border-slate-800 rounded-xl flex items-center justify-between text-xs">
                        <div>
                          <span className="text-[10px] text-amber-400 uppercase font-mono block">{ent.type}</span>
                          <span className="font-semibold text-slate-100">{ent.value}</span>
                        </div>
                        <span className="text-[11px] font-mono text-emerald-400 font-bold">
                          {Math.round(ent.confidence * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-8 text-center text-xs text-slate-500">
                    No entities extracted yet. Click "Extract to Form" to populate.
                  </div>
                )}
              </div>
            )}

          </div>
        </section>

      </main>
    </div>
  );
}
export default App;
