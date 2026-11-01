import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import {
  Box,
  Cloud,
  Download,
  Eye,
  FileArchive,
  ImageUp,
  LoaderCircle,
  PenTool,
  Play,
  Plug,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Upload,
  X,
  Zap,
} from 'lucide-react'
import { Viewer3D, type Viewer3DHandle, type ViewMode } from './components/Viewer3D'
import { SketchCanvas, type ExampleName, type SketchCanvasHandle } from './components/SketchCanvas'
import { EditorialLanding } from './components/EditorialLanding'
import {
  downloadBlob,
  exportGLB,
  exportOBJ,
  loadGLB,
  validateObject,
  verifyGLB,
  type MeshReport,
} from './lib/mesh'
import {
  DEFAULT_PREPROCESS,
  canvasToBlob,
  preprocessImage,
  preprocessSketch,
  type PreprocessOptions,
  type PreprocessResult,
} from './lib/preprocess'
import {
  DEFAULT_GENERATE,
  checkHealth,
  createJob,
  fetchArtifact,
  getJob,
  type BackendHealth,
  type GenerateOptions,
  type JobState,
  type StageId,
} from './lib/api'

type Engine = 'triposr' | 'instantmesh'
type StageState = 'idle' | 'active' | 'done' | 'skipped' | 'error'
type StageView = { id: StageId; label: string; detail: string; state: StageState; seconds?: number }
type Health = { state: 'idle' | 'checking' | 'online' | 'offline'; info?: BackendHealth; error?: string }

const STAGE_ORDER: StageId[] = ['preprocess', 'concept', 'segment', 'shape', 'texture', 'export']

const STAGE_COPY: Record<Engine, Record<StageId, [string, string]>> = {
  triposr: {
    preprocess: ['Sketch conditioning', 'Contrast, denoise, crop 512²'],
    concept: ['Concept render', 'SD 1.5 + ControlNet (Local GPU)'],
    segment: ['Foreground isolation', 'rembg background removal'],
    shape: ['Shape generation', 'TripoSR feed-forward neural mesh'],
    texture: ['Color generation', 'Forward camera texture projection'],
    export: ['Validate & export', 'trimesh repair · GLB / OBJ'],
  },
  instantmesh: {
    preprocess: ['Sketch conditioning', 'Flat-field correction & line cleanup'],
    concept: ['Lineart concept render', 'SD 1.5 + Lineart ControlNet (Kaggle T4)'],
    segment: ['Foreground isolation', 'rembg centering on white matte'],
    shape: ['Multi-view diffusion', 'Zero123++ 6 novel angles (Kaggle T4)'],
    texture: ['LRM 3D reconstruction', 'NeRF transformer & UV texture baking'],
    export: ['Validate & export', 'Watertight trimesh repair · GLB / OBJ'],
  },
}

const PROMPT_CHIPS = ['glossy ceramic', 'brushed metal', 'oak wood', 'matte plastic, red', 'low-poly toy', 'gold']

function initialStages(engine: Engine): StageView[] {
  return STAGE_ORDER.map((id) => ({
    id,
    label: STAGE_COPY[engine][id][0],
    detail: STAGE_COPY[engine][id][1],
    state: 'idle' as StageState,
  }))
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n))

export default function App() {
  const sketchRef = useRef<SketchCanvasHandle>(null)
  const viewerRef = useRef<Viewer3DHandle>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const cancelRef = useRef(false)

  // Input
  const [inputMode, setInputMode] = useState<'draw' | 'upload'>('draw')
  const [upload, setUpload] = useState<{ img: HTMLImageElement; name: string; url: string } | null>(null)
  const [sketchVersion, setSketchVersion] = useState(0)
  const [prep, setPrep] = useState<PreprocessOptions>(DEFAULT_PREPROCESS)
  const [processed, setProcessed] = useState<{ result: PreprocessResult; url: string } | null>(null)
  const [showPrep, setShowPrep] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [dragOver, setDragOver] = useState(false)

  // Engine selection
  const [engine, setEngine] = useState<Engine>('triposr')

  // Local TripoSR Engine
  const [triposrUrl, setTriposrUrl] = useState(() => localStorage.getItem('s2s.triposr') ?? 'http://127.0.0.1:8001')
  const [triposrHealth, setTriposrHealth] = useState<Health>({ state: 'idle' })
  const [showConnect, setShowConnect] = useState(false)

  // Kaggle InstantMesh Engine
  const [kaggleUrl, setKaggleUrl] = useState(() => localStorage.getItem('s2s.kaggle') ?? '')
  const [kaggleToken, setKaggleToken] = useState(() => localStorage.getItem('s2s.kaggleToken') ?? '')
  const [kaggleHealth, setKaggleHealth] = useState<Health>({ state: 'idle' })
  const [showKaggleConnect, setShowKaggleConnect] = useState(false)

  const [genOpts, setGenOpts] = useState<GenerateOptions>(DEFAULT_GENERATE)

  // Run State
  const [stages, setStages] = useState<StageView[]>(() => initialStages('triposr'))
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [object, setObject] = useState<THREE.Object3D | null>(null)
  const [report, setReport] = useState<MeshReport | null>(null)
  const [glbCheck, setGlbCheck] = useState<boolean | null>(null)
  const [remote, setRemote] = useState<{ glb: Blob; objZip?: string; jobId: string; engine: Engine } | null>(null)
  const [gallery, setGallery] = useState<{ label: string; url: string }[]>([])
  const [runSummary, setRunSummary] = useState<{ engine: Engine; seconds: number } | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('textured')
  const [autoRotate, setAutoRotate] = useState(true)

  const triposrOnline = triposrHealth.state === 'online'
  const kaggleOnline = kaggleHealth.state === 'online'

  // View state: 'landing' (editorial showcase with Fragmented Apollo) vs 'studio' (3D generator)
  const [view, setView] = useState<'landing' | 'studio'>(() => {
    if (typeof window !== 'undefined' && window.location.hash === '#studio') {
      return 'studio'
    }
    return 'landing'
  })

  useEffect(() => {
    const handleHash = () => {
      if (window.location.hash === '#studio') {
        setView('studio')
      } else {
        setView('landing')
      }
    }
    window.addEventListener('hashchange', handleHash)
    return () => window.removeEventListener('hashchange', handleHash)
  }, [])

  // ---------- Preprocessing (live preview) ----------
  const currentSource = useCallback((): HTMLCanvasElement | HTMLImageElement | null => {
    if (inputMode === 'upload') return upload?.img ?? null
    const c = sketchRef.current?.getCanvas()
    if (!c || sketchRef.current?.isBlank()) return null
    return c
  }, [inputMode, upload])

  const runPreprocess = useCallback(() => {
    const src = currentSource()
    if (!src) return null
    const result = inputMode === 'upload' ? preprocessImage(src, prep) : preprocessSketch(src, prep)
    return { result, url: result.canvas.toDataURL('image/png') }
  }, [currentSource, inputMode, prep])

  useEffect(() => {
    const t = setTimeout(() => setProcessed(runPreprocess()), 140)
    return () => clearTimeout(t)
  }, [runPreprocess, sketchVersion])

  const onSketchChange = useCallback(() => setSketchVersion((v) => v + 1), [])

  const loadFile = (file: File | undefined) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please choose an image file (PNG, JPG, WEBP).')
      return
    }
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      if (upload) URL.revokeObjectURL(upload.url)
      setUpload({ img, name: file.name, url })
      setInputMode('upload')
      setError(null)
    }
    img.onerror = () => setError('That image could not be decoded.')
    img.src = url
  }

  // ---------- Backend connections ----------
  const connectTriposr = useCallback(async (url: string) => {
    if (!url.trim()) return
    localStorage.setItem('s2s.triposr', url.trim())
    setTriposrHealth({ state: 'checking' })
    try {
      const ctrl = new AbortController()
      const t = setTimeout(() => ctrl.abort(), 6000)
      const info = await checkHealth(url, ctrl.signal)
      clearTimeout(t)
      setTriposrHealth({ state: 'online', info })
    } catch (e) {
      setTriposrHealth({
        state: 'offline',
        error: e instanceof Error && e.name !== 'AbortError' ? e.message : 'No response from local server',
      })
    }
  }, [])

  const connectKaggle = useCallback(async (url: string, token?: string) => {
    const trimmed = url.trim().replace(/\/+$/, '')
    if (!trimmed) {
      setKaggleHealth({ state: 'offline', error: 'Please enter a Kaggle URL' })
      return
    }
    setKaggleHealth({ state: 'checking' })
    try {
      const h = await checkHealth(trimmed, undefined, token)
      setKaggleHealth({ state: 'online', info: h })
      localStorage.setItem('s2s.kaggle', trimmed)
      if (token) localStorage.setItem('s2s.kaggleToken', token)
    } catch (e) {
      setKaggleHealth({ state: 'offline', error: e instanceof Error ? e.message : 'Cannot reach Kaggle instance' })
    }
  }, [])

  useEffect(() => {
    if (triposrUrl) void connectTriposr(triposrUrl)
    if (kaggleUrl) void connectKaggle(kaggleUrl, kaggleToken)
  }, [connectTriposr, connectKaggle, triposrUrl, kaggleUrl, kaggleToken])

  // ---------- Generation ----------
  const patchStage = (id: StageId, patch: Partial<StageView>) =>
    setStages((s) => s.map((st) => (st.id === id ? { ...st, ...patch } : st)))

  const finishWithObject = async (obj: THREE.Object3D) => {
    setObject(obj)
    setReport(validateObject(obj))
  }

  const applyJob = (job: JobState, preSeconds: number, targetEngine: Engine) => {
    if (targetEngine === 'instantmesh') {
      const stageText = (job.stage || '').toLowerCase()
      setStages((s) =>
        s.map((st) => {
          if (job.status === 'done') {
            return { ...st, state: 'done' }
          }
          if (job.status === 'error') {
            return st.state === 'active' ? { ...st, state: 'error' } : st
          }
          if (stageText.includes('cleaning sketch') || stageText.includes('rendering')) {
            if (st.id === 'preprocess') return { ...st, state: 'done', seconds: preSeconds }
            if (st.id === 'concept') return { ...st, state: 'active' }
            return { ...st, state: 'idle' }
          }
          if (stageText.includes('multi-view') || stageText.includes('reconstructing')) {
            if (st.id === 'preprocess' || st.id === 'concept') return { ...st, state: 'done' }
            if (st.id === 'segment' || st.id === 'shape') return { ...st, state: 'active' }
            return { ...st, state: 'idle' }
          }
          if (stageText.includes('exporting') || stageText.includes('validating')) {
            if (st.id === 'export') return { ...st, state: 'active' }
            return { ...st, state: 'done' }
          }
          if (stageText.includes('waiting')) {
            if (st.id === 'preprocess') return { ...st, state: 'done', seconds: preSeconds }
            return { ...st, state: 'idle' }
          }
          return st
        }),
      )
    } else {
      // TripoSR mapping
      const current = job.stage && STAGE_ORDER.includes(job.stage as StageId) ? STAGE_ORDER.indexOf(job.stage as StageId) : -1
      setStages((s) =>
        s.map((st, i) => {
          const skipped = (job.stages_skipped ?? []).includes(st.id)
          const t = (job.timings ?? {})[st.id]
          const seconds = st.id === 'preprocess' ? preSeconds + (t ?? 0) : t
          let state: StageState = 'idle'
          if (skipped) state = 'skipped'
          else if (job.status === 'done') state = 'done'
          else if (job.status === 'error') state = i < current ? 'done' : i === current ? 'error' : 'idle'
          else if (i < current) state = 'done'
          else if (i === current) state = 'active'
          return { ...st, state, seconds: state === 'idle' ? undefined : seconds }
        }),
      )
    }
  }

  const runRemote = async (pre: PreprocessResult, preSeconds: number, targetUrl: string, targetEngine: Engine) => {
    const t0 = performance.now()
    const blob = await canvasToBlob(pre.canvas)
    const token = targetEngine === 'instantmesh' ? kaggleToken : undefined
    const { job_id } = await createJob(targetUrl, blob, prompt, { ...genOpts, engine: targetEngine }, token, targetEngine)

    let job: JobState
    for (;;) {
      if (cancelRef.current) throw new Error('Stopped watching the job.')
      job = await getJob(targetUrl, job_id, token)
      applyJob(job, preSeconds, targetEngine)
      if (job.status === 'done' || job.status === 'error') break
      await sleep(1200)
    }
    if (job.status === 'error') throw new Error(job.error ?? 'Generation failed on backend')

    // Fetch intermediate artifacts for gallery
    const images: { label: string; url: string }[] = []
    if (targetEngine === 'instantmesh') {
      try {
        const lineartBlob = await fetchArtifact(targetUrl, `/jobs/${job_id}/lineart`, token)
        images.push({ label: 'Clean Lineart', url: URL.createObjectURL(lineartBlob) })
      } catch {
        /* optional */
      }
      try {
        const conceptBlob = await fetchArtifact(targetUrl, `/jobs/${job_id}/concept`, token)
        images.push({ label: 'Concept Render', url: URL.createObjectURL(conceptBlob) })
      } catch {
        /* optional */
      }
    } else {
      const arts = job.artifacts && !Array.isArray(job.artifacts) ? (job.artifacts as Record<string, string>) : {}
      for (const [key, label] of [
        ['control', 'Control map'],
        ['concept', 'Concept render'],
        ['segmented', 'Isolated'],
        ['preview_cutout', 'Cutout & Silhouette'],
      ] as const) {
        const path = arts[key]
        if (path) images.push({ label, url: URL.createObjectURL(await fetchArtifact(targetUrl, path)) })
      }
    }
    setGallery((g) => [...g, ...images])

    // Fetch GLB
    let glbBlob: Blob
    let objZipPath: string | undefined

    if (targetEngine === 'instantmesh') {
      glbBlob = await fetchArtifact(targetUrl, `/jobs/${job_id}/glb`, token)
      objZipPath = `/jobs/${job_id}/obj`
    } else {
      const arts = job.artifacts && !Array.isArray(job.artifacts) ? (job.artifacts as Record<string, string>) : {}
      const glbPath = arts.glb
      if (!glbPath) throw new Error('TripoSR finished without a GLB file')
      glbBlob = await fetchArtifact(targetUrl, glbPath)
      objZipPath = arts.obj_zip
    }

    const scene = await loadGLB(await glbBlob.arrayBuffer())
    setGlbCheck(validateObject(scene).triangles > 0)
    setRemote({ glb: glbBlob, objZip: objZipPath, jobId: job_id, engine: targetEngine })
    await finishWithObject(scene)
    setRunSummary({
      engine: targetEngine,
      seconds: job.seconds ?? job.total_seconds ?? (performance.now() - t0) / 1000,
    })
  }

  const generate = async () => {
    setError(null)
    const t = performance.now()
    const fresh = runPreprocess()
    if (!fresh || fresh.result.empty) {
      setError(inputMode === 'draw' ? 'Draw something on the canvas first.' : 'Upload a sketch image first.')
      return
    }
    setProcessed(fresh)
    const preSeconds = (performance.now() - t) / 1000

    if (engine === 'instantmesh') {
      if (!kaggleOnline) {
        setShowKaggleConnect(true)
        setError('Please connect your Kaggle InstantMesh GPU tunnel first, or enter its URL.')
        return
      }
    } else {
      if (!triposrOnline) {
        setShowConnect(true)
        setError('Local TripoSR engine is offline. Start run_triposr.bat or check port 8001.')
        return
      }
    }

    const mode: Engine = engine
    const activeUrl = mode === 'instantmesh' ? kaggleUrl : triposrUrl

    cancelRef.current = false
    setRunning(true)
    setStages(initialStages(mode).map((s) => (s.id === 'preprocess' ? { ...s, state: 'active' } : s)))
    gallery.forEach((g) => URL.revokeObjectURL(g.url))
    setGallery([{ label: 'Input', url: fresh.url }])
    setRemote(null)
    setGlbCheck(null)
    try {
      await runRemote(fresh.result, preSeconds, activeUrl, mode)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStages((s) => s.map((st) => (st.state === 'active' ? { ...st, state: 'error' } : st)))
    } finally {
      setRunning(false)
    }
  }

  // ---------- Export ----------
  const baseName = useMemo(() => {
    const slug = prompt.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 32).replace(/^-|-$/g, '')
    return `morph-${slug || 'model'}`
  }, [prompt])

  const downloadGLB = async () => {
    if (!object) return
    downloadBlob(remote?.glb ?? (await exportGLB(object)), `${baseName}.glb`)
  }
  const downloadOBJ = () => object && downloadBlob(exportOBJ(object), `${baseName}.obj`)
  const downloadObjZip = async () => {
    if (!remote?.objZip) return
    const activeUrl = remote.engine === 'instantmesh' ? kaggleUrl : triposrUrl
    const token = remote.engine === 'instantmesh' ? kaggleToken : undefined
    downloadBlob(await fetchArtifact(activeUrl, remote.objZip, token), `${baseName}-obj.zip`)
  }

  if (view === 'landing') {
    return (
      <>
        <EditorialLanding
          onEnterStudio={() => {
            window.location.hash = '#studio'
            setView('studio')
          }}
        />

        {/* TripoSR Connect Dialog */}
        {showConnect && (
          <ConnectDialog
            triposrUrl={triposrUrl}
            setTriposrUrl={setTriposrUrl}
            triposrHealth={triposrHealth}
            onConnectTriposr={() => connectTriposr(triposrUrl)}
            onClose={() => setShowConnect(false)}
          />
        )}

        {/* Kaggle InstantMesh Connect Dialog */}
        {showKaggleConnect && (
          <KaggleDialog
            kaggleUrl={kaggleUrl}
            setKaggleUrl={setKaggleUrl}
            kaggleToken={kaggleToken}
            setKaggleToken={setKaggleToken}
            kaggleHealth={kaggleHealth}
            onConnectKaggle={() => connectKaggle(kaggleUrl, kaggleToken)}
            onClose={() => setShowKaggleConnect(false)}
          />
        )}
      </>
    )
  }

  return (
    <div className="min-h-screen bg-paper text-ink">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-ink/10 bg-paper/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1600px] items-center gap-4 px-5 lg:px-8">
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                window.location.hash = '#landing'
                setView('landing')
              }}
              className="group flex items-center gap-2.5 rounded-xl border border-ink/10 bg-white/60 px-2.5 py-1.5 transition hover:border-ink/25 hover:bg-white text-left"
              title="Return to Editorial Landing"
            >
              <div className="relative grid h-8 w-8 place-items-center rounded-lg bg-ink text-paper">
                <Box size={16} strokeWidth={2.2} />
                <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full border border-paper bg-accent" />
              </div>
              <div className="leading-none">
                <div className="font-display text-[15px] font-semibold tracking-tight text-ink flex items-baseline gap-0.5">
                  Morph <span className="italic font-normal text-accent text-[17px]" style={{ fontFamily: "'Pinyon Script', 'Newsreader', cursive" }}>ts</span>
                </div>
                <div className="mt-0.5 font-mono text-[9.5px] uppercase tracking-wider text-ink/45">
                  ← Return Home
                </div>
              </div>
            </button>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => {
                window.location.hash = '#landing'
                setView('landing')
              }}
              className="hidden sm:flex h-9 items-center gap-1.5 rounded-full border border-ink/12 bg-white/60 px-3.5 text-[11.5px] font-mono tracking-wider uppercase text-ink/70 transition hover:border-ink/25 hover:text-ink"
            >
              <span>← Editorial</span>
            </button>
            {/* TripoSR local button */}
            <button
              onClick={() => setShowConnect(true)}
              className="flex h-9 items-center gap-2 rounded-full border border-ink/12 bg-white/60 pl-3 pr-3.5 text-[13px] font-medium transition hover:border-ink/25"
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  triposrHealth.state === 'online'
                    ? 'bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.18)]'
                    : triposrHealth.state === 'checking'
                      ? 'animate-pulse bg-amber-400'
                      : triposrHealth.state === 'offline'
                        ? 'bg-red-500'
                        : 'bg-ink/25'
                }`}
              />
              <Zap size={14} className={triposrHealth.state === 'online' ? 'text-amber-600' : 'text-ink/40'} />
              {triposrHealth.state === 'online'
                ? 'TripoSR · RTX GPU'
                : triposrHealth.state === 'checking'
                  ? 'TripoSR checking…'
                  : 'TripoSR Local'}
            </button>

            {/* InstantMesh Kaggle button */}
            <button
              onClick={() => setShowKaggleConnect(true)}
              className="flex h-9 items-center gap-2 rounded-full border border-ink/12 bg-white/60 pl-3 pr-3.5 text-[13px] font-medium transition hover:border-ink/25"
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  kaggleOnline
                    ? 'bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.18)]'
                    : kaggleHealth.state === 'checking'
                      ? 'animate-pulse bg-amber-400'
                      : kaggleHealth.state === 'offline'
                        ? 'bg-red-500'
                        : 'bg-ink/25'
                }`}
              />
              <Cloud size={14} className={kaggleOnline ? 'text-blue-500' : 'text-ink/40'} />
              {kaggleOnline ? 'InstantMesh · T4' : 'Kaggle GPU'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1600px] gap-6 px-5 py-6 lg:grid-cols-[minmax(0,460px)_minmax(0,1fr)] lg:px-8">
        {/* ---------------- Left: input ---------------- */}
        <section className="flex flex-col gap-5">
          <div className="rounded-2xl border border-ink/10 bg-paper-2/60 p-4">
            <div className="mb-4 flex items-center justify-between">
              <StepLabel n={1} title="Sketch" />
              <div className="flex rounded-lg border border-ink/10 bg-white/60 p-0.5">
                <button
                  onClick={() => setInputMode('draw')}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] font-medium transition ${
                    inputMode === 'draw' ? 'bg-ink text-paper shadow-sm' : 'text-ink/60 hover:text-ink'
                  }`}
                >
                  <PenTool size={13} /> Draw
                </button>
                <button
                  onClick={() => setInputMode('upload')}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] font-medium transition ${
                    inputMode === 'upload' ? 'bg-ink text-paper shadow-sm' : 'text-ink/60 hover:text-ink'
                  }`}
                >
                  <Upload size={13} /> Upload
                </button>
              </div>
            </div>

            {inputMode === 'draw' ? (
              <div>
                <SketchCanvas ref={sketchRef} onChange={onSketchChange} />
                <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px] text-ink/55">
                  <span>Try an example:</span>
                  {(['mug', 'vase', 'lamp'] as ExampleName[]).map((n) => (
                    <button
                      key={n}
                      onClick={() => sketchRef.current?.drawExample(n)}
                      className="rounded-full border border-ink/10 bg-white/60 px-2.5 py-0.5 font-mono text-[11px] capitalize text-ink/75 transition hover:border-ink/25 hover:text-ink"
                    >
                      {n}
                    </button>
                  ))}
                  <button
                    onClick={() => sketchRef.current?.clear()}
                    className="ml-auto flex items-center gap-1 text-[11px] text-ink/40 hover:text-ink"
                  >
                    <RotateCcw size={11} /> Clear
                  </button>
                </div>
              </div>
            ) : (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDragOver(false)
                  loadFile(e.dataTransfer.files?.[0])
                }}
                onClick={() => fileRef.current?.click()}
                className={`grid min-h-[340px] cursor-pointer place-items-center rounded-xl border-2 border-dashed transition ${
                  dragOver
                    ? 'border-accent bg-accent/5'
                    : upload
                      ? 'border-ink/20 bg-white/80'
                      : 'border-ink/15 bg-white/40 hover:border-ink/30 hover:bg-white/60'
                }`}
              >
                {upload ? (
                  <div className="flex flex-col items-center gap-3 p-4 text-center">
                    <img src={upload.url} alt={upload.name} className="max-h-[220px] rounded-lg border border-ink/10 object-contain shadow-sm" />
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[12px] text-ink/70">{upload.name}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setUpload(null)
                          fileRef.current && (fileRef.current.value = '')
                        }}
                        className="rounded-md border border-ink/10 bg-white/80 px-2 py-0.5 text-[11px] hover:bg-white"
                      >
                        Replace
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 px-8 text-center">
                    <div className="grid h-12 w-12 place-items-center rounded-full bg-ink text-paper">
                      <ImageUp size={20} />
                    </div>
                    <div className="font-display text-[15px] font-medium">Drop a sketch image</div>
                    <div className="text-[12px] text-ink/50">
                      PNG, JPG or WEBP. Paper photos, scans and dark-background sketches all work.
                    </div>
                  </div>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/bmp"
                  className="hidden"
                  onChange={(e) => loadFile(e.target.files?.[0])}
                />
              </div>
            )}

            {/* Preprocessing */}
            <div className="mt-4 rounded-xl border border-ink/10 bg-white/60">
              <button
                onClick={() => setShowPrep((v) => !v)}
                className="flex w-full items-center gap-3 p-2.5 text-left"
              >
                <div className="h-14 w-14 shrink-0 overflow-hidden rounded-md border border-ink/10 bg-white">
                  {processed && !processed.result.empty ? (
                    <img src={processed.url} alt="Preprocessed sketch" className="h-full w-full object-contain" />
                  ) : (
                    <div className="paper-dots h-full w-full" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-medium">Preprocessed input</div>
                  <div className="font-mono text-[11px] text-ink/50">
                    {processed && !processed.result.empty
                      ? `512×512 · ink ${(processed.result.inkRatio * 100).toFixed(1)}%${
                          processed.result.inverted ? ' · auto-inverted' : ''
                        }`
                      : 'Waiting for a sketch'}
                  </div>
                </div>
                <SlidersHorizontal size={16} className={showPrep ? 'text-accent' : 'text-ink/40'} />
              </button>
              {showPrep && (
                <div className="grid gap-3 border-t border-ink/10 p-3.5">
                  <Slider label="Contrast" value={prep.contrast} min={0.5} max={2.5} step={0.05}
                    onChange={(v) => setPrep({ ...prep, contrast: v })} display={`${prep.contrast.toFixed(2)}×`} />
                  <Slider label="Ink threshold" value={prep.threshold} min={60} max={240} step={1}
                    onChange={(v) => setPrep({ ...prep, threshold: v })} display={String(prep.threshold)} />
                  <Slider label="Stroke boost" value={prep.strokeBoost} min={0} max={3} step={1}
                    onChange={(v) => setPrep({ ...prep, strokeBoost: v })} display={`${prep.strokeBoost}px`} />
                  <div className="flex gap-4 pt-1">
                    <Toggle label="Denoise" checked={prep.denoise} onChange={(v) => setPrep({ ...prep, denoise: v })} />
                    <Toggle label="Auto-crop & centre" checked={prep.autoCrop} onChange={(v) => setPrep({ ...prep, autoCrop: v })} />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Prompt */}
          <div className="rounded-2xl border border-ink/10 bg-paper-2/60 p-4">
            <StepLabel n={2} title="Describe it" hint="Optional" />
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="e.g. a coffee mug in glossy blue ceramic with a speckled glaze"
              className="mt-3 w-full resize-none rounded-xl border border-ink/10 bg-white/80 px-3.5 py-3 text-[14px] leading-relaxed outline-none transition placeholder:text-ink/35 focus:border-ink/30"
            />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {PROMPT_CHIPS.map((c) => (
                <button
                  key={c}
                  onClick={() => setPrompt((p) => (p.trim() ? `${p.trim().replace(/,$/, '')}, ${c}` : c))}
                  className="rounded-full border border-ink/10 bg-white/60 px-2.5 py-1 text-[12px] text-ink/65 transition hover:border-ink/25 hover:text-ink"
                >
                  + {c}
                </button>
              ))}
            </div>
          </div>

          {/* Step 3: Engine Selector (TripoSR vs InstantMesh) */}
          <div className="rounded-2xl border border-ink/10 bg-paper-2/60 p-4">
            <StepLabel n={3} title="Engine" />
            <div className="mt-3 grid grid-cols-2 gap-2">
              <button
                onClick={() => {
                  setEngine('triposr')
                  setStages(initialStages('triposr'))
                  if (!triposrOnline) setShowConnect(true)
                }}
                className={`relative flex flex-col items-start gap-1.5 rounded-xl border p-3 text-left transition ${
                  engine === 'triposr'
                    ? 'border-ink bg-white shadow-[0_0_0_1px_var(--color-ink)]'
                    : 'border-ink/10 bg-white/50 hover:border-ink/25'
                }`}
              >
                <div className="flex w-full items-center justify-between">
                  <span className={`grid h-7 w-7 place-items-center rounded-md ${engine === 'triposr' ? 'bg-ink text-paper' : 'bg-ink/5 text-ink/60'}`}>
                    <Zap size={15} />
                  </span>
                  <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-emerald-800">
                    RTX GPU
                  </span>
                </div>
                <div>
                  <div className="text-[13.5px] font-semibold text-ink">TripoSR</div>
                  <div className="text-[11px] text-ink/50">{triposrOnline ? 'Local Port 8001' : 'Offline'}</div>
                </div>
              </button>

              <button
                onClick={() => {
                  setEngine('instantmesh')
                  setStages(initialStages('instantmesh'))
                  if (!kaggleOnline) setShowKaggleConnect(true)
                }}
                className={`relative flex flex-col items-start gap-1.5 rounded-xl border p-3 text-left transition ${
                  engine === 'instantmesh'
                    ? 'border-ink bg-white shadow-[0_0_0_1px_var(--color-ink)]'
                    : 'border-ink/10 bg-white/50 hover:border-ink/25'
                }`}
              >
                <div className="flex w-full items-center justify-between">
                  <span className={`grid h-7 w-7 place-items-center rounded-md ${engine === 'instantmesh' ? 'bg-ink text-paper' : 'bg-ink/5 text-ink/60'}`}>
                    <Cloud size={15} />
                  </span>
                  <span className="rounded-full bg-blue-500/15 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-blue-800">
                    Cloud T4
                  </span>
                </div>
                <div>
                  <div className="text-[13.5px] font-semibold text-ink">InstantMesh</div>
                  <div className="text-[11px] text-ink/50">{kaggleOnline ? 'Kaggle GPU & CPU' : 'Kaggle Cloud'}</div>
                </div>
              </button>
            </div>

            {engine === 'triposr' ? (
              <div className="mt-4 grid gap-3">
                <div className="flex flex-wrap gap-x-5 gap-y-2.5">
                  <Toggle
                    label="2D Concept pass (SD 1.5 + ControlNet)"
                    checked={genOpts.use_controlnet}
                    onChange={(v) => setGenOpts({ ...genOpts, use_controlnet: v })}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Select
                    label="Marching Cubes Res"
                    value={genOpts.octree_resolution}
                    options={[128, 192, 256]}
                    onChange={(v) => setGenOpts({ ...genOpts, octree_resolution: v })}
                  />
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] text-ink/50">Seed</span>
                    <input
                      type="number"
                      value={genOpts.seed}
                      onChange={(e) => setGenOpts({ ...genOpts, seed: Math.max(0, Math.floor(+e.target.value || 0)) })}
                      className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2.5 font-mono text-[13px] outline-none focus:border-ink/30"
                    />
                  </label>
                </div>
                <div className="rounded-lg bg-ink/5 p-2.5 text-[11.5px] text-ink/70">
                  ⚡ <strong>Stability AI TripoSR</strong> produces high-density textured 3D assets in ~2 to 4 seconds using local NVIDIA CUDA acceleration.
                </div>
              </div>
            ) : (
              <div className="mt-4 grid gap-3">
                <div className="grid grid-cols-2 gap-2">
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] text-ink/50">Input Classification</span>
                    <select
                      value={genOpts.input_type || 'auto'}
                      onChange={(e) => setGenOpts({ ...genOpts, input_type: e.target.value })}
                      className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2 font-mono text-[13px] outline-none focus:border-ink/30"
                    >
                      <option value="auto">Auto-detect</option>
                      <option value="sketch">Pencil Sketch</option>
                      <option value="photo">Render / Photo</option>
                    </select>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] text-ink/50">Zero123++ Steps</span>
                    <input
                      type="number"
                      value={genOpts.steps ?? 40}
                      min={20}
                      max={50}
                      onChange={(e) => setGenOpts({ ...genOpts, steps: Math.max(20, Math.min(50, Math.floor(+e.target.value || 40))) })}
                      className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2.5 font-mono text-[13px] outline-none focus:border-ink/30"
                    />
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Slider
                    label="Sketch Fidelity"
                    value={genOpts.fidelity ?? 1.0}
                    min={0.4}
                    max={1.6}
                    step={0.1}
                    onChange={(v) => setGenOpts({ ...genOpts, fidelity: v })}
                    display={`${(genOpts.fidelity ?? 1.0).toFixed(1)}×`}
                  />
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] text-ink/50">Seed</span>
                    <input
                      type="number"
                      value={genOpts.seed}
                      onChange={(e) => setGenOpts({ ...genOpts, seed: Math.max(0, Math.floor(+e.target.value || 0)) })}
                      className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2.5 font-mono text-[13px] outline-none focus:border-ink/30"
                    />
                  </label>
                </div>
                <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-2.5 text-[11.5px] text-blue-950">
                  ☁️ <strong>InstantMesh (TencentARC)</strong> runs on Kaggle GPU (16 GB T4 VRAM + 29 GB CPU RAM). Synthesizes 6 multi-view angles with Zero123++ and reconstructs a 360° NeRF mesh.
                </div>
                {!kaggleOnline && (
                  <button
                    onClick={() => setShowKaggleConnect(true)}
                    className="flex items-center justify-center gap-1.5 rounded-lg border border-accent/30 bg-accent/5 py-2 text-[12px] font-medium text-accent hover:bg-accent/10 transition"
                  >
                    <Cloud size={14} /> Connect Kaggle Cloudflare Tunnel
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Generate Button */}
          <div className="flex flex-col gap-2">
            {running ? (
              <button
                onClick={() => (cancelRef.current = true)}
                className="flex h-13 items-center justify-center gap-2 rounded-xl bg-ink text-[15px] font-semibold text-paper"
              >
                <LoaderCircle size={18} className="animate-spin" />
                Generating with {engine === 'instantmesh' ? 'InstantMesh (Kaggle)' : 'TripoSR'}…
                <span className="ml-2 flex items-center gap-1 rounded-md bg-white/10 px-2 py-0.5 text-[12px] font-medium">
                  <Square size={10} fill="currentColor" /> Stop
                </span>
              </button>
            ) : (
              <button
                onClick={generate}
                disabled={engine === 'instantmesh' ? !kaggleOnline : !triposrOnline}
                className="group flex h-13 items-center justify-center gap-2 rounded-xl bg-accent text-[15px] font-semibold text-white shadow-[0_8px_24px_-8px_rgba(255,79,31,0.6)] transition hover:brightness-105 disabled:cursor-not-allowed disabled:bg-ink/20 disabled:shadow-none"
              >
                <Play size={17} fill="currentColor" className="transition group-hover:translate-x-0.5" />
                {engine === 'instantmesh'
                  ? kaggleOnline
                    ? 'Generate 3D Asset (InstantMesh · T4)'
                    : 'Connect Kaggle Tunnel First'
                  : triposrOnline
                    ? 'Generate 3D Asset (TripoSR · RTX)'
                    : 'Start Local TripoSR First (Port 8001)'}
              </button>
            )}
            {error && (
              <div className="rounded-xl border border-red-500/20 bg-red-50 p-3 text-[13px] text-red-800">
                {error}
              </div>
            )}
          </div>
        </section>

        {/* ---------------- Right: 3D view + results ---------------- */}
        <section className="flex flex-col gap-4">
          <div className="relative min-h-[560px] flex-1 overflow-hidden rounded-2xl border border-white/10 bg-night text-paper shadow-2xl">
            {/* Viewport controls */}
            <div className="absolute left-4 top-4 z-10 flex flex-wrap items-center gap-2">
              <div className="flex rounded-lg border border-white/15 bg-black/40 p-0.5 backdrop-blur">
                {(['textured', 'clay', 'wireframe', 'normals'] as ViewMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setViewMode(m)}
                    className={`rounded-md px-2.5 py-1 font-mono text-[11px] capitalize transition ${
                      viewMode === m ? 'bg-white text-ink shadow-sm' : 'text-white/70 hover:text-white'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            <div className="absolute right-4 top-4 z-10 flex items-center gap-1.5">
              <button
                onClick={() => setAutoRotate((r) => !r)}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11.5px] backdrop-blur transition ${
                  autoRotate ? 'border-white/30 bg-white/15 text-white' : 'border-white/10 bg-black/40 text-white/60 hover:text-white'
                }`}
              >
                Auto-rotate
              </button>
              <button
                onClick={() => viewerRef.current?.resetCamera()}
                className="rounded-lg border border-white/10 bg-black/40 p-1.5 text-white/70 backdrop-blur hover:text-white"
                title="Reset view"
              >
                <RotateCcw size={14} />
              </button>
            </div>

            {/* 3D viewport */}
            <Viewer3D
              ref={viewerRef}
              object={object}
              mode={viewMode}
              autoRotate={autoRotate}
            />

            {!object && !running && (
              <div className="pointer-events-none absolute inset-0 grid place-items-center p-6 text-center">
                <div className="max-w-sm">
                  <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl border border-white/10 bg-white/5 text-white/40">
                    <Eye size={22} />
                  </div>
                  <div className="font-display text-[17px] font-semibold text-white">Your 3D asset appears here</div>
                  <p className="mt-1.5 text-[13px] text-white/45">
                    Draw or upload a sketch on the left, select your engine, then generate.
                  </p>
                </div>
              </div>
            )}

            {/* Bottom summary badge */}
            {runSummary && (
              <div className="absolute bottom-4 left-4 z-10 flex items-center gap-2 rounded-lg border border-white/15 bg-black/55 px-3 py-1.5 font-mono text-[11px] backdrop-blur">
                <span className="font-semibold text-accent">
                  {runSummary.engine === 'instantmesh' ? 'InstantMesh · Kaggle T4' : 'TripoSR · Local RTX'}
                </span>
                <span className="text-white/40">·</span>
                <span className="text-white/80">{runSummary.seconds.toFixed(1)}s</span>
              </div>
            )}
          </div>

          {/* Stage progression cards */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            {stages.map((s, idx) => (
              <div
                key={s.id}
                className={`flex flex-col justify-between rounded-xl border p-3 transition ${
                  s.state === 'active'
                    ? 'border-accent bg-accent/10 shadow-[0_0_0_1px_var(--color-accent)]'
                    : s.state === 'done'
                      ? 'border-emerald-600/30 bg-emerald-500/5'
                      : s.state === 'error'
                        ? 'border-red-500/40 bg-red-500/10'
                        : 'border-ink/8 bg-paper-2/40'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between font-mono text-[10px] text-ink/45">
                    <span>{String(idx + 1).padStart(2, '0')}</span>
                    {s.state === 'active' && <LoaderCircle size={11} className="animate-spin text-accent" />}
                    {s.state === 'done' && <span className="text-emerald-600">✓</span>}
                  </div>
                  <div className="mt-1 text-[12px] font-semibold leading-tight">{s.label}</div>
                  <div className="mt-1 text-[10.5px] leading-snug text-ink/55">{s.detail}</div>
                </div>
                <div className="mt-2 font-mono text-[10px] text-ink/40">
                  {s.seconds != null ? `${s.seconds.toFixed(1)}s` : s.state === 'active' ? 'working…' : '—'}
                </div>
              </div>
            ))}
          </div>

          {/* Validation & Export Cards */}
          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            {/* Validation */}
            <div className="rounded-xl border border-white/8 bg-night-2 p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="flex items-center gap-2 text-[13px] font-medium text-paper">
                  <ShieldCheck size={15} className="text-white/50" /> Mesh validation
                </span>
                {report && (
                  <span className={`rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${
                    report.valid ? 'bg-emerald-400/15 text-emerald-300' : 'bg-red-400/15 text-red-300'
                  }`}>
                    {report.valid ? 'usable' : 'invalid'}
                  </span>
                )}
              </div>
              {report ? (
                <>
                  <dl className="grid grid-cols-3 gap-x-3 gap-y-3">
                    <Stat k="Vertices" v={fmt(report.vertices)} />
                    <Stat k="Triangles" v={fmt(report.triangles)} />
                    <Stat k="Meshes" v={String(report.meshes)} />
                    <Stat k="Watertight" v={report.watertight ? 'yes' : 'no'} good={report.watertight} />
                    <Stat k="Texture" v={report.hasTexture ? 'map' : report.hasVertexColors ? 'vertex' : 'none'} good={report.hasTexture || report.hasVertexColors} />
                    <Stat k="GLB reload" v={glbCheck == null ? '—' : glbCheck ? 'pass' : 'fail'} good={!!glbCheck} />
                  </dl>
                  <div className="mt-3 font-mono text-[11px] text-white/45">
                    bbox {report.size.map((s) => s.toFixed(2)).join(' × ')}
                  </div>
                  {report.issues.length > 0 && (
                    <ul className="mt-2 space-y-0.5 text-[11px] text-amber-300/80">
                      {report.issues.map((i) => <li key={i}>· {i}</li>)}
                    </ul>
                  )}
                </>
              ) : (
                <p className="text-[12px] text-white/40">Geometry, topology and export checks appear after generation.</p>
              )}
            </div>

            {/* Export */}
            <div className="flex flex-col rounded-xl border border-white/8 bg-night-2 p-4">
              <span className="mb-3 flex items-center gap-2 text-[13px] font-medium text-paper">
                <Download size={15} className="text-white/50" /> Export
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={downloadGLB} disabled={!object}
                  className="flex h-11 items-center justify-center gap-2 rounded-lg bg-white text-[13px] font-semibold text-ink transition hover:bg-white/90 disabled:opacity-25">
                  .glb <span className="font-normal text-ink/50">textured</span>
                </button>
                <button onClick={downloadOBJ} disabled={!object}
                  className="flex h-11 items-center justify-center gap-2 rounded-lg border border-white/15 text-[13px] font-semibold text-paper transition hover:bg-white/5 disabled:opacity-25">
                  .obj <span className="font-normal text-white/45">geometry</span>
                </button>
              </div>
              {remote?.objZip && (
                <button onClick={downloadObjZip}
                  className="mt-2 flex h-10 items-center justify-center gap-2 rounded-lg border border-white/10 text-[12px] text-white/75 transition hover:bg-white/5">
                  <FileArchive size={14} /> OBJ + MTL + texture (.zip bundle)
                </button>
              )}
              {gallery.length > 0 && (
                <div className="mt-auto flex gap-2 overflow-x-auto pt-3">
                  {gallery.map((g) => (
                    <figure key={g.url} className="shrink-0">
                      <img src={g.url} alt={g.label} className="h-16 w-16 rounded-md border border-white/10 bg-white object-contain" />
                      <figcaption className="mt-1 text-center font-mono text-[9px] uppercase tracking-wider text-white/40">{g.label}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      {/* TripoSR Connect Dialog */}
      {showConnect && (
        <ConnectDialog
          triposrUrl={triposrUrl}
          setTriposrUrl={setTriposrUrl}
          triposrHealth={triposrHealth}
          onConnectTriposr={() => connectTriposr(triposrUrl)}
          onClose={() => setShowConnect(false)}
        />
      )}

      {/* Kaggle InstantMesh Connect Dialog */}
      {showKaggleConnect && (
        <KaggleDialog
          kaggleUrl={kaggleUrl}
          setKaggleUrl={setKaggleUrl}
          kaggleToken={kaggleToken}
          setKaggleToken={setKaggleToken}
          kaggleHealth={kaggleHealth}
          onConnectKaggle={() => connectKaggle(kaggleUrl, kaggleToken)}
          onClose={() => setShowKaggleConnect(false)}
        />
      )}
    </div>
  )
}

// ---------------- Small UI components ----------------

function StepLabel({ n, title, hint }: { n: number; title: string; hint?: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid h-6 w-6 place-items-center rounded-full border border-ink/15 font-mono text-[11px] text-ink/60">{n}</span>
      <span className="font-display text-[15px] font-semibold">{title}</span>
      {hint && <span className="font-mono text-[11px] uppercase tracking-wider text-ink/40">· {hint}</span>}
    </div>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-2 text-[12.5px] text-ink/75 hover:text-ink">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5 rounded border-ink/20 text-accent focus:ring-accent/30"
      />
      {label}
    </label>
  )
}

function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string
}) {
  return (
    <label className="flex flex-col gap-1 text-[12px]">
      <div className="flex justify-between text-ink/60">
        <span>{label}</span>
        <span className="font-mono text-[11px]">{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(+e.target.value)}
        className="accent-accent"
      />
    </label>
  )
}

function Select({ label, value, options, onChange, format = String }: {
  label: string; value: number; options: number[]; onChange: (v: number) => void; format?: (n: number) => string
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] text-ink/50">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2 font-mono text-[13px] outline-none focus:border-ink/30"
      >
        {options.map((o) => <option key={o} value={o}>{format(o)}</option>)}
      </select>
    </label>
  )
}

function Stat({ k, v, good }: { k: string; v: string; good?: boolean }) {
  return (
    <div>
      <dt className="text-[10.5px] text-white/40">{k}</dt>
      <dd className={`font-mono text-[14px] ${good === undefined ? 'text-white' : good ? 'text-emerald-300' : 'text-white/60'}`}>{v}</dd>
    </div>
  )
}

function ConnectDialog(props: {
  triposrUrl: string
  setTriposrUrl: (s: string) => void
  triposrHealth: Health
  onConnectTriposr: () => void
  onClose: () => void
}) {
  const { triposrHealth } = props

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/40 p-4 backdrop-blur-sm" onClick={props.onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-ink/10 bg-paper p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <div>
            <div className="font-display text-[19px] font-semibold tracking-tight">Local TripoSR Engine</div>
            <p className="mt-1 text-[13px] text-ink/55">Configure your high-speed local GPU backend.</p>
          </div>
          <button onClick={props.onClose} className="text-ink/40 hover:text-ink"><X size={18} /></button>
        </div>

        <div className="mt-4">
          <div className="rounded-xl border border-ink/10 bg-white/60 p-3.5 text-[12.5px] text-ink/75">
            <div className="font-medium text-ink">⚡ Local Machine GPU Pipeline</div>
            <p className="mt-1 text-ink/60">
              Runs directly on your machine using <strong>Stability AI TripoSR</strong> with forward camera texture projection for photorealistic 3D models.
            </p>
            <div className="mt-2 font-mono text-[11px] text-ink/50">
              Server script: <code>backend/triposr_server.py</code> (Port 8001)
            </div>
          </div>

          <form className="mt-4 flex gap-2" onSubmit={(e) => { e.preventDefault(); props.onConnectTriposr() }}>
            <input
              value={props.triposrUrl}
              onChange={(e) => props.setTriposrUrl(e.target.value)}
              placeholder="http://127.0.0.1:8001"
              className="h-11 flex-1 rounded-xl border border-ink/15 bg-white px-3.5 font-mono text-[13px] outline-none focus:border-ink/40"
            />
            <button
              type="submit"
              disabled={!props.triposrUrl.trim() || triposrHealth.state === 'checking'}
              className="flex h-11 items-center gap-2 rounded-xl bg-ink px-4 text-[13px] font-semibold text-paper disabled:opacity-40"
            >
              {triposrHealth.state === 'checking' ? <LoaderCircle size={15} className="animate-spin" /> : <Plug size={15} />} Test
            </button>
          </form>

          {triposrHealth.state === 'online' && triposrHealth.info && (
            <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-xl border border-emerald-600/20 bg-emerald-50 p-3.5 font-mono text-[11.5px] text-emerald-900">
              <span>Local GPU</span><span className="text-right">{typeof triposrHealth.info.gpu === 'object' ? triposrHealth.info.gpu?.name : triposrHealth.info.gpu} · {triposrHealth.info.vram_gb?.toFixed(0)} GB</span>
              <span>Shape Engine</span><span className="truncate text-right">{triposrHealth.info.shape_model}</span>
              <span>2D Diffusion</span><span className="text-right">SD 1.5 + ControlNet Scribble (FP16)</span>
              <span>Texture</span><span className="text-right">Forward Camera Projection</span>
            </div>
          )}
          {triposrHealth.state === 'offline' && (
            <div className="mt-4 rounded-xl border border-red-500/25 bg-red-50 p-3.5 text-[12.5px] text-red-800">
              Could not connect to TripoSR server: {triposrHealth.error}. Ensure <code>run_triposr.bat</code> is running on port 8001.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function KaggleDialog(props: {
  kaggleUrl: string
  setKaggleUrl: (s: string) => void
  kaggleToken: string
  setKaggleToken: (s: string) => void
  kaggleHealth: Health
  onConnectKaggle: () => void
  onClose: () => void
}) {
  const { kaggleHealth } = props

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/40 p-4 backdrop-blur-sm" onClick={props.onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-ink/10 bg-paper p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <div>
            <div className="font-display text-[19px] font-semibold tracking-tight">Kaggle InstantMesh Engine</div>
            <p className="mt-1 text-[13px] text-ink/55">Connect to your Kaggle Tesla T4 GPU & CPU backend.</p>
          </div>
          <button onClick={props.onClose} className="text-ink/40 hover:text-ink"><X size={18} /></button>
        </div>

        <div className="mt-4 space-y-3.5">
          <div className="rounded-xl border border-ink/10 bg-white/60 p-3.5 text-[12.5px] text-ink/75">
            <div className="flex items-center justify-between font-medium text-ink">
              <span className="flex items-center gap-1.5"><Cloud size={15} className="text-blue-500" /> Cloud T4 GPU Acceleration</span>
              <a
                href="/sketch2shape_api--good.ipynb"
                download="sketch2shape_api--good.ipynb"
                className="inline-flex items-center gap-1 text-[11.5px] font-semibold text-accent hover:underline"
              >
                <Download size={13} /> Download Notebook
              </a>
            </div>
            <p className="mt-1.5 text-ink/60">
              Runs <strong>sketch2shape_api--good.ipynb</strong> on Kaggle with 16 GB Tesla T4 VRAM and 29 GB CPU RAM. Generates Zero123++ 6-view novel angles and reconstructs 360° NeRF meshes with UV textures.
            </p>
            <div className="mt-2.5 space-y-1 font-mono text-[11px] text-ink/65">
              <div>1. Open <code>sketch2shape_api--good.ipynb</code> in Kaggle (GPU T4 x2, Internet: On).</div>
              <div>2. Run all cells through Cell 13 (starts server & Cloudflare tunnel).</div>
              <div>3. Copy the <code>PUBLIC API URL</code> and <code>API TOKEN</code> printed in Cell 13.</div>
            </div>
          </div>

          <form className="space-y-2.5" onSubmit={(e) => { e.preventDefault(); props.onConnectKaggle() }}>
            <div>
              <label className="mb-1 block text-[11px] font-medium text-ink/60">Public Cloudflare Tunnel URL</label>
              <input
                value={props.kaggleUrl}
                onChange={(e) => props.setKaggleUrl(e.target.value)}
                placeholder="https://xxxx-xxxx.trycloudflare.com"
                className="h-10 w-full rounded-xl border border-ink/15 bg-white px-3 font-mono text-[12.5px] outline-none focus:border-ink/40"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] font-medium text-ink/60">API Token (From notebook Cell 13)</label>
              <input
                value={props.kaggleToken}
                onChange={(e) => props.setKaggleToken(e.target.value)}
                placeholder="Paste API_TOKEN from Kaggle notebook"
                className="h-10 w-full rounded-xl border border-ink/15 bg-white px-3 font-mono text-[12.5px] outline-none focus:border-ink/40"
              />
            </div>
            <div className="flex justify-end pt-1">
              <button
                type="submit"
                disabled={!props.kaggleUrl.trim() || kaggleHealth.state === 'checking'}
                className="flex h-10 items-center gap-2 rounded-xl bg-ink px-4 text-[13px] font-semibold text-paper disabled:opacity-40"
              >
                {kaggleHealth.state === 'checking' ? <LoaderCircle size={15} className="animate-spin" /> : <Plug size={15} />} Connect Kaggle Engine
              </button>
            </div>
          </form>

          {kaggleHealth.state === 'online' && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-xl border border-emerald-600/20 bg-emerald-50 p-3.5 font-mono text-[11.5px] text-emerald-900">
              <span>Cloud GPU</span>
              <span className="text-right">
                {typeof kaggleHealth.info?.gpu === 'object'
                  ? `${kaggleHealth.info?.gpu?.name || 'Tesla T4'} · ${kaggleHealth.info?.gpu?.memory_total_mb ? Math.round(kaggleHealth.info.gpu.memory_total_mb / 1024) : 16} GB`
                  : 'Tesla T4 (16 GB)'}
              </span>
              <span>Host CPU & RAM</span>
              <span className="text-right">Intel Xeon · 29 GB RAM</span>
              <span>Free Disk</span>
              <span className="text-right">{kaggleHealth.info?.disk_free_gb ? `${kaggleHealth.info.disk_free_gb} GB Free` : '18+ GB Free'}</span>
              <span>Shape Model</span>
              <span className="truncate text-right">{kaggleHealth.info?.model || 'instant-mesh-base'}</span>
            </div>
          )}
          {kaggleHealth.state === 'offline' && (
            <div className="rounded-xl border border-red-500/25 bg-red-50 p-3.5 text-[12.5px] text-red-800">
              Could not connect to Kaggle server: {kaggleHealth.error}. Please check the URL and ensure the Kaggle notebook cell is running.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
