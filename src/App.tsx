import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type * as THREE from 'three'
import {
  Box,
  Cpu,
  Download,
  FileArchive,
  ImageUp,
  Laptop,
  LoaderCircle,
  PenLine,
  Play,
  Plug,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  TriangleAlert,
  X,
} from 'lucide-react'
import { SketchCanvas, type ExampleName, type SketchCanvasHandle } from './components/SketchCanvas'
import { Viewer3D, type ViewMode, type Viewer3DHandle } from './components/Viewer3D'
import { PipelineTracker, type StageState, type StageView } from './components/PipelineTracker'
import { DEFAULT_PREPROCESS, canvasToBlob, preprocessSketch, type PreprocessOptions, type PreprocessResult } from './lib/preprocess'
import { MATERIAL_PRESETS, buildProceduralMesh, resolveMaterial } from './lib/procedural'
import { downloadBlob, exportGLB, exportOBJ, loadGLB, validateObject, verifyGLB, type MeshReport } from './lib/mesh'
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

type Engine = 'hunyuan' | 'local'
type Health = { state: 'idle' | 'checking' | 'online' | 'offline'; info?: BackendHealth; error?: string }

const STAGE_ORDER: StageId[] = ['preprocess', 'concept', 'segment', 'shape', 'texture', 'export']

const STAGE_COPY: Record<Engine, Record<StageId, [string, string]>> = {
  hunyuan: {
    preprocess: ['Sketch conditioning', 'Contrast, denoise, crop 512²'],
    concept: ['Concept render', 'SD 1.5 + ControlNet Scribble'],
    segment: ['Foreground isolation', 'Background removal'],
    shape: ['Shape generation', 'Hunyuan3D-DiT flow model'],
    texture: ['Texture painting', 'Hunyuan3D-Paint'],
    export: ['Validate & export', 'trimesh checks · GLB / OBJ'],
  },
  local: {
    preprocess: ['Sketch conditioning', 'Contrast, denoise, crop 512²'],
    concept: ['Concept render', 'Needs the GPU backend'],
    segment: ['Silhouette extraction', 'Outline fill + distance field'],
    shape: ['Inflation mesh', 'Height field, closed shell'],
    texture: ['Surface material', 'Prompt colour + sketch lines'],
    export: ['Validate & export', 'Watertight check · GLB round-trip'],
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

  // Engine
  const [engine, setEngine] = useState<Engine>('local')
  const [backendUrl, setBackendUrl] = useState(() => localStorage.getItem('s2s.backend') ?? '')
  const [health, setHealth] = useState<Health>({ state: 'idle' })
  const [showConnect, setShowConnect] = useState(false)
  const [genOpts, setGenOpts] = useState<GenerateOptions>(DEFAULT_GENERATE)
  const [materialId, setMaterialId] = useState('clay')
  const [depth, setDepth] = useState(0.9)
  const [relief, setRelief] = useState(0.6)

  // Run
  const [stages, setStages] = useState<StageView[]>(() => initialStages('local'))
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [object, setObject] = useState<THREE.Object3D | null>(null)
  const [report, setReport] = useState<MeshReport | null>(null)
  const [glbCheck, setGlbCheck] = useState<boolean | null>(null)
  const [remote, setRemote] = useState<{ glb: Blob; objZip?: string; jobId: string } | null>(null)
  const [gallery, setGallery] = useState<{ label: string; url: string }[]>([])
  const [runSummary, setRunSummary] = useState<{ engine: Engine; seconds: number } | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('textured')
  const [autoRotate, setAutoRotate] = useState(true)

  const online = health.state === 'online'

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
    const result = preprocessSketch(src, prep)
    return { result, url: result.canvas.toDataURL('image/png') }
  }, [currentSource, prep])

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

  // ---------- Backend connection ----------
  const connect = useCallback(async (url: string) => {
    if (!url.trim()) return
    localStorage.setItem('s2s.backend', url.trim())
    setHealth({ state: 'checking' })
    try {
      const ctrl = new AbortController()
      const t = setTimeout(() => ctrl.abort(), 10000)
      const info = await checkHealth(url, ctrl.signal)
      clearTimeout(t)
      setHealth({ state: 'online', info })
      setEngine('hunyuan')
      setStages(initialStages('hunyuan'))
      setGenOpts((o) => ({ ...o, texture: o.texture && info.texture_available }))
    } catch (e) {
      setHealth({
        state: 'offline',
        error: e instanceof Error && e.name !== 'AbortError' ? e.message : 'No response within 10s',
      })
    }
  }, [])

  useEffect(() => {
    if (backendUrl) void connect(backendUrl)
    // Only auto-connect with the saved URL on first load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---------- Generation ----------
  const patchStage = (id: StageId, patch: Partial<StageView>) =>
    setStages((s) => s.map((st) => (st.id === id ? { ...st, ...patch } : st)))

  const finishWithObject = async (obj: THREE.Object3D) => {
    setObject(obj)
    setReport(validateObject(obj))
  }

  const runLocal = async (pre: PreprocessResult, preSeconds: number) => {
    const t0 = performance.now()
    patchStage('preprocess', { state: 'done', seconds: preSeconds })
    patchStage('concept', { state: 'skipped' })
    patchStage('segment', { state: 'active' })
    await sleep(30)
    const material = resolveMaterial(prompt, materialId)
    const tShape = performance.now()
    patchStage('segment', { state: 'done' })
    patchStage('shape', { state: 'active' })
    await sleep(30)
    const group = buildProceduralMesh(pre, { resolution: 128, depth, relief, material })
    const shapeSec = (performance.now() - tShape) / 1000
    patchStage('shape', { state: 'done', seconds: shapeSec })
    patchStage('texture', { state: 'done' })
    patchStage('export', { state: 'active' })
    const tExp = performance.now()
    const glb = await exportGLB(group)
    const ok = await verifyGLB(glb)
    setGlbCheck(ok)
    patchStage('export', { state: ok ? 'done' : 'error', seconds: (performance.now() - tExp) / 1000 })
    await finishWithObject(group)
    setRunSummary({ engine: 'local', seconds: preSeconds + (performance.now() - t0) / 1000 })
  }

  const applyJob = (job: JobState, preSeconds: number) => {
    const current = job.stage ? STAGE_ORDER.indexOf(job.stage) : -1
    setStages((s) =>
      s.map((st, i) => {
        const skipped = job.stages_skipped.includes(st.id)
        const t = job.timings[st.id]
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

  const runRemote = async (pre: PreprocessResult, preSeconds: number) => {
    const t0 = performance.now()
    const blob = await canvasToBlob(pre.canvas)
    const { job_id } = await createJob(backendUrl, blob, prompt, genOpts)
    let job: JobState
    for (;;) {
      if (cancelRef.current) throw new Error('Stopped watching the job. It may still finish on Colab.')
      job = await getJob(backendUrl, job_id)
      applyJob(job, preSeconds)
      if (job.status === 'done' || job.status === 'error') break
      await sleep(1500)
    }
    if (job.status === 'error') throw new Error(job.error ?? 'Generation failed on the backend')

    const images: { label: string; url: string }[] = []
    for (const [key, label] of [
      ['control', 'Control map'],
      ['concept', 'Concept render'],
      ['segmented', 'Isolated'],
    ] as const) {
      const path = job.artifacts[key]
      if (path) images.push({ label, url: URL.createObjectURL(await fetchArtifact(backendUrl, path)) })
    }
    setGallery((g) => [...g, ...images])

    if (!job.artifacts.glb) throw new Error('Backend finished without a GLB file')
    const glb = await fetchArtifact(backendUrl, job.artifacts.glb)
    const scene = await loadGLB(await glb.arrayBuffer())
    setGlbCheck(validateObject(scene).triangles > 0)
    setRemote({ glb, objZip: job.artifacts.obj_zip, jobId: job_id })
    await finishWithObject(scene)
    setRunSummary({ engine: 'hunyuan', seconds: job.total_seconds ?? (performance.now() - t0) / 1000 })
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
    const mode = engine === 'hunyuan' && online ? 'hunyuan' : 'local'
    cancelRef.current = false
    setRunning(true)
    setStages(initialStages(mode).map((s) => (s.id === 'preprocess' ? { ...s, state: 'active' } : s)))
    gallery.forEach((g) => URL.revokeObjectURL(g.url))
    setGallery([{ label: 'Input', url: fresh.url }])
    setRemote(null)
    setGlbCheck(null)
    try {
      if (mode === 'hunyuan') await runRemote(fresh.result, preSeconds)
      else await runLocal(fresh.result, preSeconds)
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
    return `sketch2shape-${slug || 'model'}`
  }, [prompt])

  const downloadGLB = async () => {
    if (!object) return
    downloadBlob(remote?.glb ?? (await exportGLB(object)), `${baseName}.glb`)
  }
  const downloadOBJ = () => object && downloadBlob(exportOBJ(object), `${baseName}.obj`)
  const downloadObjZip = async () => {
    if (!remote?.objZip) return
    downloadBlob(await fetchArtifact(backendUrl, remote.objZip), `${baseName}-obj.zip`)
  }

  const effectiveEngine: Engine = engine === 'hunyuan' && online ? 'hunyuan' : 'local'
  const blockedRemote = engine === 'hunyuan' && !online

  return (
    <div className="min-h-screen bg-paper text-ink">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-ink/10 bg-paper/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1600px] items-center gap-4 px-5 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="relative grid h-9 w-9 place-items-center rounded-lg bg-ink text-paper">
              <Box size={18} strokeWidth={2.2} />
              <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full border-2 border-paper bg-accent" />
            </div>
            <div className="leading-none">
              <div className="font-display text-[17px] font-semibold tracking-tight">Sketch2Shape AI</div>
              <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.16em] text-ink/45">
                2D sketch → textured 3D asset
              </div>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setShowConnect(true)}
              className="flex h-9 items-center gap-2 rounded-full border border-ink/12 bg-white/60 pl-3 pr-3.5 text-[13px] font-medium transition hover:border-ink/25"
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  health.state === 'online'
                    ? 'bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.18)]'
                    : health.state === 'checking'
                      ? 'animate-pulse bg-amber-400'
                      : health.state === 'offline'
                        ? 'bg-red-500'
                        : 'bg-ink/25'
                }`}
              />
              {health.state === 'online'
                ? `Colab · ${health.info?.gpu ?? 'GPU'}`
                : health.state === 'checking'
                  ? 'Connecting…'
                  : 'Connect Colab GPU'}
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
              <div className="flex rounded-lg border border-ink/10 bg-white/70 p-0.5">
                {(
                  [
                    ['draw', 'Draw', PenLine],
                    ['upload', 'Upload', ImageUp],
                  ] as const
                ).map(([id, label, Icon]) => (
                  <button
                    key={id}
                    onClick={() => setInputMode(id)}
                    className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-[13px] font-medium transition ${
                      inputMode === id ? 'bg-ink text-paper' : 'text-ink/60 hover:text-ink'
                    }`}
                  >
                    <Icon size={14} /> {label}
                  </button>
                ))}
              </div>
            </div>

            <div className={inputMode === 'draw' ? '' : 'hidden'}>
              <SketchCanvas ref={sketchRef} onChange={onSketchChange} />
              <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px] text-ink/55">
                <span>Try an example:</span>
                {(['mug', 'vase', 'lamp'] as ExampleName[]).map((n) => (
                  <button
                    key={n}
                    onClick={() => sketchRef.current?.drawExample(n)}
                    className="rounded-full border border-ink/12 bg-white/70 px-2.5 py-1 capitalize text-ink/75 transition hover:border-ink/30 hover:text-ink"
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>

            {inputMode === 'upload' && (
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragOver(true)
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDragOver(false)
                  loadFile(e.dataTransfer.files[0])
                }}
                onClick={() => fileRef.current?.click()}
                className={`relative grid aspect-square w-full cursor-pointer place-items-center overflow-hidden rounded-xl border-2 border-dashed transition ${
                  dragOver ? 'border-accent bg-accent/5' : 'border-ink/15 bg-white/70 hover:border-ink/30'
                }`}
              >
                {upload ? (
                  <>
                    <img src={upload.url} alt="Uploaded sketch" className="h-full w-full object-contain p-4" />
                    <span className="absolute bottom-3 left-3 max-w-[70%] truncate rounded-md bg-ink/80 px-2 py-1 font-mono text-[11px] text-paper">
                      {upload.name}
                    </span>
                    <span className="absolute bottom-3 right-3 rounded-md bg-white px-2 py-1 text-[11px] font-medium shadow">
                      Replace
                    </span>
                  </>
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

          {/* Engine */}
          <div className="rounded-2xl border border-ink/10 bg-paper-2/60 p-4">
            <StepLabel n={3} title="Engine" />
            <div className="mt-3 grid grid-cols-2 gap-2">
              <EngineCard
                active={engine === 'hunyuan'}
                onClick={() => {
                  setEngine('hunyuan')
                  setStages(initialStages('hunyuan'))
                  if (!online) setShowConnect(true)
                }}
                icon={<Cpu size={16} />}
                title="Hunyuan3D-2"
                sub={online ? 'Colab GPU connected' : 'Requires Colab GPU'}
                badge={online ? 'live' : undefined}
              />
              <EngineCard
                active={engine === 'local'}
                onClick={() => {
                  setEngine('local')
                  setStages(initialStages('local'))
                }}
                icon={<Laptop size={16} />}
                title="Offline preview"
                sub="Runs in your browser"
              />
            </div>

            {engine === 'hunyuan' ? (
              <div className="mt-4 grid gap-3">
                <div className="flex flex-wrap gap-x-5 gap-y-2.5">
                  <Toggle label="ControlNet concept pass" checked={genOpts.use_controlnet}
                    onChange={(v) => setGenOpts({ ...genOpts, use_controlnet: v })} />
                  <Toggle label="Paint texture" checked={genOpts.texture}
                    disabled={online && !health.info?.texture_available}
                    onChange={(v) => setGenOpts({ ...genOpts, texture: v })} />
                  <Toggle label="Turbo (5 steps)" checked={genOpts.turbo}
                    onChange={(v) => setGenOpts({ ...genOpts, turbo: v })} />
                </div>
                {genOpts.use_controlnet && (
                  <Slider label="Sketch adherence" value={genOpts.controlnet_scale} min={0.4} max={1.5} step={0.05}
                    onChange={(v) => setGenOpts({ ...genOpts, controlnet_scale: v })} display={genOpts.controlnet_scale.toFixed(2)} />
                )}
                <div className="grid grid-cols-3 gap-2">
                  <Select label="Octree" value={genOpts.octree_resolution} options={[196, 256, 380]}
                    onChange={(v) => setGenOpts({ ...genOpts, octree_resolution: v })} />
                  <Select label="Max faces" value={genOpts.max_faces} options={[20000, 40000, 80000]}
                    onChange={(v) => setGenOpts({ ...genOpts, max_faces: v })} format={fmt} />
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] text-ink/50">Seed</span>
                    <input type="number" value={genOpts.seed}
                      onChange={(e) => setGenOpts({ ...genOpts, seed: Math.max(0, Math.floor(+e.target.value || 0)) })}
                      className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2.5 font-mono text-[13px] outline-none focus:border-ink/30" />
                  </label>
                </div>
              </div>
            ) : (
              <div className="mt-4 grid gap-3">
                <div>
                  <span className="text-[11px] text-ink/50">Material (prompt keywords override)</span>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {MATERIAL_PRESETS.map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setMaterialId(m.id)}
                        className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12px] transition ${
                          materialId === m.id ? 'border-ink bg-ink text-paper' : 'border-ink/10 bg-white/60 text-ink/70 hover:border-ink/25'
                        }`}
                      >
                        <span className="h-3 w-3 rounded-full border border-black/10" style={{ background: m.color }} />
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>
                <Slider label="Inflation depth" value={depth} min={0.3} max={1.6} step={0.05} onChange={setDepth} display={depth.toFixed(2)} />
                <Slider label="Stroke relief" value={relief} min={0} max={1.5} step={0.05} onChange={setRelief} display={relief.toFixed(2)} />
              </div>
            )}
          </div>

          {/* Generate */}
          <div className="flex flex-col gap-2">
            {running ? (
              <button
                onClick={() => (cancelRef.current = true)}
                disabled={effectiveEngine === 'local'}
                className="flex h-13 items-center justify-center gap-2 rounded-xl bg-ink text-[15px] font-semibold text-paper"
              >
                <LoaderCircle size={18} className="animate-spin" />
                Generating…
                {effectiveEngine === 'hunyuan' && (
                  <span className="ml-2 flex items-center gap-1 rounded-md bg-white/10 px-2 py-0.5 text-[12px] font-medium">
                    <Square size={10} fill="currentColor" /> Stop
                  </span>
                )}
              </button>
            ) : (
              <button
                onClick={generate}
                disabled={blockedRemote}
                className="group flex h-13 items-center justify-center gap-2 rounded-xl bg-accent text-[15px] font-semibold text-white shadow-[0_8px_24px_-8px_rgba(255,79,31,0.6)] transition hover:brightness-105 disabled:cursor-not-allowed disabled:bg-ink/20 disabled:shadow-none"
              >
                <Play size={17} fill="currentColor" className="transition group-hover:translate-x-0.5" />
                {blockedRemote ? 'Connect the Colab backend first' : 'Generate 3D model'}
              </button>
            )}
            {blockedRemote && (
              <button onClick={() => { setEngine('local'); setStages(initialStages('local')) }}
                className="text-[12px] text-ink/55 underline-offset-2 hover:underline">
                or run the offline preview engine instead
              </button>
            )}
            {error && (
              <div className="flex items-start gap-2 rounded-xl border border-red-500/25 bg-red-50 px-3.5 py-3 text-[13px] text-red-800">
                <TriangleAlert size={16} className="mt-0.5 shrink-0" />
                <span className="flex-1">{error}</span>
                <button onClick={() => setError(null)} className="text-red-800/60 hover:text-red-800"><X size={14} /></button>
              </div>
            )}
          </div>
        </section>

        {/* ---------------- Right: output ---------------- */}
        <section className="flex min-w-0 flex-col gap-4 rounded-2xl bg-night p-3 text-white lg:sticky lg:top-[88px] lg:self-start">
          <div className="relative h-[52vh] min-h-[380px] overflow-hidden rounded-xl bg-[radial-gradient(ellipse_at_50%_35%,#2a2724_0%,#151416_70%)] lg:h-[60vh]">
            <Viewer3D ref={viewerRef} object={object} mode={viewMode} autoRotate={autoRotate} />

            {!object && !running && (
              <div className="pointer-events-none absolute inset-0 grid place-items-center">
                <div className="flex max-w-xs flex-col items-center gap-3 text-center">
                  <div className="grid h-14 w-14 place-items-center rounded-2xl border border-white/10 bg-white/5">
                    <Sparkles size={22} className="text-accent" />
                  </div>
                  <div className="font-display text-[18px] font-medium">Your 3D asset appears here</div>
                  <div className="text-[13px] leading-relaxed text-white/45">
                    Sketch on the left, then press Generate. Drag to orbit, scroll to zoom.
                  </div>
                </div>
              </div>
            )}
            {running && !object && (
              <div className="pointer-events-none absolute inset-0 grid place-items-center">
                <LoaderCircle size={28} className="animate-spin text-white/40" />
              </div>
            )}

            {/* Viewer toolbar */}
            <div className="absolute left-3 top-3 flex rounded-lg border border-white/10 bg-black/40 p-0.5 backdrop-blur">
              {(['textured', 'clay', 'wireframe', 'normals'] as ViewMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setViewMode(m)}
                  className={`h-7 rounded-md px-2.5 text-[12px] font-medium capitalize transition ${
                    viewMode === m ? 'bg-white text-ink' : 'text-white/60 hover:text-white'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
            <div className="absolute right-3 top-3 flex gap-1.5">
              <button
                onClick={() => setAutoRotate((v) => !v)}
                className={`h-8 rounded-lg border border-white/10 px-2.5 text-[12px] font-medium backdrop-blur transition ${
                  autoRotate ? 'bg-white/15 text-white' : 'bg-black/40 text-white/55 hover:text-white'
                }`}
              >
                Auto-rotate
              </button>
              <button
                onClick={() => viewerRef.current?.resetCamera()}
                title="Reset camera"
                className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 bg-black/40 text-white/60 backdrop-blur transition hover:text-white"
              >
                <RotateCcw size={14} />
              </button>
            </div>
            {runSummary && object && (
              <div className="absolute bottom-3 left-3 rounded-lg border border-white/10 bg-black/45 px-3 py-2 font-mono text-[11px] text-white/70 backdrop-blur">
                {runSummary.engine === 'hunyuan' ? `Hunyuan3D · ${health.info?.gpu ?? 'GPU'}` : 'Offline engine · browser'} ·{' '}
                <span className="text-white">{runSummary.seconds.toFixed(1)}s</span>
              </div>
            )}
          </div>

          <PipelineTracker stages={stages} />

          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            {/* Validation */}
            <div className="rounded-xl border border-white/8 bg-night-2 p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="flex items-center gap-2 text-[13px] font-medium">
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
              <span className="mb-3 flex items-center gap-2 text-[13px] font-medium">
                <Download size={15} className="text-white/50" /> Export
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={downloadGLB} disabled={!object}
                  className="flex h-11 items-center justify-center gap-2 rounded-lg bg-white text-[13px] font-semibold text-ink transition hover:bg-white/90 disabled:opacity-25">
                  .glb <span className="font-normal text-ink/50">textured</span>
                </button>
                <button onClick={downloadOBJ} disabled={!object}
                  className="flex h-11 items-center justify-center gap-2 rounded-lg border border-white/15 text-[13px] font-semibold transition hover:bg-white/5 disabled:opacity-25">
                  .obj <span className="font-normal text-white/45">geometry</span>
                </button>
              </div>
              {remote?.objZip && (
                <button onClick={downloadObjZip}
                  className="mt-2 flex h-10 items-center justify-center gap-2 rounded-lg border border-white/10 text-[12px] text-white/75 transition hover:bg-white/5">
                  <FileArchive size={14} /> OBJ + MTL + texture (.zip from Colab)
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

      {showConnect && (
        <ConnectDialog
          url={backendUrl}
          setUrl={setBackendUrl}
          health={health}
          onConnect={() => connect(backendUrl)}
          onClose={() => setShowConnect(false)}
        />
      )}
    </div>
  )
}

// ---------------- Small UI pieces ----------------

function StepLabel({ n, title, hint }: { n: number; title: string; hint?: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid h-6 w-6 place-items-center rounded-full border border-ink/15 font-mono text-[11px] text-ink/60">{n}</span>
      <span className="font-display text-[15px] font-semibold tracking-tight">{title}</span>
      {hint && <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink/40">{hint}</span>}
    </div>
  )
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; display: string; onChange: (v: number) => void }) {
  return (
    <label className="grid gap-1.5">
      <span className="flex justify-between text-[12px] text-ink/60">
        {props.label}
        <span className="font-mono text-ink/80">{props.display}</span>
      </span>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value}
        onChange={(e) => props.onChange(+e.target.value)} className="w-full accent-[var(--color-accent)]" />
    </label>
  )
}

function Toggle({ label, checked, onChange, disabled }: { label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <label className={`flex items-center gap-2 text-[12.5px] ${disabled ? 'opacity-40' : 'cursor-pointer'}`}>
      <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full transition ${checked ? 'bg-ink' : 'bg-ink/20'}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? 'left-[18px]' : 'left-0.5'}`} />
      </button>
      <span className="text-ink/75">{label}</span>
    </label>
  )
}

function Select({ label, value, options, onChange, format = String }: { label: string; value: number; options: number[]; onChange: (v: number) => void; format?: (n: number) => string }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] text-ink/50">{label}</span>
      <select value={value} onChange={(e) => onChange(+e.target.value)}
        className="h-9 rounded-lg border border-ink/10 bg-white/80 px-2 font-mono text-[13px] outline-none focus:border-ink/30">
        {options.map((o) => <option key={o} value={o}>{format(o)}</option>)}
      </select>
    </label>
  )
}

function EngineCard(props: { active: boolean; onClick: () => void; icon: React.ReactNode; title: string; sub: string; badge?: string }) {
  return (
    <button onClick={props.onClick}
      className={`relative flex flex-col items-start gap-2 rounded-xl border p-3 text-left transition ${
        props.active ? 'border-ink bg-white shadow-[0_0_0_1px_var(--color-ink)]' : 'border-ink/10 bg-white/50 hover:border-ink/25'
      }`}>
      <span className={`grid h-7 w-7 place-items-center rounded-md ${props.active ? 'bg-ink text-paper' : 'bg-ink/5 text-ink/60'}`}>{props.icon}</span>
      <span>
        <span className="block text-[13.5px] font-semibold">{props.title}</span>
        <span className="block text-[11.5px] text-ink/50">{props.sub}</span>
      </span>
      {props.badge && (
        <span className="absolute right-2.5 top-2.5 rounded-full bg-emerald-500/12 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-emerald-700">{props.badge}</span>
      )}
    </button>
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

function ConnectDialog(props: { url: string; setUrl: (s: string) => void; health: Health; onConnect: () => void; onClose: () => void }) {
  const { health } = props
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/40 p-4 backdrop-blur-sm" onClick={props.onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-ink/10 bg-paper p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <div>
            <div className="font-display text-[19px] font-semibold tracking-tight">Connect the Colab GPU backend</div>
            <p className="mt-1 text-[13px] text-ink/55">Hunyuan3D-2 runs in Google Colab and the app talks to it over a public tunnel.</p>
          </div>
          <button onClick={props.onClose} className="text-ink/40 hover:text-ink"><X size={18} /></button>
        </div>

        <ol className="mt-5 space-y-3 text-[13px]">
          {[
            ['Open the notebook', <>Upload <code className="rounded bg-ink/5 px-1 font-mono text-[12px]">colab/Sketch2Shape_Hunyuan3D.ipynb</code> to Colab and pick a GPU runtime (T4, L4 or A100).</>],
            ['Run all cells', <>Installs Hunyuan3D-2, compiles the texture rasterizer and starts the FastAPI server. The first run takes about 15–20 min.</>],
            ['Paste the tunnel URL', <>Copy the <code className="rounded bg-ink/5 px-1 font-mono text-[12px]">https://….trycloudflare.com</code> link from the last cell into the field below.</>],
          ].map(([t, d], i) => (
            <li key={i} className="flex gap-3">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-ink font-mono text-[11px] text-paper">{i + 1}</span>
              <div><div className="font-medium">{t}</div><div className="text-ink/55">{d}</div></div>
            </li>
          ))}
        </ol>

        <form className="mt-5 flex gap-2" onSubmit={(e) => { e.preventDefault(); props.onConnect() }}>
          <input value={props.url} onChange={(e) => props.setUrl(e.target.value)} placeholder="https://your-tunnel.trycloudflare.com"
            className="h-11 flex-1 rounded-xl border border-ink/15 bg-white px-3.5 font-mono text-[13px] outline-none focus:border-ink/40" />
          <button type="submit" disabled={!props.url.trim() || health.state === 'checking'}
            className="flex h-11 items-center gap-2 rounded-xl bg-ink px-4 text-[13px] font-semibold text-paper disabled:opacity-40">
            {health.state === 'checking' ? <LoaderCircle size={15} className="animate-spin" /> : <Plug size={15} />} Test
          </button>
        </form>

        {health.state === 'online' && health.info && (
          <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-xl border border-emerald-600/20 bg-emerald-50 p-3.5 font-mono text-[11.5px] text-emerald-900">
            <span>GPU</span><span className="text-right">{health.info.gpu} · {health.info.vram_gb?.toFixed(0)} GB</span>
            <span>Shape model</span><span className="truncate text-right">{health.info.shape_model}</span>
            <span>Texture</span><span className="text-right">{health.info.texture_available ? 'Hunyuan3D-Paint' : 'unavailable'}</span>
            <span>Memory mode</span><span className="text-right">{health.info.sequential ? 'sequential (load per stage)' : health.info.low_vram ? 'CPU offload' : 'all resident'}</span>
            {health.info.ram_gb != null && (<><span>System RAM</span><span className="text-right">{health.info.ram_gb.toFixed(1)} GB</span></>)}
          </div>
        )}
        {health.state === 'offline' && (
          <div className="mt-4 rounded-xl border border-red-500/25 bg-red-50 p-3.5 text-[12.5px] text-red-800">
            Could not reach the backend: {health.error}. Check that the Colab cell is still running and that the URL starts with https://.
          </div>
        )}
      </div>
    </div>
  )
}
