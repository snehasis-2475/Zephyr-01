// Client for Local TripoSR & Remote Kaggle InstantMesh backends.

export type StageId = 'preprocess' | 'concept' | 'segment' | 'shape' | 'texture' | 'export'
export type JobStatus = 'queued' | 'running' | 'done' | 'error'

export interface BackendHealth {
  status?: string
  ok?: boolean
  engine?: string
  model?: string
  supported_engines?: string[]
  gpu?: string | null | { name?: string; memory_used_mb?: number; memory_total_mb?: number }
  vram_gb?: number | null
  low_vram?: boolean
  ram_gb?: number | null
  queued?: number
  disk_free_gb?: number
  shape_model?: string
  texture_available?: boolean
  controlnet_model?: string
  loaded?: Record<string, boolean>
  version?: string
}

export interface JobState {
  id: string
  status: JobStatus
  stage: string | null
  message?: string
  error?: string | null
  created?: number
  updated?: number
  elapsed?: number | null
  seconds?: number | null
  total_seconds?: number | null
  timings?: Partial<Record<StageId, number>>
  stages_skipped?: StageId[]
  artifacts?: Record<string, string> | string[]
  stats?: {
    vertices: number
    faces: number
    watertight: boolean
    extents?: number[]
    textured: boolean
    glb_mb?: number
  } | null
  profile?: {
    gpu_total_mb?: number
    ram_total_mb?: number
    samples?: [number, number, number][]
  }
}

export interface GenerateOptions {
  engine?: 'triposr' | 'instantmesh'
  use_controlnet: boolean
  controlnet_scale: number
  texture: boolean
  turbo: boolean
  octree_resolution: number
  max_faces: number
  seed: number
  steps?: number
  input_type?: string
  fidelity?: number
}

export const DEFAULT_GENERATE: GenerateOptions = {
  engine: 'triposr',
  use_controlnet: true,
  controlnet_scale: 0.72,
  texture: true,
  turbo: true,
  octree_resolution: 192,
  max_faces: 40000,
  seed: 1234,
  steps: 40,
  input_type: 'auto',
  fidelity: 1.0,
}

// Header lets the API work through ngrok free tunnels and cloudflared
export function buildHeaders(token?: string): Record<string, string> {
  const h: Record<string, string> = { 'ngrok-skip-browser-warning': 'true' }
  if (token?.trim()) {
    h['x-api-token'] = token.trim()
  }
  return h
}

export function normaliseBase(url: string) {
  return url.trim().replace(/\/+$/, '')
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export async function checkHealth(base: string, signal?: AbortSignal, token?: string): Promise<BackendHealth> {
  const t = token?.trim()
  const b = normaliseBase(base)
  const url = `${b}/health${t ? `?token=${encodeURIComponent(t)}` : ''}`
  return await request<BackendHealth>(url, { signal, headers: buildHeaders(token) })
}

export async function createJob(
  base: string,
  image: Blob,
  prompt: string,
  options: GenerateOptions,
  token?: string,
  targetEngine: 'triposr' | 'instantmesh' = 'triposr'
) {
  const form = new FormData()
  form.append('file', image, 'sketch.png')
  form.append('image', image, 'sketch.png')
  form.append('sketch', image, 'sketch.png')
  form.append('prompt', prompt)
  form.append('options', JSON.stringify(options))
  form.append('steps', String(options.steps || (targetEngine === 'instantmesh' ? 40 : 35)))
  form.append('texture', String(options.texture ?? true))
  form.append('remove_bg', 'false')
  form.append('enhance', 'true')
  form.append('input_type', options.input_type || 'auto')
  form.append('fidelity', String(options.fidelity || 1.0))
  if (options.seed != null) {
    form.append('seed', String(options.seed))
  }

  const t = token?.trim()
  const url = `${normaliseBase(base)}/jobs${t ? `?token=${encodeURIComponent(t)}` : ''}`
  const res = await request<{ job_id?: string; id?: string }>(url, {
    method: 'POST',
    body: form,
    headers: buildHeaders(token),
  })
  return { job_id: res.job_id || res.id || '' }
}

export function getJob(base: string, id: string, token?: string) {
  const t = token?.trim()
  const url = `${normaliseBase(base)}/jobs/${id}${t ? `?token=${encodeURIComponent(t)}` : ''}`
  return request<JobState>(url, { headers: buildHeaders(token) })
}

export async function fetchArtifact(base: string, path: string, token?: string): Promise<Blob> {
  const t = token?.trim()
  const b = normaliseBase(base)
  let fullUrl = ''
  if (path.startsWith('http://') || path.startsWith('https://')) {
    fullUrl = path
  } else {
    const sep = path.startsWith('/') ? '' : '/'
    fullUrl = `${b}${sep}${path}`
  }
  const glue = fullUrl.includes('?') ? '&' : '?'
  const urlWithToken = `${fullUrl}${t ? `${glue}token=${encodeURIComponent(t)}` : ''}`
  const res = await fetch(urlWithToken, { headers: buildHeaders(token) })
  if (!res.ok) throw new Error(`Could not download ${path} (${res.status})`)
  return res.blob()
}
