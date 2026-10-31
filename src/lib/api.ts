// Client for the Colab FastAPI backend (backend/server.py).

export type StageId = 'preprocess' | 'concept' | 'segment' | 'shape' | 'texture' | 'export'
export type JobStatus = 'queued' | 'running' | 'done' | 'error'

export interface BackendHealth {
  status: string
  gpu: string | null
  vram_gb: number | null
  low_vram: boolean
  ram_gb?: number | null
  sequential?: boolean
  shape_model: string
  texture_available: boolean
  controlnet_model: string
  loaded: Record<string, boolean>
  version: string
}

export interface JobState {
  id: string
  status: JobStatus
  stage: StageId | null
  message: string
  error: string | null
  timings: Partial<Record<StageId, number>>
  stages_skipped: StageId[]
  artifacts: Partial<Record<'control' | 'concept' | 'segmented' | 'glb' | 'obj_zip', string>>
  stats: {
    vertices: number
    faces: number
    watertight: boolean
    extents: number[]
    textured: boolean
  } | null
  total_seconds: number | null
}

export interface GenerateOptions {
  use_controlnet: boolean
  controlnet_scale: number
  texture: boolean
  turbo: boolean
  octree_resolution: number
  max_faces: number
  seed: number
}

export const DEFAULT_GENERATE: GenerateOptions = {
  use_controlnet: true,
  controlnet_scale: 0.9,
  texture: true,
  turbo: true,
  octree_resolution: 256,
  max_faces: 40000,
  seed: 1234,
}

// Header lets the API work through ngrok free tunnels as well as cloudflared.
const HEADERS = { 'ngrok-skip-browser-warning': 'true' }

export function normaliseBase(url: string) {
  return url.trim().replace(/\/+$/, '')
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { ...init, headers: { ...HEADERS, ...(init?.headers ?? {}) } })
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

export function checkHealth(base: string, signal?: AbortSignal) {
  return request<BackendHealth>(`${normaliseBase(base)}/health`, { signal })
}

export async function createJob(base: string, image: Blob, prompt: string, options: GenerateOptions) {
  const form = new FormData()
  form.append('image', image, 'sketch.png')
  form.append('prompt', prompt)
  form.append('options', JSON.stringify(options))
  return request<{ job_id: string }>(`${normaliseBase(base)}/jobs`, { method: 'POST', body: form })
}

export function getJob(base: string, id: string) {
  return request<JobState>(`${normaliseBase(base)}/jobs/${id}`)
}

export async function fetchArtifact(base: string, path: string): Promise<Blob> {
  const res = await fetch(`${normaliseBase(base)}${path}`, { headers: HEADERS })
  if (!res.ok) throw new Error(`Could not download ${path} (${res.status})`)
  return res.blob()
}
