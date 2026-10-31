// Offline "silhouette inflation" engine. Turns the preprocessed sketch into a
// closed, two-sided 3D mesh with real depth so the app works without a GPU
// backend:
//   1. close the outline and flood-fill from the border to get a silhouette
//   2. chamfer distance transform -> rounded height field (pebble profile)
//   3. emboss the sketch strokes as surface relief
//   4. stitch front, back and side walls into a watertight mesh with UVs
import * as THREE from 'three'
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { PreprocessResult } from './preprocess'

export interface MaterialPreset {
  id: string
  label: string
  color: string
  roughness: number
  metalness: number
}

export const MATERIAL_PRESETS: MaterialPreset[] = [
  { id: 'clay', label: 'Clay', color: '#d9c8b4', roughness: 0.85, metalness: 0 },
  { id: 'ceramic', label: 'Ceramic', color: '#f1ede4', roughness: 0.25, metalness: 0 },
  { id: 'wood', label: 'Wood', color: '#a2693d', roughness: 0.7, metalness: 0 },
  { id: 'metal', label: 'Brushed metal', color: '#b9bec4', roughness: 0.35, metalness: 0.9 },
  { id: 'gold', label: 'Gold', color: '#d4a53c', roughness: 0.3, metalness: 1 },
  { id: 'plastic', label: 'Plastic', color: '#ff5a36', roughness: 0.45, metalness: 0 },
]

const COLOR_WORDS: Record<string, string> = {
  red: '#d23a2a',
  orange: '#ff7a1a',
  yellow: '#f2c230',
  green: '#3f9c5a',
  blue: '#2f6fd6',
  purple: '#7a4bc4',
  pink: '#ef7fae',
  black: '#232326',
  white: '#f2f0ea',
  grey: '#8b8d92',
  gray: '#8b8d92',
  brown: '#7b4f2c',
  teal: '#2a9d8f',
}

/** Pick a material preset from prompt keywords, falling back to the user choice. */
export function resolveMaterial(prompt: string, fallbackId: string): MaterialPreset {
  const p = prompt.toLowerCase()
  const base =
    MATERIAL_PRESETS.find((m) => p.includes(m.id) || p.includes(m.label.toLowerCase())) ??
    (p.includes('steel') || p.includes('chrome') || p.includes('silver')
      ? MATERIAL_PRESETS[3]
      : p.includes('porcelain')
        ? MATERIAL_PRESETS[1]
        : MATERIAL_PRESETS.find((m) => m.id === fallbackId)!)
  const word = Object.keys(COLOR_WORDS).find((c) => new RegExp(`\\b${c}\\b`).test(p))
  return word ? { ...base, color: COLOR_WORDS[word] } : base
}

export interface ProceduralOptions {
  resolution: number // grid cells per side
  depth: number // inflation multiplier
  relief: number // stroke emboss depth
  material: MaterialPreset
}

function dilate(m: Uint8Array, n: number, iters: number): Uint8Array {
  let cur = m
  for (let k = 0; k < iters; k++) {
    const out = new Uint8Array(cur)
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const i = y * n + x
        if (cur[i]) continue
        if (
          (x > 0 && cur[i - 1]) ||
          (x < n - 1 && cur[i + 1]) ||
          (y > 0 && cur[i - n]) ||
          (y < n - 1 && cur[i + n])
        )
          out[i] = 1
      }
    cur = out
  }
  return cur
}

function buildTexture(pre: PreprocessResult, material: MaterialPreset) {
  const S = 1024
  const c = document.createElement('canvas')
  c.width = S
  c.height = S
  const ctx = c.getContext('2d')!
  ctx.fillStyle = material.color
  ctx.fillRect(0, 0, S, S)
  // Subtle vertical gradient for a less flat finish.
  const g = ctx.createLinearGradient(0, 0, 0, S)
  g.addColorStop(0, 'rgba(255,255,255,0.12)')
  g.addColorStop(1, 'rgba(0,0,0,0.12)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, S, S)
  // Sketch strokes as darkened detail lines.
  ctx.globalCompositeOperation = 'multiply'
  ctx.globalAlpha = 0.55
  ctx.drawImage(pre.canvas, 0, 0, S, S)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.flipY = false
  return tex
}

export function buildProceduralMesh(pre: PreprocessResult, opts: ProceduralOptions): THREE.Group {
  const N = opts.resolution
  const S = pre.size
  const block = S / N

  // Max-pool the ink mask down to the grid.
  let lines: Uint8Array = new Uint8Array(N * N)
  for (let gy = 0; gy < N; gy++)
    for (let gx = 0; gx < N; gx++) {
      let hit = 0
      const x0 = Math.floor(gx * block)
      const y0 = Math.floor(gy * block)
      const x1 = Math.floor((gx + 1) * block)
      const y1 = Math.floor((gy + 1) * block)
      for (let y = y0; y < y1 && !hit; y++)
        for (let x = x0; x < x1; x++)
          if (pre.mask[y * S + x]) {
            hit = 1
            break
          }
      lines[gy * N + gx] = hit
    }
  const rawLines = lines
  // Close small gaps in hand-drawn outlines.
  lines = dilate(lines, N, 1)

  // Flood-fill outside region from the border.
  const outside = new Uint8Array(N * N)
  const stack: number[] = []
  for (let i = 0; i < N; i++) stack.push(i, (N - 1) * N + i, i * N, i * N + N - 1)
  while (stack.length) {
    const i = stack.pop()!
    if (outside[i] || lines[i]) continue
    outside[i] = 1
    const x = i % N
    const y = (i / N) | 0
    if (x > 0) stack.push(i - 1)
    if (x < N - 1) stack.push(i + 1)
    if (y > 0) stack.push(i - N)
    if (y < N - 1) stack.push(i + N)
  }
  let sil: Uint8Array = new Uint8Array(N * N)
  let enclosed = 0
  for (let i = 0; i < N * N; i++) {
    sil[i] = outside[i] ? 0 : 1
    if (sil[i] && !lines[i]) enclosed++
  }
  // Open strokes (nothing enclosed): fall back to thick tube-like strokes.
  if (enclosed < N * N * 0.01) sil = dilate(rawLines, N, 3)
  // Keep a one-cell empty border so the boundary is always closed.
  for (let i = 0; i < N; i++) sil[i] = sil[(N - 1) * N + i] = sil[i * N] = sil[i * N + N - 1] = 0

  // Chamfer (3-4) distance transform inside the silhouette.
  const INF = 1e9
  const d = new Float32Array(N * N)
  for (let i = 0; i < N * N; i++) d[i] = sil[i] ? INF : 0
  for (let y = 1; y < N - 1; y++)
    for (let x = 1; x < N - 1; x++) {
      const i = y * N + x
      if (!d[i]) continue
      d[i] = Math.min(d[i], d[i - 1] + 3, d[i - N] + 3, d[i - N - 1] + 4, d[i - N + 1] + 4)
    }
  for (let y = N - 2; y > 0; y--)
    for (let x = N - 2; x > 0; x--) {
      const i = y * N + x
      if (!d[i]) continue
      d[i] = Math.min(d[i], d[i + 1] + 3, d[i + N] + 3, d[i + N + 1] + 4, d[i + N - 1] + 4)
    }
  let R = 0
  for (let i = 0; i < N * N; i++) {
    d[i] /= 3
    if (d[i] > R) R = d[i]
  }
  R = Math.max(R, 1)

  // Circular inflation profile + stroke relief, then smooth.
  const cell = 2 / N
  let hgt = new Float32Array(N * N)
  for (let i = 0; i < N * N; i++) {
    if (!sil[i]) continue
    const t = Math.min(1, d[i] / R)
    const profile = Math.sqrt(1 - (1 - t) * (1 - t))
    hgt[i] = Math.max(cell * 0.35, profile * R * cell * opts.depth)
  }
  for (let pass = 0; pass < 3; pass++) {
    const nh = new Float32Array(hgt)
    for (let y = 1; y < N - 1; y++)
      for (let x = 1; x < N - 1; x++) {
        const i = y * N + x
        if (!sil[i] || d[i] <= 1.01) continue
        nh[i] = (hgt[i] * 4 + hgt[i - 1] + hgt[i + 1] + hgt[i - N] + hgt[i + N]) / 8
      }
    hgt = nh
  }
  const front = new Float32Array(hgt)
  for (let i = 0; i < N * N; i++)
    if (sil[i] && rawLines[i] && d[i] > 1.5) front[i] = Math.max(cell * 0.3, front[i] - opts.relief * cell)

  // Vertices: front (z = +h) and back (z = -h) per silhouette cell.
  const fIdx = new Int32Array(N * N).fill(-1)
  const bIdx = new Int32Array(N * N).fill(-1)
  const pos: number[] = []
  const uv: number[] = []
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const i = y * N + x
      if (!sil[i]) continue
      const wx = (x + 0.5 - N / 2) * cell
      const wy = (N / 2 - y - 0.5) * cell
      const u = (x + 0.5) / N
      const v = (y + 0.5) / N
      fIdx[i] = pos.length / 3
      pos.push(wx, wy, front[i])
      uv.push(u, v)
      bIdx[i] = pos.length / 3
      pos.push(wx, wy, -hgt[i])
      uv.push(u, v)
    }

  // Quads exist where all four cell centres are inside the silhouette.
  const Q = N - 1
  const quad = new Uint8Array(Q * Q)
  for (let y = 0; y < Q; y++)
    for (let x = 0; x < Q; x++) {
      const i = y * N + x
      quad[y * Q + x] = sil[i] && sil[i + 1] && sil[i + N] && sil[i + N + 1] ? 1 : 0
    }
  const hasQuad = (x: number, y: number) => x >= 0 && y >= 0 && x < Q && y < Q && quad[y * Q + x] === 1

  const idx: number[] = []
  for (let y = 0; y < Q; y++)
    for (let x = 0; x < Q; x++) {
      if (!hasQuad(x, y)) continue
      const c00 = y * N + x
      const c10 = c00 + 1
      const c01 = c00 + N
      const c11 = c01 + 1
      // Counter-clockwise seen from +z: c01 -> c11 -> c10 -> c00.
      idx.push(fIdx[c01], fIdx[c11], fIdx[c10], fIdx[c01], fIdx[c10], fIdx[c00])
      idx.push(bIdx[c01], bIdx[c10], bIdx[c11], bIdx[c01], bIdx[c00], bIdx[c10])
      const edges: [number, number, boolean][] = [
        [c01, c11, !hasQuad(x, y + 1)],
        [c11, c10, !hasQuad(x + 1, y)],
        [c10, c00, !hasQuad(x, y - 1)],
        [c00, c01, !hasQuad(x - 1, y)],
      ]
      for (const [a, b, open] of edges) {
        if (!open) continue
        idx.push(fIdx[a], bIdx[a], bIdx[b], fIdx[a], bIdx[b], fIdx[b])
      }
    }

  let geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
  geo.setIndex(idx)
  geo = mergeVertices(geo)
  geo.computeVertexNormals()
  geo.computeBoundingBox()

  const mat = new THREE.MeshStandardMaterial({
    map: buildTexture(pre, opts.material),
    roughness: opts.material.roughness,
    metalness: opts.material.metalness,
    name: `s2s_${opts.material.id}`,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.name = 'sketch2shape_mesh'
  mesh.castShadow = true
  const group = new THREE.Group()
  group.name = 'Sketch2Shape'
  group.add(mesh)
  return group
}
