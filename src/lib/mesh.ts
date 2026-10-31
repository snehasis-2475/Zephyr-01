// Mesh validation and export helpers shared by both engines.
import * as THREE from 'three'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

export interface MeshReport {
  meshes: number
  vertices: number
  triangles: number
  size: [number, number, number]
  boundaryEdges: number
  nonManifoldEdges: number
  watertight: boolean
  hasUV: boolean
  hasTexture: boolean
  hasVertexColors: boolean
  valid: boolean
  issues: string[]
}

export function validateObject(root: THREE.Object3D): MeshReport {
  let meshes = 0
  let vertices = 0
  let triangles = 0
  let boundaryEdges = 0
  let nonManifoldEdges = 0
  let hasUV = false
  let hasTexture = false
  let hasVertexColors = false
  root.updateMatrixWorld(true)

  root.traverse((o) => {
    const m = o as THREE.Mesh
    if (!m.isMesh || !m.geometry) return
    meshes++
    const g = m.geometry
    const p = g.getAttribute('position')
    if (!p) return
    vertices += p.count
    triangles += g.index ? g.index.count / 3 : p.count / 3
    hasUV ||= !!g.getAttribute('uv')
    hasVertexColors ||= !!g.getAttribute('color')
    const mats = Array.isArray(m.material) ? m.material : [m.material]
    hasTexture ||= mats.some((mt) => !!(mt as THREE.MeshStandardMaterial).map)

    // Topology check on position-only welded geometry.
    const welded = mergeVertices(new THREE.BufferGeometry().setAttribute('position', p.clone()), 1e-5)
    const index = welded.index
    if (!index) return
    const edges = new Map<number, number>()
    const nv = welded.getAttribute('position').count
    for (let t = 0; t < index.count; t += 3) {
      const a = index.getX(t)
      const b = index.getX(t + 1)
      const c = index.getX(t + 2)
      for (const [u, v] of [
        [a, b],
        [b, c],
        [c, a],
      ]) {
        const key = Math.min(u, v) * nv + Math.max(u, v)
        edges.set(key, (edges.get(key) ?? 0) + 1)
      }
    }
    for (const n of edges.values()) {
      if (n === 1) boundaryEdges++
      else if (n > 2) nonManifoldEdges++
    }
    welded.dispose()
  })

  const box = new THREE.Box3().setFromObject(root)
  const s = box.isEmpty() ? new THREE.Vector3() : box.getSize(new THREE.Vector3())
  const size: [number, number, number] = [s.x, s.y, s.z]
  const issues: string[] = []
  if (!meshes || !triangles) issues.push('No geometry found')
  const flat = Math.min(...size) < Math.max(...size) * 0.02
  if (triangles && flat) issues.push('Geometry is nearly flat (no depth)')
  if (boundaryEdges) issues.push(`${boundaryEdges} open boundary edges`)
  if (nonManifoldEdges) issues.push(`${nonManifoldEdges} non-manifold edges`)

  return {
    meshes,
    vertices,
    triangles: Math.round(triangles),
    size,
    boundaryEdges,
    nonManifoldEdges,
    watertight: triangles > 0 && boundaryEdges === 0 && nonManifoldEdges === 0,
    hasUV,
    hasTexture,
    hasVertexColors,
    valid: meshes > 0 && triangles > 0 && !flat,
    issues,
  }
}

export async function exportGLB(root: THREE.Object3D): Promise<Blob> {
  const exporter = new GLTFExporter()
  const result = await exporter.parseAsync(root, { binary: true })
  return new Blob([result as ArrayBuffer], { type: 'model/gltf-binary' })
}

export function exportOBJ(root: THREE.Object3D): Blob {
  const text = new OBJExporter().parse(root)
  return new Blob([`# Exported by Sketch2Shape AI\n${text}`], { type: 'text/plain' })
}

export async function loadGLB(data: ArrayBuffer): Promise<THREE.Group> {
  const gltf = await new GLTFLoader().parseAsync(data, '')
  return gltf.scene
}

/** Round-trip check: the exported GLB must load back with non-empty geometry. */
export async function verifyGLB(blob: Blob): Promise<boolean> {
  try {
    const scene = await loadGLB(await blob.arrayBuffer())
    return validateObject(scene).triangles > 0
  } catch {
    return false
  }
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}
