import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js'
import { Pause, Play, RotateCcw } from 'lucide-react'

export type ApolloPose = 'profile' | 'fragment' | 'monolith' | 'aerial'
export type ApolloMaterialMode = 'textured' | 'alabaster' | 'wireframe'

interface PoseDef {
  id: ApolloPose
  label: string
  cam: [number, number, number]
  target: [number, number, number]
}

const POSES: PoseDef[] = [
  { id: 'profile', label: '3/4', cam: [2.8, 0.6, 3.6], target: [0, -0.05, 0] },
  { id: 'fragment', label: 'Detail', cam: [1.3, 0.45, 1.7], target: [0, 0.12, 0] },
  { id: 'monolith', label: 'Front', cam: [0, 0.25, 4.0], target: [0, -0.05, 0] },
  { id: 'aerial', label: 'Top', cam: [2.0, 2.7, 2.4], target: [0, -0.15, 0] },
]

// Global cache so Apollo loads once in the session and persists instantly.
let cachedApolloGroup: THREE.Group | null = null
let cachedOriginalMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>()
let cachedTexture: THREE.Texture | null = null

// Offscreen canvas to generate soft studio radial drop-shadow
function createStudioShadowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 512
  const ctx = canvas.getContext('2d')!
  const gradient = ctx.createRadialGradient(256, 256, 16, 256, 256, 240)
  gradient.addColorStop(0, 'rgba(20, 20, 22, 0.26)')
  gradient.addColorStop(0.25, 'rgba(20, 20, 22, 0.14)')
  gradient.addColorStop(0.65, 'rgba(20, 20, 22, 0.03)')
  gradient.addColorStop(1, 'rgba(20, 20, 22, 0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 512, 512)
  const tex = new THREE.CanvasTexture(canvas)
  return tex
}

export function ApolloHeroViewer({ className = '' }: { className?: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [loading, setLoading] = useState(!cachedApolloGroup)
  const [progress, setProgress] = useState(cachedApolloGroup ? 100 : 0)
  const [progressText, setProgressText] = useState(cachedApolloGroup ? 'Ready' : 'Streaming OBJ...')
  const [activePose, setActivePose] = useState<ApolloPose>('profile')
  const [matMode, setMatMode] = useState<ApolloMaterialMode>('textured')
  const [autoRotate, setAutoRotate] = useState(true)

  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const modelRef = useRef<THREE.Group | null>(null)
  const animFrameRef = useRef<number | null>(null)

  const targetCamPos = useRef(new THREE.Vector3(2.8, 0.6, 3.6))
  const targetControlTarget = useRef(new THREE.Vector3(0, -0.05, 0))
  const isTransitioning = useRef(false)
  const isInteracting = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(34, container.clientWidth / container.clientHeight, 0.1, 50)
    camera.position.set(2.8, 0.6, 3.6)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.02
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap

    container.appendChild(renderer.domElement)

    // 2. Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = 1.2
    controls.maxDistance = 6.5
    controls.target.set(0, -0.05, 0)
    controls.maxPolarAngle = Math.PI * 0.82
    controlsRef.current = controls

    controls.addEventListener('start', () => {
      isInteracting.current = true
      isTransitioning.current = false
    })
    controls.addEventListener('end', () => {
      isInteracting.current = false
    })

    // 3. Studio Lighting Setup (Diffused softbox + subtle rim)
    const softboxKey = new THREE.DirectionalLight('#fffcf4', 2.0)
    softboxKey.position.set(3.2, 4.5, 3.0)
    softboxKey.castShadow = true
    softboxKey.shadow.mapSize.set(2048, 2048)
    softboxKey.shadow.bias = -0.0001
    scene.add(softboxKey)

    const studioRim = new THREE.DirectionalLight('#dce7f5', 1.2)
    studioRim.position.set(-3.0, 2.5, -2.5)
    scene.add(studioRim)

    const studioFill = new THREE.DirectionalLight('#ede6d8', 0.65)
    studioFill.position.set(-2.0, -0.5, 2.5)
    scene.add(studioFill)

    const ambientStudio = new THREE.HemisphereLight('#fcf8f0', '#2a2825', 0.6)
    scene.add(ambientStudio)

    // 4. Minimal Studio Plinth & Soft Radial Shadow
    const plinthGeo = new THREE.CylinderGeometry(0.72, 0.76, 0.07, 64)
    const plinthMat = new THREE.MeshStandardMaterial({
      color: '#ebe6db',
      roughness: 0.94,
      metalness: 0.01,
    })
    const plinthMesh = new THREE.Mesh(plinthGeo, plinthMat)
    plinthMesh.position.y = -0.74
    plinthMesh.receiveShadow = true
    scene.add(plinthMesh)

    // Soft Studio Drop Shadow Disc
    const shadowGeo = new THREE.PlaneGeometry(3.6, 3.6)
    const shadowMat = new THREE.MeshBasicMaterial({
      map: createStudioShadowTexture(),
      transparent: true,
      depthWrite: false,
    })
    const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat)
    shadowMesh.rotation.x = -Math.PI / 2
    shadowMesh.position.y = -0.77
    scene.add(shadowMesh)

    // 5. Material Applier Helper
    const applyMaterial = (mode: ApolloMaterialMode, group: THREE.Group) => {
      group.traverse((c) => {
        const mesh = c as THREE.Mesh
        if (!mesh.isMesh) return

        if (mode === 'textured') {
          const orig = cachedOriginalMaterials.get(mesh)
          if (orig) {
            mesh.material = orig
          } else if (cachedTexture) {
            mesh.material = new THREE.MeshStandardMaterial({
              map: cachedTexture,
              roughness: 0.7,
              metalness: 0.03,
            })
          }
        } else if (mode === 'alabaster') {
          mesh.material = new THREE.MeshStandardMaterial({
            color: '#eee9dd',
            roughness: 0.88,
            metalness: 0.01,
          })
        } else if (mode === 'wireframe') {
          mesh.material = new THREE.MeshBasicMaterial({
            color: '#ff4f1f',
            wireframe: true,
          })
        }
      })
    }

    // 6. Load Model (Scaled down to 0.58 for quiet, minimal studio proportion)
    const setupModel = (object: THREE.Group, tex: THREE.Texture) => {
      const box = new THREE.Box3().setFromObject(object)
      const center = box.getCenter(new THREE.Vector3())
      object.position.sub(center)

      // Smaller, minimal studio scale
      object.scale.setScalar(0.58)
      object.position.y = -0.06

      object.rotation.y = -0.28
      object.rotation.x = -0.03

      object.traverse((c) => {
        const mesh = c as THREE.Mesh
        if (mesh.isMesh) {
          mesh.castShadow = true
          mesh.receiveShadow = true
          const pbrMat = new THREE.MeshStandardMaterial({
            map: tex,
            roughness: 0.7,
            metalness: 0.03,
          })
          mesh.material = pbrMat
          cachedOriginalMaterials.set(mesh, pbrMat)
        }
      })

      cachedApolloGroup = object
      modelRef.current = object
      scene.add(object)
      applyMaterial(matMode, object)
      setProgress(100)
      setLoading(false)
    }

    if (cachedApolloGroup) {
      cachedApolloGroup.scale.setScalar(0.58)
      cachedApolloGroup.position.y = -0.06
      modelRef.current = cachedApolloGroup
      scene.add(cachedApolloGroup)
      applyMaterial(matMode, cachedApolloGroup)
      setLoading(false)
      setProgress(100)
    } else {
      setLoading(true)
      setProgress(10)
      setProgressText('Loading texture & geometry...')

      const textureLoader = new THREE.TextureLoader()
      const apolloTex = textureLoader.load(
        '/mc/Meshy_AI_Fragmented_Apollo_20260816_texture.png',
        (tex) => {
          tex.colorSpace = THREE.SRGBColorSpace
          cachedTexture = tex
        }
      )

      const mtlLoader = new MTLLoader()
      mtlLoader.setPath('/mc/')
      mtlLoader.load(
        'Meshy_AI_Fragmented_Apollo_20260816_texture.mtl',
        (materials) => {
          materials.preload()
          const objLoader = new OBJLoader()
          objLoader.setMaterials(materials)
          objLoader.setPath('/mc/')

          const createFallback = () => {
            const geo = new THREE.TorusKnotGeometry(0.55, 0.18, 128, 32)
            const mat = new THREE.MeshStandardMaterial({
              color: '#ebe6db',
              roughness: 0.85,
              metalness: 0.02,
            })
            const mesh = new THREE.Mesh(geo, mat)
            mesh.castShadow = true
            mesh.receiveShadow = true
            const grp = new THREE.Group()
            grp.add(mesh)
            grp.position.y = 0.1
            cachedApolloGroup = grp
            modelRef.current = grp
            scene.add(grp)
            setLoading(false)
            setProgress(100)
          }

          objLoader.load(
            'Meshy_AI_Fragmented_Apollo_20260816_texture.obj',
            (object) => setupModel(object, apolloTex),
            (xhr) => {
              if (xhr.lengthComputable && xhr.total > 0) {
                const percent = Math.min(99, Math.round((xhr.loaded / xhr.total) * 90) + 10)
                setProgress(percent)
                const mb = (xhr.loaded / (1024 * 1024)).toFixed(0)
                setProgressText(`Streaming OBJ · ${mb} MB`)
              }
            },
            () => createFallback()
          )
        },
        undefined,
        () => {
          const objLoader = new OBJLoader()
          objLoader.load(
            '/mc/Meshy_AI_Fragmented_Apollo_20260816_texture.obj',
            (obj) => setupModel(obj, apolloTex),
            undefined,
            () => {
              const geo = new THREE.TorusKnotGeometry(0.55, 0.18, 128, 32)
              const mat = new THREE.MeshStandardMaterial({
                color: '#ebe6db',
                roughness: 0.85,
                metalness: 0.02,
              })
              const mesh = new THREE.Mesh(geo, mat)
              mesh.castShadow = true
              mesh.receiveShadow = true
              const grp = new THREE.Group()
              grp.add(mesh)
              grp.position.y = 0.1
              cachedApolloGroup = grp
              modelRef.current = grp
              scene.add(grp)
              setLoading(false)
              setProgress(100)
            }
          )
        }
      )
    }

    // 7. Animation Loop
    let clock = new THREE.Clock()
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate)
      const delta = clock.getDelta()

      if (isTransitioning.current && cameraRef.current && controlsRef.current) {
        cameraRef.current.position.lerp(targetCamPos.current, 0.065)
        controlsRef.current.target.lerp(targetControlTarget.current, 0.065)
        controlsRef.current.update()
        if (cameraRef.current.position.distanceTo(targetCamPos.current) < 0.02) {
          cameraRef.current.position.copy(targetCamPos.current)
          controlsRef.current.target.copy(targetControlTarget.current)
          isTransitioning.current = false
        }
      } else if (controlsRef.current) {
        controlsRef.current.update()
      }

      // Quiet, slow studio rotation when untouched
      if (modelRef.current && autoRotate && !isInteracting.current && !isTransitioning.current) {
        modelRef.current.rotation.y += delta * 0.1
      }

      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!container) return
      const w = container.clientWidth
      const h = container.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      controls.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [autoRotate])

  const setPose = (pose: PoseDef) => {
    setActivePose(pose.id)
    targetCamPos.current.set(...pose.cam)
    targetControlTarget.current.set(...pose.target)
    isTransitioning.current = true
  }

  const changeMaterial = (mode: ApolloMaterialMode) => {
    setMatMode(mode)
    if (!modelRef.current) return

    modelRef.current.traverse((c) => {
      const mesh = c as THREE.Mesh
      if (!mesh.isMesh) return

      if (mode === 'textured') {
        const orig = cachedOriginalMaterials.get(mesh)
        if (orig) {
          mesh.material = orig
        } else if (cachedTexture) {
          mesh.material = new THREE.MeshStandardMaterial({
            map: cachedTexture,
            roughness: 0.7,
            metalness: 0.03,
          })
        }
      } else if (mode === 'alabaster') {
        mesh.material = new THREE.MeshStandardMaterial({
          color: '#eee9dd',
          roughness: 0.88,
          metalness: 0.01,
        })
      } else if (mode === 'wireframe') {
        mesh.material = new THREE.MeshBasicMaterial({
          color: '#ff4f1f',
          wireframe: true,
        })
      }
    })
  }

  return (
    <div className={`relative flex flex-col items-center justify-center overflow-hidden select-none ${className}`}>
      {/* 3D Viewport */}
      <div ref={containerRef} className="h-full w-full cursor-grab active:cursor-grabbing" />

      {/* Minimal Loading HUD */}
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-paper/85 backdrop-blur-xs p-6 text-center z-20">
          <div className="w-48 space-y-2">
            <div className="h-0.5 w-full bg-ink/10 rounded-full overflow-hidden">
              <div className="h-full bg-accent transition-all duration-300" style={{ width: `${progress}%` }} />
            </div>
            <div className="flex justify-between font-mono text-[10px] text-ink/50 uppercase tracking-widest">
              <span>{progressText}</span>
              <span>{progress}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Bottom Minimal Studio Controls Pill */}
      <div className="absolute bottom-4 z-10 flex items-center gap-2 rounded-full border border-ink/10 bg-white/70 px-2 py-1 shadow-xs backdrop-blur-md">
        {/* Poses */}
        <div className="flex items-center gap-0.5">
          {POSES.map((pose) => (
            <button
              key={pose.id}
              onClick={() => setPose(pose)}
              className={`rounded-full px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-all ${
                activePose === pose.id
                  ? 'bg-ink text-paper font-semibold'
                  : 'text-ink/50 hover:text-ink'
              }`}
            >
              {pose.label}
            </button>
          ))}
        </div>

        <span className="h-2.5 w-px bg-ink/15" />

        {/* Material */}
        <div className="flex items-center gap-0.5">
          <button
            onClick={() => changeMaterial('textured')}
            className={`rounded-full px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-all ${
              matMode === 'textured' ? 'bg-ink text-paper font-semibold' : 'text-ink/50 hover:text-ink'
            }`}
          >
            Marble
          </button>
          <button
            onClick={() => changeMaterial('alabaster')}
            className={`rounded-full px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-all ${
              matMode === 'alabaster' ? 'bg-ink text-paper font-semibold' : 'text-ink/50 hover:text-ink'
            }`}
          >
            Plaster
          </button>
          <button
            onClick={() => changeMaterial('wireframe')}
            className={`rounded-full px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-all ${
              matMode === 'wireframe' ? 'bg-ink text-paper font-semibold' : 'text-ink/50 hover:text-ink'
            }`}
          >
            Wire
          </button>
        </div>

        <span className="h-2.5 w-px bg-ink/15" />

        {/* Auto Rotate & Reset */}
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className={`p-1 rounded-full text-ink/40 hover:text-ink transition-colors ${autoRotate ? 'text-ink' : ''}`}
          title={autoRotate ? 'Pause Rotate' : 'Auto Rotate'}
        >
          {autoRotate ? <Pause size={11} /> : <Play size={11} className="translate-x-0.25" />}
        </button>
        <button
          onClick={() => setPose(POSES[0])}
          className="p-1 rounded-full text-ink/40 hover:text-ink transition-colors"
          title="Reset View"
        >
          <RotateCcw size={11} />
        </button>
      </div>
    </div>
  )
}
