import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

export type ViewMode = 'textured' | 'clay' | 'wireframe' | 'normals'

export interface Viewer3DHandle {
  resetCamera: () => void
}

interface Props {
  object: THREE.Object3D | null
  mode: ViewMode
  autoRotate: boolean
}

const clayMat = new THREE.MeshStandardMaterial({ color: '#cfc9bf', roughness: 0.8, metalness: 0 })
const wireMat = new THREE.MeshBasicMaterial({ color: '#ff6a3d', wireframe: true })
const normalMat = new THREE.MeshNormalMaterial()

export const Viewer3D = forwardRef<Viewer3DHandle, Props>(function Viewer3D({ object, mode, autoRotate }, ref) {
  const hostRef = useRef<HTMLDivElement>(null)
  const targetY = useRef(0)
  const ctx = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: OrbitControls
    stage: THREE.Group
    display: THREE.Object3D | null
    originals: Map<THREE.Mesh, THREE.Material | THREE.Material[]>
  } | null>(null)

  // One-time renderer setup.
  useEffect(() => {
    const host = hostRef.current!
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const pmrem = new THREE.PMREMGenerator(renderer)
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture

    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
    camera.position.set(2.6, 1.6, 3.4)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.target.set(0, 0, 0)
    controls.minDistance = 1
    controls.maxDistance = 12

    const key = new THREE.DirectionalLight('#ffffff', 1.6)
    key.position.set(3, 5, 4)
    key.castShadow = true
    key.shadow.mapSize.set(1024, 1024)
    scene.add(key, new THREE.HemisphereLight('#ffffff', '#3a3530', 0.5))

    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(3, 64),
      new THREE.ShadowMaterial({ opacity: 0.35 }),
    )
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -1.001
    ground.receiveShadow = true
    const grid = new THREE.GridHelper(6, 24, '#4a4640', '#2c2a27')
    grid.position.y = -1
    ;(grid.material as THREE.Material).transparent = true
    ;(grid.material as THREE.Material).opacity = 0.55
    scene.add(ground, grid)

    const stage = new THREE.Group()
    scene.add(stage)

    ctx.current = { renderer, scene, camera, controls, stage, display: null, originals: new Map() }

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = host
      renderer.setSize(w, h, false)
      renderer.domElement.style.width = '100%'
      renderer.domElement.style.height = '100%'
      camera.aspect = w / Math.max(h, 1)
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      controls.update()
      renderer.render(scene, camera)
    }
    loop()

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      pmrem.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      ctx.current = null
    }
  }, [])

  const resetCamera = () => {
    const c = ctx.current
    if (!c) return
    const y = targetY.current
    c.camera.position.set(2.6, 1.6 + y, 3.4)
    c.controls.target.set(0, y, 0)
    c.controls.update()
  }

  useImperativeHandle(ref, () => ({ resetCamera }))

  // Swap in a normalised display clone whenever the source object changes.
  useEffect(() => {
    const c = ctx.current
    if (!c) return
    if (c.display) c.stage.remove(c.display)
    c.originals.clear()
    c.display = null
    if (!object) return
    const display = object.clone(true)
    const box = new THREE.Box3().setFromObject(display)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())
    const s = 2 / Math.max(size.x, size.y, size.z, 1e-6)
    const wrapper = new THREE.Group()
    display.position.sub(center)
    wrapper.add(display)
    wrapper.scale.setScalar(s)
    wrapper.position.y = -1 + (size.y * s) / 2
    wrapper.traverse((o) => {
      const m = o as THREE.Mesh
      if (m.isMesh) {
        m.castShadow = true
        c.originals.set(m, m.material)
      }
    })
    c.stage.add(wrapper)
    c.display = wrapper
    targetY.current = wrapper.position.y
    resetCamera()
  }, [object])

  useEffect(() => {
    const c = ctx.current
    if (!c) return
    c.originals.forEach((orig, mesh) => {
      mesh.material =
        mode === 'textured' ? orig : mode === 'clay' ? clayMat : mode === 'wireframe' ? wireMat : normalMat
    })
  }, [mode, object])

  useEffect(() => {
    const c = ctx.current
    if (!c) return
    c.controls.autoRotate = autoRotate
    c.controls.autoRotateSpeed = 1.6
  }, [autoRotate])

  return <div ref={hostRef} className="absolute inset-0" />
})
