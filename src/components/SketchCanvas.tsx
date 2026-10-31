import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { Eraser, PenLine, Redo2, Trash2, Undo2 } from 'lucide-react'

export interface SketchCanvasHandle {
  getCanvas: () => HTMLCanvasElement | null
  isBlank: () => boolean
  clear: () => void
  drawExample: (name: ExampleName) => void
}

export type ExampleName = 'mug' | 'vase' | 'lamp'

const SIZE = 1024
const MAX_HISTORY = 40

interface Props {
  onChange: () => void
}

export const SketchCanvas = forwardRef<SketchCanvasHandle, Props>(function SketchCanvas({ onChange }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [tool, setTool] = useState<'pen' | 'eraser'>('pen')
  const [size, setSize] = useState(8)
  const [undoStack, setUndo] = useState<ImageData[]>([])
  const [redoStack, setRedo] = useState<ImageData[]>([])
  const drawing = useRef(false)
  const pts = useRef<{ x: number; y: number }[]>([])
  const blank = useRef(true)

  const ctx = () => canvasRef.current?.getContext('2d', { willReadFrequently: true }) ?? null

  const reset = useCallback(() => {
    const c = ctx()
    if (!c) return
    c.globalCompositeOperation = 'source-over'
    c.fillStyle = '#ffffff'
    c.fillRect(0, 0, SIZE, SIZE)
    blank.current = true
  }, [])

  useEffect(() => {
    reset()
  }, [reset])

  const snapshot = () => {
    const c = ctx()
    if (!c) return
    const img = c.getImageData(0, 0, SIZE, SIZE)
    setUndo((s) => [...s.slice(-MAX_HISTORY + 1), img])
    setRedo([])
  }

  const strokeStyle = (c: CanvasRenderingContext2D, pressure: number) => {
    c.lineCap = 'round'
    c.lineJoin = 'round'
    c.globalCompositeOperation = 'source-over'
    c.strokeStyle = tool === 'eraser' ? '#ffffff' : '#141416'
    c.lineWidth = (tool === 'eraser' ? size * 3 : size) * (0.6 + pressure * 0.8)
  }

  const toCanvas = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE }
  }

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return
    e.currentTarget.setPointerCapture(e.pointerId)
    snapshot()
    drawing.current = true
    const p = toCanvas(e)
    pts.current = [p]
    const c = ctx()!
    strokeStyle(c, e.pressure || 0.5)
    c.beginPath()
    c.arc(p.x, p.y, c.lineWidth / 2, 0, Math.PI * 2)
    c.fillStyle = c.strokeStyle
    c.fill()
    if (tool === 'pen') blank.current = false
  }

  const onMove = (e: React.PointerEvent) => {
    if (!drawing.current) return
    const c = ctx()!
    const p = toCanvas(e)
    const list = pts.current
    list.push(p)
    if (list.length < 3) return
    // Quadratic smoothing through midpoints.
    const [a, b, d] = list.slice(-3)
    const m1 = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
    const m2 = { x: (b.x + d.x) / 2, y: (b.y + d.y) / 2 }
    strokeStyle(c, e.pressure || 0.5)
    c.beginPath()
    c.moveTo(m1.x, m1.y)
    c.quadraticCurveTo(b.x, b.y, m2.x, m2.y)
    c.stroke()
  }

  const onUp = () => {
    if (!drawing.current) return
    drawing.current = false
    pts.current = []
    onChange()
  }

  const undo = () => {
    const c = ctx()
    if (!c || !undoStack.length) return
    setRedo((r) => [...r, c.getImageData(0, 0, SIZE, SIZE)])
    c.putImageData(undoStack[undoStack.length - 1], 0, 0)
    setUndo((s) => s.slice(0, -1))
    blank.current = false
    onChange()
  }

  const redo = () => {
    const c = ctx()
    if (!c || !redoStack.length) return
    setUndo((s) => [...s, c.getImageData(0, 0, SIZE, SIZE)])
    c.putImageData(redoStack[redoStack.length - 1], 0, 0)
    setRedo((r) => r.slice(0, -1))
    blank.current = false
    onChange()
  }

  const clear = () => {
    snapshot()
    reset()
    onChange()
  }

  const drawExample = (name: ExampleName) => {
    const c = ctx()
    if (!c) return
    snapshot()
    reset()
    c.strokeStyle = '#141416'
    c.lineWidth = 9
    c.lineCap = 'round'
    c.lineJoin = 'round'
    c.beginPath()
    if (name === 'mug') {
      c.moveTo(330, 300)
      c.lineTo(345, 760)
      c.quadraticCurveTo(350, 800, 400, 805)
      c.lineTo(600, 805)
      c.quadraticCurveTo(650, 800, 655, 760)
      c.lineTo(670, 300)
      c.closePath()
      c.moveTo(668, 390)
      c.bezierCurveTo(820, 380, 830, 640, 660, 650)
      c.moveTo(664, 445)
      c.bezierCurveTo(760, 445, 760, 590, 662, 595)
      c.stroke()
      c.beginPath()
      c.ellipse(500, 300, 170, 34, 0, 0, Math.PI * 2)
      c.moveTo(360, 520)
      c.quadraticCurveTo(500, 560, 645, 520)
    } else if (name === 'vase') {
      c.moveTo(440, 180)
      c.bezierCurveTo(450, 300, 330, 360, 320, 520)
      c.bezierCurveTo(310, 700, 390, 840, 500, 850)
      c.bezierCurveTo(610, 840, 690, 700, 680, 520)
      c.bezierCurveTo(670, 360, 550, 300, 560, 180)
      c.closePath()
      c.moveTo(345, 480)
      c.quadraticCurveTo(500, 520, 655, 480)
      c.moveTo(330, 620)
      c.quadraticCurveTo(500, 665, 670, 620)
    } else {
      c.moveTo(380, 420)
      c.lineTo(320, 600)
      c.lineTo(680, 600)
      c.lineTo(620, 420)
      c.closePath()
      c.moveTo(500, 600)
      c.lineTo(500, 820)
      c.moveTo(380, 860)
      c.quadraticCurveTo(500, 800, 620, 860)
      c.closePath()
      c.moveTo(420, 420)
      c.quadraticCurveTo(500, 330, 580, 420)
    }
    c.stroke()
    blank.current = false
    onChange()
  }

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    isBlank: () => blank.current,
    clear,
    drawExample,
  }))

  const toolBtn = (active: boolean) =>
    `flex h-9 items-center gap-1.5 rounded-lg px-3 text-[13px] font-medium transition ${
      active ? 'bg-ink text-paper' : 'text-ink/70 hover:bg-ink/5 hover:text-ink'
    }`
  const iconBtn =
    'grid h-9 w-9 place-items-center rounded-lg text-ink/70 transition hover:bg-ink/5 hover:text-ink disabled:opacity-30 disabled:hover:bg-transparent'

  return (
    <div className="flex flex-col gap-3">
      <div className="relative aspect-square w-full overflow-hidden rounded-xl border border-ink/10 bg-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.6)]">
        <canvas
          ref={canvasRef}
          width={SIZE}
          height={SIZE}
          className={`h-full w-full touch-none ${tool === 'eraser' ? 'cursor-cell' : 'cursor-crosshair'}`}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onPointerLeave={onUp}
        />
        <div className="paper-dots pointer-events-none absolute inset-0 mix-blend-multiply" />
      </div>

      <div className="flex flex-wrap items-center gap-1 rounded-xl border border-ink/10 bg-paper-2 p-1">
        <button className={toolBtn(tool === 'pen')} onClick={() => setTool('pen')}>
          <PenLine size={15} /> Brush
        </button>
        <button className={toolBtn(tool === 'eraser')} onClick={() => setTool('eraser')}>
          <Eraser size={15} /> Eraser
        </button>
        <label className="ml-1 flex items-center gap-2 px-2 text-[12px] text-ink/60">
          <span className="font-mono">{size}px</span>
          <input
            type="range"
            min={2}
            max={28}
            value={size}
            onChange={(e) => setSize(+e.target.value)}
            className="w-20 accent-[var(--color-accent)]"
            aria-label="Brush size"
          />
        </label>
        <div className="ml-auto flex items-center">
          <button className={iconBtn} onClick={undo} disabled={!undoStack.length} title="Undo">
            <Undo2 size={16} />
          </button>
          <button className={iconBtn} onClick={redo} disabled={!redoStack.length} title="Redo">
            <Redo2 size={16} />
          </button>
          <button className={iconBtn} onClick={clear} title="Reset canvas">
            <Trash2 size={16} />
          </button>
        </div>
      </div>
    </div>
  )
})
