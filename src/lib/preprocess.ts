// Client-side sketch conditioning: normalises a drawn or uploaded sketch into a
// clean 512x512 "black ink on white paper" image that both the local engine and
// the Hunyuan3D backend consume.

export interface PreprocessOptions {
  contrast: number // 0.5 – 2.5
  threshold: number // 0 – 255, luminance below this counts as ink
  denoise: boolean
  autoCrop: boolean
  strokeBoost: number // 0 – 3 dilation passes
}

export const DEFAULT_PREPROCESS: PreprocessOptions = {
  contrast: 1.3,
  threshold: 175,
  denoise: true,
  autoCrop: true,
  strokeBoost: 1,
}

export interface PreprocessResult {
  canvas: HTMLCanvasElement
  mask: Uint8Array // 1 = ink, row-major SIZE x SIZE
  size: number
  inkRatio: number
  inverted: boolean
  empty: boolean
}

export const OUTPUT_SIZE = 512
const WORK_MAX = 1024

type Source = HTMLCanvasElement | HTMLImageElement

function sourceSize(src: Source) {
  return src instanceof HTMLImageElement
    ? { w: src.naturalWidth, h: src.naturalHeight }
    : { w: src.width, h: src.height }
}

function dilate(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(mask)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      if (mask[i]) continue
      if (mask[i - 1] || mask[i + 1] || mask[i - w] || mask[i + w]) out[i] = 1
    }
  }
  return out
}

export function preprocessSketch(src: Source, opts: PreprocessOptions): PreprocessResult {
  const { w: sw, h: sh } = sourceSize(src)
  const scale = Math.min(1, WORK_MAX / Math.max(sw, sh, 1))
  const w = Math.max(1, Math.round(sw * scale))
  const h = Math.max(1, Math.round(sh * scale))

  const work = document.createElement('canvas')
  work.width = w
  work.height = h
  const wctx = work.getContext('2d', { willReadFrequently: true })!
  // Transparent pixels become paper white.
  wctx.fillStyle = '#fff'
  wctx.fillRect(0, 0, w, h)
  wctx.drawImage(src, 0, 0, w, h)
  const { data } = wctx.getImageData(0, 0, w, h)

  // 1. Luminance + auto-invert for dark-background sketches (chalkboard, scans).
  let lum = new Float32Array(w * h)
  let sum = 0
  for (let i = 0; i < w * h; i++) {
    const v = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]
    lum[i] = v
    sum += v
  }
  const inverted = sum / (w * h) < 110
  if (inverted) for (let i = 0; i < lum.length; i++) lum[i] = 255 - lum[i]

  // 2. Contrast around mid-grey.
  for (let i = 0; i < lum.length; i++) lum[i] = (lum[i] - 128) * opts.contrast + 128

  // 3. Noise reduction: 3x3 box blur before thresholding.
  if (opts.denoise) {
    const b = new Float32Array(lum.length)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let acc = 0
        let n = 0
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy
          if (yy < 0 || yy >= h) continue
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx
            if (xx < 0 || xx >= w) continue
            acc += lum[yy * w + xx]
            n++
          }
        }
        b[y * w + x] = acc / n
      }
    }
    lum = b
  }

  // 4. Threshold into an ink mask.
  let mask: Uint8Array = new Uint8Array(w * h)
  for (let i = 0; i < lum.length; i++) mask[i] = lum[i] < opts.threshold ? 1 : 0

  // 5. Remove isolated specks (paper grain, scanner dust).
  if (opts.denoise) {
    const clean = new Uint8Array(mask)
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x
        if (!mask[i]) continue
        let n = 0
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && mask[i + dy * w + dx]) n++
        if (n < 2) clean[i] = 0
      }
    }
    mask = clean
  }

  // 6. Bounding box of ink for auto-crop + centring.
  let minX = w
  let minY = h
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }
  const empty = maxX < 0
  if (empty || !opts.autoCrop) {
    minX = 0
    minY = 0
    maxX = w - 1
    maxY = h - 1
  }

  // Paint the cleaned mask, then fit it into the square output with padding.
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = w
  maskCanvas.height = h
  const mctx = maskCanvas.getContext('2d')!
  const mimg = mctx.createImageData(w, h)
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] ? 0 : 255
    mimg.data[i * 4] = v
    mimg.data[i * 4 + 1] = v
    mimg.data[i * 4 + 2] = v
    mimg.data[i * 4 + 3] = 255
  }
  mctx.putImageData(mimg, 0, 0)

  const S = OUTPUT_SIZE
  const out = document.createElement('canvas')
  out.width = S
  out.height = S
  const octx = out.getContext('2d', { willReadFrequently: true })!
  octx.fillStyle = '#fff'
  octx.fillRect(0, 0, S, S)
  const bw = maxX - minX + 1
  const bh = maxY - minY + 1
  const pad = opts.autoCrop ? 0.14 : 0
  const fit = (S * (1 - pad * 2)) / Math.max(bw, bh)
  const dw = bw * fit
  const dh = bh * fit
  octx.imageSmoothingEnabled = true
  octx.drawImage(maskCanvas, minX, minY, bw, bh, (S - dw) / 2, (S - dh) / 2, dw, dh)

  // Re-binarise after resampling and apply stroke boost.
  const od = octx.getImageData(0, 0, S, S)
  let outMask: Uint8Array = new Uint8Array(S * S)
  for (let i = 0; i < S * S; i++) outMask[i] = od.data[i * 4] < 160 ? 1 : 0
  for (let k = 0; k < opts.strokeBoost; k++) outMask = dilate(outMask, S, S)
  let ink = 0
  for (let i = 0; i < S * S; i++) {
    const v = outMask[i] ? 0 : 255
    ink += outMask[i]
    od.data[i * 4] = v
    od.data[i * 4 + 1] = v
    od.data[i * 4 + 2] = v
    od.data[i * 4 + 3] = 255
  }
  octx.putImageData(od, 0, 0)

  return { canvas: out, mask: outMask, size: S, inkRatio: ink / (S * S), inverted, empty }
}

export function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Canvas export failed'))), 'image/png'),
  )
}
