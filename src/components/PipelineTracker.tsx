import { Check, CircleDashed, LoaderCircle, MinusCircle, XCircle } from 'lucide-react'
import type { StageId } from '../lib/api'

export type StageState = 'idle' | 'active' | 'done' | 'skipped' | 'error'

export interface StageView {
  id: StageId
  label: string
  detail: string
  state: StageState
  seconds?: number
}

export function PipelineTracker({ stages }: { stages: StageView[] }) {
  return (
    <ol className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-white/8 bg-white/8 sm:grid-cols-3 xl:grid-cols-6">
      {stages.map((s, i) => (
        <li
          key={s.id}
          className={`relative flex flex-col gap-1.5 bg-night-2 px-3.5 py-3 transition ${
            s.state === 'active' ? 'bg-night-3' : ''
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] tracking-[0.14em] text-white/35">0{i + 1}</span>
            <StageIcon state={s.state} />
          </div>
          <span
            className={`text-[13px] font-medium leading-tight ${
              s.state === 'idle' || s.state === 'skipped' ? 'text-white/45' : 'text-white'
            }`}
          >
            {s.label}
          </span>
          <span className="text-[11px] leading-snug text-white/40">{s.detail}</span>
          <span className="font-mono text-[11px] text-white/55">
            {s.state === 'skipped' ? 'skipped' : s.seconds != null ? `${s.seconds.toFixed(1)}s` : '—'}
          </span>
          {s.state === 'active' && (
            <span className="absolute inset-x-0 bottom-0 h-0.5 overflow-hidden bg-white/5">
              <span className="progress-sweep block h-full w-1/3 bg-accent" />
            </span>
          )}
        </li>
      ))}
    </ol>
  )
}

function StageIcon({ state }: { state: StageState }) {
  if (state === 'active') return <LoaderCircle size={14} className="animate-spin text-accent" />
  if (state === 'done') return <Check size={14} className="text-emerald-400" />
  if (state === 'error') return <XCircle size={14} className="text-red-400" />
  if (state === 'skipped') return <MinusCircle size={14} className="text-white/30" />
  return <CircleDashed size={14} className="text-white/25" />
}
