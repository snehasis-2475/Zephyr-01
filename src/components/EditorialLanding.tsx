import { ArrowRight } from 'lucide-react'
import { ApolloHeroViewer } from './ApolloHeroViewer'

interface Props {
  onEnterStudio: () => void
}

export function EditorialLanding({ onEnterStudio }: Props) {
  return (
    <div className="relative h-screen w-screen overflow-hidden bg-paper text-ink select-none flex flex-col justify-between p-6 sm:p-8 lg:p-12 antialiased">
      {/* Background Subtle Texture */}
      <div className="absolute inset-0 paper-dots opacity-35 pointer-events-none" />

      {/* 1. Minimal Masthead */}
      <header className="relative z-20 flex items-center justify-between border-b border-ink/10 pb-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 font-display text-[16px] font-bold tracking-tight text-ink">
            <span className="inline-block h-2 w-2 rounded-full bg-accent" />
            <span>Morph</span>
            <span
              className="italic font-normal text-accent text-[20px] -ml-0.5"
              style={{ fontFamily: "'Pinyon Script', 'Newsreader', cursive" }}
            >
              ts
            </span>
          </div>
          <span className="h-3 w-px bg-ink/20" />
          <div className="font-mono text-[10.5px] uppercase tracking-widest text-ink/45 hidden sm:block">
            3D SYNTHESIS ATELIER
          </div>
        </div>

        {/* Top-Right Action to Studio */}
        <div className="flex items-center gap-3">
          <a
            href="/Morph_ts_Technical_Documentation.pdf"
            download="Morph_ts_Technical_Documentation.pdf"
            className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-ink/12 bg-white/60 px-3.5 py-1.5 font-mono text-[11px] text-ink/70 hover:border-ink/25 hover:text-ink transition"
          >
            <span>Dossier (PDF)</span>
          </a>
          <button
            onClick={onEnterStudio}
            className="group flex items-center gap-2 rounded-full bg-ink px-5 py-2 font-sans text-[12.5px] font-semibold text-paper shadow-md hover:bg-night transition-all hover:shadow-lg active:scale-98"
          >
            <span>Enter Studio</span>
            <ArrowRight size={14} className="transition-transform group-hover:translate-x-1" />
          </button>
        </div>
      </header>

      {/* 2. Hero Section */}
      <main className="relative z-10 flex-1 grid grid-cols-1 lg:grid-cols-12 items-center gap-6 lg:gap-12 py-4 sm:py-6 overflow-hidden">
        {/* Left Narrative */}
        <div className="lg:col-span-5 flex flex-col justify-center max-w-lg z-20">
          <h1 className="text-5xl sm:text-7xl lg:text-8xl font-light tracking-tight text-ink leading-none mb-5 flex items-baseline">
            <span className="font-serif">Morph</span>
            <span
              className="italic font-normal text-accent text-[1.25em] ml-2.5 tracking-normal"
              style={{ fontFamily: "'Pinyon Script', 'Newsreader', cursive" }}
            >
              ts
            </span>
          </h1>

          <p className="font-sans text-[14px] sm:text-[15px] text-ink/65 leading-relaxed mb-8 max-w-md">
            Synthesizing hand-drawn human gesture into watertight spatial geometry in real time.
          </p>

          {/* Primary Action Button to Studio */}
          <div>
            <button
              onClick={onEnterStudio}
              className="group flex items-center gap-3 rounded-2xl bg-ink px-7 py-4 font-sans text-[14px] font-semibold text-paper shadow-xl hover:bg-night transition-all hover:scale-102 active:scale-98"
            >
              <span>Launch 3D Studio</span>
              <ArrowRight size={16} className="transition-transform group-hover:translate-x-1 text-accent" />
            </button>
          </div>
        </div>

        {/* Right Stage: Studio Apollo */}
        <div className="lg:col-span-7 h-[420px] sm:h-[500px] lg:h-full w-full relative rounded-3xl border border-ink/8 bg-linear-to-b from-white/70 to-paper-2/30 shadow-xs overflow-hidden flex items-center justify-center">
          <ApolloHeroViewer className="h-full w-full" />
        </div>
      </main>

      {/* 3. Minimal Footer */}
      <footer className="relative z-20 flex items-center justify-between border-t border-ink/10 pt-3 font-mono text-[10.5px] text-ink/40">
        <div className="flex items-center gap-1">
          <span>MORPH</span>
          <span
            className="italic text-accent text-[13px]"
            style={{ fontFamily: "'Pinyon Script', 'Newsreader', cursive" }}
          >
            ts
          </span>
        </div>
        <div>DRAG TO ORBIT · SCROLL TO ZOOM</div>
      </footer>
    </div>
  )
}
