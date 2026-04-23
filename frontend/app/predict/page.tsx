'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { ERR_UNAUTHORIZED, formatClassName, isLoggedIn, predict, PredictResult } from '@/lib/api'

export default function PredictPage() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!isLoggedIn()) router.replace('/login')
  }, [router])

  const pickFile = useCallback((f: File) => {
    setFile(f)
    setResult(null)
    setError('')
    setPreview(prev => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(f)
    })
  }, [])

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) pickFile(f)
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) pickFile(f)
  }

  async function handleAnalyze() {
    if (!file) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await predict(file)
      setResult(res)
    } catch (err: unknown) {
      if (err instanceof Error && err.message === ERR_UNAUTHORIZED) {
        router.replace('/login')
      } else {
        setError(err instanceof Error ? err.message : 'Something went wrong.')
      }
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setPreview(prev => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const isHealthy = result?.predicted_class.toLowerCase().includes('healthy')

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">Analyze Plant Leaf</h1>
        <p className="text-sm text-gray-500 mb-6">
          Upload a clear image of a plant leaf to detect diseases.
        </p>

        {/* Upload area */}
        {!result && (
          <div
            className={`border-2 border-dashed rounded-2xl p-8 text-center transition-colors cursor-pointer ${
              dragging
                ? 'border-green-400 bg-green-50'
                : 'border-green-200 hover:border-green-400 hover:bg-green-50 bg-white'
            }`}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              className="hidden"
              onChange={onInputChange}
            />

            {preview ? (
              <div className="flex flex-col items-center gap-4">
                <img
                  src={preview}
                  alt="Selected leaf"
                  className="max-h-56 rounded-xl object-contain shadow-sm"
                />
                <p className="text-sm text-gray-500">{file?.name}</p>
                <div className="flex gap-3">
                  <button
                    onClick={e => { e.stopPropagation(); handleReset() }}
                    className="text-sm text-gray-500 hover:text-gray-700 border border-gray-200 rounded-lg px-4 py-1.5 transition-colors"
                  >
                    Change
                  </button>
                  <button
                    onClick={e => { e.stopPropagation(); handleAnalyze() }}
                    disabled={loading}
                    className="text-sm bg-green-600 text-white rounded-lg px-5 py-1.5 hover:bg-green-700 disabled:opacity-50 font-medium transition-colors"
                  >
                    {loading ? 'Analyzing...' : 'Analyze'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 text-gray-400">
                <svg className="w-12 h-12 text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M12 16v-8m0 0l-3 3m3-3l3 3M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1" />
                </svg>
                <p className="text-sm font-medium text-gray-600">
                  Drop an image here or <span className="text-green-600">browse</span>
                </p>
                <p className="text-xs">JPEG or PNG</p>
              </div>
            )}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center gap-3 py-12 text-green-600">
            <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="text-sm font-medium">Running inference...</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 bg-red-50 border border-red-100 text-red-700 text-sm rounded-xl px-4 py-3">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Diagnosis card */}
            <div className={`rounded-2xl p-6 border ${isHealthy ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-gray-500 mb-1">Diagnosis</p>
                  <h2 className="text-xl font-semibold text-gray-900">
                    {formatClassName(result.predicted_class)}
                  </h2>
                  <p className="text-sm text-gray-500 mt-1">Record #{result.record_id}</p>
                </div>
                <span className={`text-2xl font-bold ${isHealthy ? 'text-green-600' : 'text-amber-600'}`}>
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {/* Confidence bar */}
              <div className="mt-4">
                <div className="h-2 bg-white rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${isHealthy ? 'bg-green-500' : 'bg-amber-500'}`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Top-k */}
              <div className="mt-4 space-y-2">
                {result.top_k.map(item => (
                  <div key={item.label} className="flex items-center justify-between text-sm">
                    <span className="text-gray-700">{formatClassName(item.label)}</span>
                    <span className="font-medium text-gray-900">{(item.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Grad-CAM images */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-white rounded-2xl border border-gray-100 p-4">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
                  Grad-CAM Heatmap
                </p>
                <img
                  src={`data:image/png;base64,${result.gradcam_heatmap}`}
                  alt="Grad-CAM heatmap"
                  className="w-full rounded-xl object-contain"
                />
              </div>
              <div className="bg-white rounded-2xl border border-gray-100 p-4">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
                  Overlay
                </p>
                <img
                  src={`data:image/png;base64,${result.overlay_image}`}
                  alt="Overlay image"
                  className="w-full rounded-xl object-contain"
                />
              </div>
            </div>

            <button
              onClick={handleReset}
              className="w-full border border-green-200 text-green-700 hover:bg-green-50 rounded-xl py-2.5 text-sm font-medium transition-colors"
            >
              Analyze another image
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
