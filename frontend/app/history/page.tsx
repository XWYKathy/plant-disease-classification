'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import {
  CLASS_NAMES,
  ERR_UNAUTHORIZED,
  fetchImageBlob,
  formatClassName,
  getHistory,
  HistoryItem,
  isLoggedIn,
  submitFeedback,
} from '@/lib/api'

function ImageModal({
  item,
  onClose,
}: {
  item: HistoryItem
  onClose: () => void
}) {
  const [src, setSrc] = useState<string | null>(null)
  const [err, setErr] = useState(false)

  useEffect(() => {
    let objectUrl: string | null = null
    fetchImageBlob(item.record_id)
      .then(url => { objectUrl = url; setSrc(url) })
      .catch(() => setErr(true))
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [item.record_id])

  // Close on backdrop click
  const backdropRef = useRef<HTMLDivElement>(null)
  function onBackdropClick(e: React.MouseEvent) {
    if (e.target === backdropRef.current) onClose()
  }

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      ref={backdropRef}
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
      onClick={onBackdropClick}
    >
      <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <div>
            <p className="font-medium text-gray-900 text-sm truncate max-w-[280px]">
              {item.original_filename}
            </p>
            <p className="text-xs text-gray-400 mt-0.5">Record #{item.record_id}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Image */}
        <div className="flex items-center justify-center bg-gray-50 min-h-[200px] p-4">
          {err ? (
            <p className="text-sm text-red-500">Image not available.</p>
          ) : src ? (
            <img
              src={src}
              alt={item.original_filename}
              className="max-h-96 max-w-full rounded-xl object-contain"
            />
          ) : (
            <svg className="animate-spin h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
        </div>
      </div>
    </div>
  )
}

export default function HistoryPage() {
  const router = useRouter()
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [feedbackError, setFeedbackError] = useState('')
  const [labelPicker, setLabelPicker] = useState<{ recordId: number; label: string } | null>(null)
  const [submitting, setSubmitting] = useState<number | null>(null)
  const [modalItem, setModalItem] = useState<HistoryItem | null>(null)

  useEffect(() => {
    if (!isLoggedIn()) { router.replace('/login'); return }
    getHistory()
      .then(setItems)
      .catch(err => {
        if (err.message === ERR_UNAUTHORIZED) router.replace('/login')
        else setError('Failed to load history.')
      })
      .finally(() => setLoading(false))
  }, [router])

  async function handleFeedback(recordId: number, isCorrect: boolean, correctLabel?: string) {
    setSubmitting(recordId)
    setFeedbackError('')
    try {
      const updated = await submitFeedback(recordId, isCorrect, correctLabel)
      setItems(prev =>
        prev.map(item =>
          item.record_id === recordId
            ? { ...item, user_feedback_correct: updated.user_feedback_correct, user_feedback_label: updated.user_feedback_label }
            : item,
        ),
      )
      setLabelPicker(null)
    } catch {
      setFeedbackError('Failed to submit feedback.')
    } finally {
      setSubmitting(null)
    }
  }

  function formatDate(iso: string) {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">Prediction History</h1>
        <p className="text-sm text-gray-500 mb-6">Your past analyses and feedback.</p>

        {loading && (
          <div className="flex items-center gap-2 text-green-600 text-sm py-8">
            <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading...
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-100 text-red-700 text-sm rounded-xl px-4 py-3">
            {error}
          </div>
        )}

        {feedbackError && (
          <div className="bg-red-50 border border-red-100 text-red-700 text-sm rounded-xl px-4 py-3">
            {feedbackError}
          </div>
        )}

        {!loading && items.length === 0 && !error && (
          <div className="text-center py-16 text-gray-400">
            <p className="text-4xl mb-3">🌿</p>
            <p className="text-sm">No predictions yet. Go analyze a plant leaf!</p>
          </div>
        )}

        {items.length > 0 && (
          <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-xs text-gray-500 uppercase tracking-wide">
                  <th className="text-left px-5 py-3 font-medium">File</th>
                  <th className="text-left px-5 py-3 font-medium">Prediction</th>
                  <th className="text-left px-5 py-3 font-medium">Confidence</th>
                  <th className="text-left px-5 py-3 font-medium">Date</th>
                  <th className="text-left px-5 py-3 font-medium">Feedback</th>
                </tr>
              </thead>
              <tbody>
                {items.map(item => {
                  const isHealthy = item.predicted_class.toLowerCase().includes('healthy')
                  return (
                    <tr key={item.record_id} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                      {/* Clickable filename */}
                      <td className="px-5 py-3 max-w-[160px]">
                        <button
                          onClick={() => setModalItem(item)}
                          className="text-green-600 hover:text-green-800 hover:underline text-left truncate block max-w-full"
                          title="View image"
                        >
                          {item.original_filename}
                        </button>
                      </td>
                      <td className="px-5 py-3">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                          isHealthy ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                        }`}>
                          {formatClassName(item.predicted_class)}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-gray-700 font-medium">
                        {(item.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-5 py-3 text-gray-500">
                        {formatDate(item.uploaded_at)}
                      </td>
                      <td className="px-5 py-3">
                        {item.user_feedback_correct !== null ? (
                          <div>
                            <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                              item.user_feedback_correct
                                ? 'bg-green-100 text-green-700'
                                : 'bg-red-100 text-red-700'
                            }`}>
                              {item.user_feedback_correct ? 'Correct' : 'Incorrect'}
                            </span>
                            {item.user_feedback_label && (
                              <p className="text-xs text-gray-400 mt-1">
                                → {formatClassName(item.user_feedback_label)}
                              </p>
                            )}
                          </div>
                        ) : labelPicker?.recordId === item.record_id ? (
                          <div className="flex flex-col gap-2">
                            <select
                              value={labelPicker.label}
                              onChange={e => setLabelPicker({ recordId: item.record_id, label: e.target.value })}
                              className="text-xs border border-gray-200 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-green-500"
                            >
                              <option value="">Select correct label...</option>
                              {CLASS_NAMES.map(cn => (
                                <option key={cn} value={cn}>{formatClassName(cn)}</option>
                              ))}
                            </select>
                            <div className="flex gap-2">
                              <button
                                disabled={!labelPicker.label || submitting === item.record_id}
                                onClick={() => handleFeedback(item.record_id, false, labelPicker.label)}
                                className="text-xs bg-red-500 text-white px-2.5 py-1 rounded-lg hover:bg-red-600 disabled:opacity-40 transition-colors"
                              >
                                {submitting === item.record_id ? '...' : 'Submit'}
                              </button>
                              <button
                                onClick={() => setLabelPicker(null)}
                                className="text-xs text-gray-500 hover:text-gray-700"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="flex gap-2">
                            <button
                              disabled={submitting === item.record_id}
                              onClick={() => handleFeedback(item.record_id, true)}
                              className="text-xs text-green-700 border border-green-200 px-2.5 py-1 rounded-lg hover:bg-green-50 disabled:opacity-40 transition-colors"
                            >
                              Correct
                            </button>
                            <button
                              onClick={() => setLabelPicker({ recordId: item.record_id, label: '' })}
                              className="text-xs text-red-600 border border-red-200 px-2.5 py-1 rounded-lg hover:bg-red-50 transition-colors"
                            >
                              Wrong
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </main>

      {modalItem && (
        <ImageModal item={modalItem} onClose={() => setModalItem(null)} />
      )}
    </div>
  )
}
