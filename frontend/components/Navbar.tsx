'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter, usePathname } from 'next/navigation'
import { getUsername, logout } from '@/lib/api'

export default function Navbar() {
  const router = useRouter()
  const pathname = usePathname()
  const [username, setUsername] = useState<string | null>(null)

  useEffect(() => {
    setUsername(getUsername())
  }, [])

  function handleLogout() {
    logout()
    router.push('/login')
  }

  const navLink = (href: string, label: string) => (
    <Link
      href={href}
      className={`text-sm font-medium px-3 py-1.5 rounded-lg transition-colors ${
        pathname === href
          ? 'bg-green-100 text-green-700'
          : 'text-gray-600 hover:text-green-700 hover:bg-green-50'
      }`}
    >
      {label}
    </Link>
  )

  return (
    <header className="bg-white border-b border-green-100 sticky top-0 z-10">
      <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <span className="font-semibold text-green-700 flex items-center gap-1.5">
            <span>🌿</span> Plant Disease AI
          </span>
          <nav className="flex gap-1">
            {navLink('/predict', 'Analyze')}
            {navLink('/history', 'History')}
          </nav>
        </div>
        <div className="flex items-center gap-3 text-sm text-gray-500">
          <span>{username}</span>
          <button
            onClick={handleLogout}
            className="text-gray-500 hover:text-red-600 transition-colors"
          >
            Sign out
          </button>
        </div>
      </div>
    </header>
  )
}
