"use client"

import { useEffect, useRef } from "react"
import { useTmsStore } from "@/store"
import type { LiveEvent } from "@/lib/types"

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true"

// SSE endpoint — no auth header needed (EventSource limitation)
// Use full URL when backend is cross-origin (NEXT_PUBLIC_API_URL set),
// otherwise fall back to relative path for same-origin setups.
const _API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""
const SSE_URL = `${_API_BASE}/api/v1/events/stream`

export function useWebSocket() {
  const { setWsStatus, appendEvent } = useTmsStore()
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (USE_MOCK) {
      // In mock mode, simulate a connected stream (no real connection)
      setWsStatus("connected")
      return
    }

    function connect() {
      setWsStatus("connecting")

      const es = new EventSource(SSE_URL)
      esRef.current = es

      es.onopen = () => {
        setWsStatus("connected")
      }

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data as string) as LiveEvent
          appendEvent(parsed)
        } catch {
          // Silently ignore unparseable frames
        }
      }

      es.onerror = () => {
        // EventSource auto-reconnects — just surface the error state briefly
        setWsStatus("error")
        // The browser will reconnect automatically; status will flip to "connected" on reconnect
      }
    }

    connect()

    return () => {
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
      setWsStatus("disconnected")
    }
  }, [setWsStatus, appendEvent])
}
