"use client"

import { useEffect, useRef } from "react"
import { useTmsStore } from "@/store"
import type { LiveEvent } from "@/lib/types"

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true"
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws"

export function useWebSocket() {
  const { setWsStatus, appendEvent } = useTmsStore()
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (USE_MOCK) {
      // In mock mode, simulate a connected WS and replay mock events
      setWsStatus("connected")
      return
    }

    function connect() {
      setWsStatus("connecting")

      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setWsStatus("connected")
      }

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data as string) as LiveEvent
          appendEvent(parsed)
        } catch {
          // Silently ignore unparseable frames
        }
      }

      ws.onerror = () => {
        setWsStatus("error")
      }

      ws.onclose = () => {
        setWsStatus("disconnected")
        // Reconnect after 5 s
        reconnectTimeoutRef.current = setTimeout(connect, 5_000)
      }
    }

    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect on unmount
        wsRef.current.close()
      }
      setWsStatus("disconnected")
    }
  }, [setWsStatus, appendEvent])
}
