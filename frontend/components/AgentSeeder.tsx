"use client"

import { useEffect, useRef } from "react"
import { apiFetch } from "@/lib/api"

/**
 * Fires POST /api/v1/dev/trigger once when the dashboard is first opened.
 * Renders nothing — purely a side-effect component.
 */
export function AgentSeeder() {
  const hasFired = useRef(false)

  useEffect(() => {
    if (hasFired.current) return
    hasFired.current = true

    apiFetch("/api/v1/dev/trigger", { method: "POST" }).catch(() => {
      // Silently ignore — trigger is best-effort for demo
    })
  }, [])

  return null
}
