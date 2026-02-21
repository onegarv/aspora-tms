import { create } from "zustand"
import type { LiveEvent, User } from "@/lib/types"

export type WsStatus = "connecting" | "connected" | "disconnected" | "error"

interface TmsStore {
  // WebSocket connection state
  wsStatus: WsStatus
  setWsStatus: (status: WsStatus) => void

  // Authenticated user (mock: TREASURY_ADMIN)
  currentUser: User

  // Live event feed (last 50 events, newest first)
  liveEvents: LiveEvent[]
  appendEvent: (event: LiveEvent) => void
  clearEvents: () => void
}

export const useTmsStore = create<TmsStore>((set) => ({
  wsStatus: "disconnected",
  setWsStatus: (wsStatus) => set({ wsStatus }),

  currentUser: {
    id: "user-admin-1",
    name: "Vedant Tiwari",
    role: "TREASURY_ADMIN",
  },

  liveEvents: [],
  appendEvent: (event) =>
    set((state) => ({
      liveEvents: [event, ...state.liveEvents].slice(0, 50),
    })),
  clearEvents: () => set({ liveEvents: [] }),
}))
