"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  LayoutDashboard,
  ArrowLeftRight,
  TrendingUp,
  BarChart2,
  CheckSquare,
  Settings,
  Activity,
  Wifi,
  WifiOff,
  Loader2,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useTmsStore } from "@/store"
import type { WsStatus } from "@/store"

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, exact: true },
  { href: "/dashboard/deals", label: "Deals", icon: ArrowLeftRight },
  { href: "/dashboard/pnl", label: "P&L", icon: TrendingUp },
  { href: "/dashboard/exposure", label: "Exposure", icon: BarChart2 },
  { href: "/approvals", label: "Approvals", icon: CheckSquare },
  { href: "/admin", label: "Admin", icon: Settings },
]

function WsIndicator({ status }: { status: WsStatus }) {
  if (status === "connected")
    return <Wifi className="size-3.5 text-green-500" aria-label="Connected" />
  if (status === "connecting")
    return (
      <Loader2 className="size-3.5 animate-spin text-yellow-500" aria-label="Connecting" />
    )
  if (status === "error")
    return <WifiOff className="size-3.5 text-destructive" aria-label="Error" />
  return <WifiOff className="size-3.5 text-muted-foreground" aria-label="Disconnected" />
}

export function Sidebar() {
  const pathname = usePathname()
  const { wsStatus, currentUser } = useTmsStore()

  function isActive(href: string, exact?: boolean) {
    if (exact) return pathname === href
    return pathname.startsWith(href)
  }

  return (
    <aside className="flex h-screen w-56 flex-col border-r bg-sidebar text-sidebar-foreground">
      {/* Logo */}
      <div className="flex items-center gap-2.5 border-b px-5 py-4">
        <Activity className="size-5 text-sidebar-primary" />
        <span className="text-sm font-semibold tracking-tight">Aspora TMS</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto px-3 py-4">
        <ul className="space-y-0.5">
          {NAV_ITEMS.map(({ href, label, icon: Icon, exact }) => (
            <li key={href}>
              <Link
                href={href}
                className={cn(
                  "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive(href, exact)
                    ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/60"
                )}
              >
                <Icon className="size-4 shrink-0" />
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer — user + WS status */}
      <div className="border-t px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="min-w-0">
            <p className="truncate text-xs font-medium">{currentUser.name}</p>
            <p className="truncate text-[10px] text-muted-foreground">{currentUser.role}</p>
          </div>
          <WsIndicator status={wsStatus} />
        </div>
      </div>
    </aside>
  )
}
