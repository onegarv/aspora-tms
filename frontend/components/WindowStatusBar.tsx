"use client"

import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useWindows } from "@/hooks/useDashboard"
import type { BankingWindow, WindowStatus } from "@/lib/types"
import { Clock } from "lucide-react"

function statusVariant(status: WindowStatus) {
  if (status === "OPEN") return "default" as const
  if (status === "CLOSING") return "destructive" as const
  return "secondary" as const
}

function WindowChip({ window: w }: { window: BankingWindow }) {
  return (
    <div className="flex items-center gap-2 rounded-md border bg-card px-3 py-2">
      <div className="flex flex-col">
        <span className="text-xs font-semibold">
          {w.currency} / {w.system}
        </span>
        {w.status === "CLOSING" && w.minutes_until_close !== undefined && (
          <span className="flex items-center gap-1 text-[10px] text-destructive">
            <Clock className="size-3" />
            {w.minutes_until_close}m left
          </span>
        )}
      </div>
      <Badge variant={statusVariant(w.status)} className="text-[10px]">
        {w.status}
      </Badge>
    </div>
  )
}

export function WindowStatusBar() {
  const { data: windows, isLoading } = useWindows()

  if (isLoading) {
    return (
      <div className="flex gap-3">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-14 w-36 rounded-md" />
        ))}
      </div>
    )
  }

  return (
    <div className="flex flex-wrap gap-3">
      {(windows ?? []).map((w) => (
        <WindowChip key={w.id} window={w} />
      ))}
    </div>
  )
}
