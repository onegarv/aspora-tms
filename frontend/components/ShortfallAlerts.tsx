"use client"

import Decimal from "decimal.js"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useShortfalls } from "@/hooks/useDashboard"
import { AlertTriangle, AlertOctagon } from "lucide-react"
import type { RDAShortfall, ShortfallSeverity } from "@/lib/types"

function SeverityIcon({ severity }: { severity: ShortfallSeverity }) {
  if (severity === "critical")
    return <AlertOctagon className="size-4 text-destructive" />
  return <AlertTriangle className="size-4 text-yellow-500" />
}

function ShortfallRow({ s }: { s: RDAShortfall }) {
  const fmt = (v: string) =>
    new Decimal(v).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")

  return (
    <div className="flex items-start gap-3 rounded-md border bg-card p-3">
      <SeverityIcon severity={s.severity} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-semibold">{s.currency}</span>
          <Badge
            variant={s.severity === "critical" ? "destructive" : "outline"}
            className={
              s.severity === "warning"
                ? "border-yellow-500 text-yellow-600 text-[10px]"
                : "text-[10px]"
            }
          >
            {s.severity.toUpperCase()}
          </Badge>
        </div>
        <div className="mt-1 grid grid-cols-3 gap-x-4 text-xs text-muted-foreground">
          <span>Required: {fmt(s.required_amount)}</span>
          <span>Available: {fmt(s.available_balance)}</span>
          <span className="text-destructive font-medium">
            Short: {fmt(s.shortfall)}
          </span>
        </div>
      </div>
    </div>
  )
}

export function ShortfallAlerts() {
  const { data: shortfalls, isLoading } = useShortfalls()

  if (isLoading) {
    return <Skeleton className="h-24 rounded-xl" />
  }

  const list = shortfalls ?? []

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            RDA Shortfall Alerts
          </CardTitle>
          {list.length > 0 && (
            <Badge variant="destructive">{list.length}</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {list.length === 0 ? (
          <p className="text-sm text-muted-foreground">All corridors sufficiently funded.</p>
        ) : (
          <div className="space-y-2">
            {list.map((s) => (
              <ShortfallRow key={s.currency} s={s} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
