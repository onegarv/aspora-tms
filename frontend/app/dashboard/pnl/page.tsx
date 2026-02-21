"use client"

import Decimal from "decimal.js"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { PnLChart } from "@/components/PnLChart"
import { usePnL } from "@/hooks/useDashboard"

function PnLRow({ label, value }: { label: string; value: string }) {
  const d = new Decimal(value)
  const positive = d.gte(0)
  return (
    <div className="flex items-center justify-between py-1.5 border-b last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span
        className={`font-mono tabular-nums font-medium text-sm ${
          positive ? "text-green-600" : "text-destructive"
        }`}
      >
        {positive ? "+" : ""}$
        {d.abs().toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
      </span>
    </div>
  )
}

export default function PnLPage() {
  const { data: pnl, isLoading } = usePnL()

  const totalRealised = (pnl ?? []).reduce(
    (s, e) => s.add(new Decimal(e.realised)),
    new Decimal(0)
  )
  const totalUnrealised = (pnl ?? []).reduce(
    (s, e) => s.add(new Decimal(e.unrealised)),
    new Decimal(0)
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Profit &amp; Loss</h1>
        <p className="text-sm text-muted-foreground">Currency-level P&amp;L — today</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {!isLoading &&
          (pnl ?? []).map((entry) => (
            <Card key={entry.currency}>
              <CardHeader className="pb-1">
                <CardTitle className="text-xs text-muted-foreground">
                  {entry.currency}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-1">
                <PnLRow label="Realised" value={entry.realised} />
                <PnLRow label="Unrealised" value={entry.unrealised} />
                <PnLRow label="Total" value={entry.total} />
              </CardContent>
            </Card>
          ))}
        {isLoading &&
          [...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-40 rounded-xl" />
          ))}
      </div>

      {/* Totals */}
      {!isLoading && (
        <div className="flex gap-6">
          <div className="rounded-xl border bg-card px-5 py-3">
            <p className="text-xs text-muted-foreground">Total Realised</p>
            <p className="text-2xl font-bold tabular-nums text-green-600">
              +$
              {totalRealised.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
            </p>
          </div>
          <div className="rounded-xl border bg-card px-5 py-3">
            <p className="text-xs text-muted-foreground">Total Unrealised</p>
            <p
              className={`text-2xl font-bold tabular-nums ${
                totalUnrealised.gte(0) ? "text-green-600" : "text-destructive"
              }`}
            >
              {totalUnrealised.gte(0) ? "+" : ""}$
              {totalUnrealised
                .abs()
                .toFixed(2)
                .replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
            </p>
          </div>
        </div>
      )}

      {/* Chart */}
      <PnLChart />
    </div>
  )
}
