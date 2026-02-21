"use client"

import Decimal from "decimal.js"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useBalances } from "@/hooks/useDashboard"
import type { NostroBalance } from "@/lib/types"

function formatAmount(value: string, currency: string) {
  const d = new Decimal(value)
  return `${currency} ${d.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}`
}

function healthBadge(balance: NostroBalance) {
  if (!balance.threshold_critical || !balance.threshold_warning) return null
  const bal = new Decimal(balance.balance)
  const crit = new Decimal(balance.threshold_critical)
  const warn = new Decimal(balance.threshold_warning)
  if (bal.lt(crit)) return <Badge variant="destructive">Critical</Badge>
  if (bal.lt(warn)) return <Badge variant="outline" className="border-yellow-500 text-yellow-600">Low</Badge>
  return <Badge variant="secondary" className="text-green-700">Healthy</Badge>
}

function BalanceCard({ balance }: { balance: NostroBalance }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            {balance.currency} Nostro
          </CardTitle>
          {healthBadge(balance)}
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-xl font-bold tabular-nums">
          {formatAmount(balance.balance, balance.currency)}
        </p>
        <p className="mt-1 text-[10px] text-muted-foreground">
          Updated {new Date(balance.last_updated).toLocaleTimeString()}
        </p>
      </CardContent>
    </Card>
  )
}

export function BalanceGrid() {
  const { data: balances, isLoading } = useBalances()

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-28 rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {(balances ?? []).map((b) => (
        <BalanceCard key={b.currency} balance={b} />
      ))}
    </div>
  )
}
