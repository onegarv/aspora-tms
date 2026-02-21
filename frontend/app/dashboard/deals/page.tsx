"use client"

import Decimal from "decimal.js"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import { useDeals } from "@/hooks/useDashboard"
import type { DealStatus } from "@/lib/types"

function statusVariant(status: DealStatus) {
  if (status === "OPEN") return "default" as const
  if (status === "COVERED") return "secondary" as const
  if (status === "SETTLED") return "outline" as const
  return "destructive" as const
}

function fmt(v: string, decimals = 2) {
  return new Decimal(v).toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ",")
}

export default function DealsPage() {
  const { data: deals, isLoading } = useDeals()

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">FX Deal Blotter</h1>
        <p className="text-sm text-muted-foreground">All deals — today</p>
      </div>

      {isLoading ? (
        <Skeleton className="h-64 rounded-xl" />
      ) : (
        <div className="rounded-xl border bg-card">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Deal ID</TableHead>
                <TableHead>Corridor</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">USD Amount</TableHead>
                <TableHead className="text-right">Rate</TableHead>
                <TableHead>Trader</TableHead>
                <TableHead>Counterparty</TableHead>
                <TableHead>Executed</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(deals ?? []).map((deal) => (
                <TableRow key={deal.id}>
                  <TableCell className="pl-6 font-mono text-xs">{deal.id}</TableCell>
                  <TableCell>{deal.corridor}</TableCell>
                  <TableCell>
                    <Badge
                      variant={deal.side === "BUY" ? "default" : "outline"}
                      className="text-[10px]"
                    >
                      {deal.side}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    ${fmt(deal.amount_usd, 0)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {fmt(deal.rate)}
                  </TableCell>
                  <TableCell>{deal.trader}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {deal.counterparty ?? "—"}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {new Date(deal.executed_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Badge variant={statusVariant(deal.status)} className="text-[10px]">
                      {deal.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
