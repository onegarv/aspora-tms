"use client"

import Decimal from "decimal.js"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useDeals } from "@/hooks/useDashboard"
import type { DealStatus } from "@/lib/types"

function statusVariant(status: DealStatus) {
  if (status === "OPEN") return "default" as const
  if (status === "COVERED") return "secondary" as const
  if (status === "SETTLED") return "outline" as const
  return "destructive" as const
}

export function DealBlotter() {
  const { data: deals, isLoading } = useDeals()

  if (isLoading) {
    return <Skeleton className="h-64 rounded-xl" />
  }

  // Show most recent 8 deals
  const rows = (deals ?? []).slice(0, 8)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-muted-foreground">
          Deal Blotter
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-6 text-xs">ID</TableHead>
              <TableHead className="text-xs">Corridor</TableHead>
              <TableHead className="text-xs">Side</TableHead>
              <TableHead className="text-right text-xs">USD Amt</TableHead>
              <TableHead className="text-right text-xs">Rate</TableHead>
              <TableHead className="text-xs">Trader</TableHead>
              <TableHead className="text-xs">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((deal) => (
              <TableRow key={deal.id}>
                <TableCell className="pl-6 font-mono text-xs">{deal.id}</TableCell>
                <TableCell className="text-xs">{deal.corridor}</TableCell>
                <TableCell>
                  <Badge
                    variant={deal.side === "BUY" ? "default" : "outline"}
                    className="text-[10px]"
                  >
                    {deal.side}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-mono text-xs tabular-nums">
                  $
                  {new Decimal(deal.amount_usd)
                    .toFixed(0)
                    .replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                </TableCell>
                <TableCell className="text-right font-mono text-xs tabular-nums">
                  {new Decimal(deal.rate).toFixed(2)}
                </TableCell>
                <TableCell className="text-xs">{deal.trader}</TableCell>
                <TableCell>
                  <Badge variant={statusVariant(deal.status)} className="text-[10px]">
                    {deal.status}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
