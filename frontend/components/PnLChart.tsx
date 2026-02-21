"use client"

import Decimal from "decimal.js"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { usePnL } from "@/hooks/useDashboard"

export function PnLChart() {
  const { data: pnl, isLoading } = usePnL()

  if (isLoading) {
    return <Skeleton className="h-64 rounded-xl" />
  }

  const chartData = (pnl ?? []).map((entry) => ({
    currency: entry.currency,
    Realised: new Decimal(entry.realised).toDecimalPlaces(0).toNumber(),
    Unrealised: new Decimal(entry.unrealised).toDecimalPlaces(0).toNumber(),
  }))

  const totalRealised = (pnl ?? []).reduce(
    (sum, e) => sum.add(new Decimal(e.realised)),
    new Decimal(0)
  )
  const totalUnrealised = (pnl ?? []).reduce(
    (sum, e) => sum.add(new Decimal(e.unrealised)),
    new Decimal(0)
  )

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            P&amp;L Summary
          </CardTitle>
          <div className="text-right text-xs text-muted-foreground">
            <span className="mr-3">
              Realised:{" "}
              <span className="font-medium text-green-600">
                ${totalRealised.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
              </span>
            </span>
            <span>
              Unrealised:{" "}
              <span
                className={
                  totalUnrealised.gte(0)
                    ? "font-medium text-green-600"
                    : "font-medium text-destructive"
                }
              >
                ${totalUnrealised.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
              </span>
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={chartData} barCategoryGap="35%">
            <XAxis
              dataKey="currency"
              tick={{ fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
              tick={{ fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={42}
            />
            <Tooltip
              formatter={(value: number | undefined, name: string | undefined) => [
                value !== undefined ? `$${value.toLocaleString()}` : "—",
                name ?? "",
              ]}
              cursor={{ fill: "hsl(var(--muted))" }}
            />
            <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="Realised" fill="#10b981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Unrealised" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
