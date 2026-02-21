"use client"

import Decimal from "decimal.js"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useExposure } from "@/hooks/useDashboard"

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

export function ExposureChart() {
  const { data: exposure, isLoading } = useExposure()

  if (isLoading) {
    return <Skeleton className="h-64 rounded-xl" />
  }

  if (!exposure) return null

  const chartData = Object.entries(exposure.by_currency)
    .filter(([, v]) => new Decimal(v).gt(0))
    .map(([currency, value]) => ({
      currency,
      usd_m: new Decimal(value).div(1_000_000).toDecimalPlaces(2).toNumber(),
    }))

  const openPct = new Decimal(exposure.open_pct).mul(100).toFixed(1)

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            FX Exposure by Currency
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            {openPct}% open
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {/* Summary row */}
        <div className="mb-4 grid grid-cols-3 gap-3 text-center">
          {[
            { label: "Total", value: exposure.total_exposure_usd },
            { label: "Covered", value: exposure.covered_usd },
            { label: "Open", value: exposure.open_usd },
          ].map(({ label, value }) => (
            <div key={label} className="rounded-md bg-muted p-2">
              <p className="text-[10px] text-muted-foreground">{label}</p>
              <p className="text-sm font-bold tabular-nums">
                ${new Decimal(value).div(1_000_000).toFixed(1)}M
              </p>
            </div>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={chartData} barCategoryGap="40%">
            <XAxis
              dataKey="currency"
              tick={{ fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tickFormatter={(v: number) => `$${v}M`}
              tick={{ fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={40}
            />
            <Tooltip
              formatter={(value: number | undefined) => [
                value !== undefined ? `$${value}M` : "—",
                "Exposure",
              ]}
              cursor={{ fill: "hsl(var(--muted))" }}
            />
            <Bar dataKey="usd_m" radius={[4, 4, 0, 0]}>
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
