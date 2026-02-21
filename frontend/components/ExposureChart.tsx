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

const COLORS = ["#3b82f6", "#10b981", "#ef4444"]

function toCr(val: string | null | undefined): number {
  if (!val) return 0
  return new Decimal(val).div(10_000_000).toDecimalPlaces(2).toNumber()
}

export function ExposureChart() {
  const { data: exposure, isLoading } = useExposure()

  if (isLoading) {
    return <Skeleton className="h-64 rounded-xl" />
  }

  if (!exposure) return null

  const chartData = [
    { label: "Total", value: toCr(exposure.total_inr_required) },
    { label: "Covered", value: toCr(exposure.covered_inr) },
    { label: "Open", value: toCr(exposure.open_inr) },
  ]

  const openPct =
    exposure.total_inr_required && exposure.open_inr
      ? new Decimal(exposure.open_inr)
          .div(new Decimal(exposure.total_inr_required).gt(0) ? exposure.total_inr_required : 1)
          .mul(100)
          .toFixed(1)
      : "0.0"

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            FX Exposure (INR)
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            {openPct}% open
            {exposure.deal_count != null && ` · ${exposure.deal_count} deals`}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {/* Summary row */}
        <div className="mb-4 grid grid-cols-3 gap-3 text-center">
          {chartData.map(({ label, value }) => (
            <div key={label} className="rounded-md bg-muted p-2">
              <p className="text-[10px] text-muted-foreground">{label}</p>
              <p className="text-sm font-bold tabular-nums">₹{value}Cr</p>
            </div>
          ))}
        </div>

        {exposure.blended_rate && (
          <p className="mb-3 text-center text-xs text-muted-foreground">
            Blended rate: <span className="font-semibold">{new Decimal(exposure.blended_rate).toFixed(4)}</span>
          </p>
        )}

        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={chartData} barCategoryGap="40%">
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tickFormatter={(v: number) => `₹${v}Cr`}
              tick={{ fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={52}
            />
            <Tooltip
              formatter={(value: number | undefined) => [
                value !== undefined ? `₹${value}Cr` : "—",
                "Exposure",
              ]}
              cursor={{ fill: "hsl(var(--muted))" }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
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
