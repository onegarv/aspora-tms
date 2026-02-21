"use client"

import Decimal from "decimal.js"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useForecast } from "@/hooks/useDashboard"
import type { ForecastConfidence } from "@/lib/types"

function confidenceVariant(c: ForecastConfidence) {
  if (c === "high") return "default" as const
  if (c === "medium") return "secondary" as const
  return "outline" as const
}

export function ForecastCard() {
  const { data: forecast, isLoading } = useForecast()

  if (isLoading) {
    return <Skeleton className="h-44 rounded-xl" />
  }

  if (!forecast) return null

  const multiplierKeys = Object.keys(forecast.multipliers_applied)

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            Daily Liquidity Forecast
          </CardTitle>
          <Badge variant={confidenceVariant(forecast.confidence)} className="capitalize">
            {forecast.confidence} confidence
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <p className="text-3xl font-bold tabular-nums">
            ₹{new Decimal(forecast.total_inr_crores).toFixed(2)} Cr
          </p>
          <p className="text-xs text-muted-foreground">
            {forecast.forecast_date}
          </p>
        </div>

        {/* Currency split */}
        <div className="grid grid-cols-3 gap-2">
          {Object.entries(forecast.currency_split).map(([ccy, val]) => (
            <div key={ccy} className="rounded-md bg-muted px-2 py-1.5 text-center">
              <p className="text-[10px] text-muted-foreground">{ccy}</p>
              <p className="text-xs font-semibold tabular-nums">
                ₹{new Decimal(val).toFixed(1)} Cr
              </p>
            </div>
          ))}
        </div>

        {/* Active multipliers */}
        {multiplierKeys.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {multiplierKeys.map((k) => (
              <Badge key={k} variant="outline" className="text-[10px] capitalize">
                {k} {new Decimal(forecast.multipliers_applied[k]).toFixed(1)}×
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
