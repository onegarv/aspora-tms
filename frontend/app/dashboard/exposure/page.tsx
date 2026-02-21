"use client"

import Decimal from "decimal.js"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { ExposureChart } from "@/components/ExposureChart"
import { useExposure } from "@/hooks/useDashboard"

export default function ExposurePage() {
  const { data: exposure, isLoading } = useExposure()

  const maxOpenPct = new Decimal("0.30") // from settings: max_open_exposure_pct

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">FX Exposure</h1>
        <p className="text-sm text-muted-foreground">
          Open position vs covered — max allowed 30%
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-3 gap-4">
          {[...Array(3)].map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-4">
            {exposure && (
              <>
                <Card>
                  <CardHeader className="pb-1">
                    <CardTitle className="text-xs text-muted-foreground">
                      Total Exposure
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold tabular-nums">
                      $
                      {new Decimal(exposure.total_exposure_usd)
                        .div(1_000_000)
                        .toFixed(1)}
                      M
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-1">
                    <CardTitle className="text-xs text-muted-foreground">
                      Covered
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold tabular-nums text-green-600">
                      $
                      {new Decimal(exposure.covered_usd)
                        .div(1_000_000)
                        .toFixed(1)}
                      M
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-1">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xs text-muted-foreground">
                        Open (Uncovered)
                      </CardTitle>
                      {new Decimal(exposure.open_pct).gt(maxOpenPct) && (
                        <span className="text-[10px] text-destructive font-medium">
                          OVER LIMIT
                        </span>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p
                      className={`text-2xl font-bold tabular-nums ${
                        new Decimal(exposure.open_pct).gt(maxOpenPct)
                          ? "text-destructive"
                          : ""
                      }`}
                    >
                      $
                      {new Decimal(exposure.open_usd)
                        .div(1_000_000)
                        .toFixed(1)}
                      M
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Decimal(exposure.open_pct).mul(100).toFixed(1)}% of
                      total
                    </p>
                  </CardContent>
                </Card>
              </>
            )}
          </div>

          <ExposureChart />
        </>
      )}
    </div>
  )
}
