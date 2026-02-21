"use client"

import Decimal from "decimal.js"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { ExposureChart } from "@/components/ExposureChart"
import { useExposure } from "@/hooks/useDashboard"

const MAX_OPEN_PCT = new Decimal("0.30")

function toCr(val: string | null | undefined): Decimal {
  if (!val) return new Decimal(0)
  return new Decimal(val).div(10_000_000)
}

export default function ExposurePage() {
  const { data: exposure, isLoading } = useExposure()

  const total = toCr(exposure?.total_inr_required)
  const covered = toCr(exposure?.covered_inr)
  const open = toCr(exposure?.open_inr)
  const openPct = total.gt(0) ? open.div(total) : new Decimal(0)
  const isOverLimit = openPct.gt(MAX_OPEN_PCT)

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
                      ₹{total.toFixed(2)}Cr
                    </p>
                    {exposure.deal_count != null && (
                      <p className="text-xs text-muted-foreground">
                        {exposure.deal_count} deals
                      </p>
                    )}
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
                      ₹{covered.toFixed(2)}Cr
                    </p>
                    {exposure.blended_rate && (
                      <p className="text-xs text-muted-foreground">
                        @ {new Decimal(exposure.blended_rate).toFixed(4)}
                      </p>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-1">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xs text-muted-foreground">
                        Open (Uncovered)
                      </CardTitle>
                      {isOverLimit && (
                        <span className="text-[10px] font-medium text-destructive">
                          OVER LIMIT
                        </span>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p
                      className={`text-2xl font-bold tabular-nums ${
                        isOverLimit ? "text-destructive" : ""
                      }`}
                    >
                      ₹{open.toFixed(2)}Cr
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {openPct.mul(100).toFixed(1)}% of total
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
