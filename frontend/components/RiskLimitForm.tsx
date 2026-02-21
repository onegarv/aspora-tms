"use client"

import { useState } from "react"
import Decimal from "decimal.js"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRiskLimits, useUpdateRiskLimit } from "@/hooks/useAdmin"
import { Skeleton } from "@/components/ui/skeleton"

interface FieldDef {
  key: keyof ReturnType<typeof buildFields>
  label: string
  hint: string
  prefix?: string
  suffix?: string
}

function buildFields(limits: {
  max_single_deal_usd: string
  max_open_exposure_pct: string
  stop_loss_paise: string
  dual_checker_threshold_usd: string
  prefunding_buffer_pct: string
}) {
  return {
    max_single_deal_usd: limits.max_single_deal_usd,
    max_open_exposure_pct: limits.max_open_exposure_pct,
    stop_loss_paise: limits.stop_loss_paise,
    dual_checker_threshold_usd: limits.dual_checker_threshold_usd,
    prefunding_buffer_pct: limits.prefunding_buffer_pct,
  }
}

const FIELD_DEFS: FieldDef[] = [
  {
    key: "max_single_deal_usd",
    label: "Max Single Deal",
    hint: "Maximum USD value for a single FX deal",
    prefix: "$",
  },
  {
    key: "max_open_exposure_pct",
    label: "Max Open Exposure",
    hint: "Maximum open (uncovered) exposure as % of total",
    suffix: "%",
  },
  {
    key: "stop_loss_paise",
    label: "Stop-Loss",
    hint: "INR rate stop-loss threshold in paise",
    suffix: "paise",
  },
  {
    key: "dual_checker_threshold_usd",
    label: "Dual Checker Threshold",
    hint: "Proposals above this USD amount require two checkers",
    prefix: "$",
  },
  {
    key: "prefunding_buffer_pct",
    label: "Prefunding Buffer",
    hint: "Safety buffer added to RDA required amounts",
    suffix: "%",
  },
]

export function RiskLimitForm() {
  const { data: limits, isLoading } = useRiskLimits()
  const updateMutation = useUpdateRiskLimit()
  const [draft, setDraft] = useState<Record<string, string> | null>(null)

  if (isLoading) return <Skeleton className="h-72 rounded-xl" />
  if (!limits) return null

  const current = draft ?? buildFields(limits)

  function displayValue(key: string, raw: string) {
    if (key === "max_open_exposure_pct" || key === "prefunding_buffer_pct") {
      return new Decimal(raw).mul(100).toFixed(1)
    }
    return raw
  }

  function internalValue(key: string, display: string) {
    if (key === "max_open_exposure_pct" || key === "prefunding_buffer_pct") {
      return new Decimal(display).div(100).toFixed(4)
    }
    return display
  }

  function handleChange(key: string, display: string) {
    setDraft((prev) => ({
      ...(prev ?? buildFields(limits!)),
      [key]: internalValue(key, display),
    }))
  }

  function checkOver20Pct(key: string, newVal: string) {
    const original = buildFields(limits!)[key as keyof ReturnType<typeof buildFields>]
    const origD = new Decimal(original)
    if (origD.eq(0)) return false
    return new Decimal(newVal)
      .sub(origD)
      .abs()
      .div(origD)
      .gt(new Decimal("0.20"))
  }

  async function handleSave() {
    if (!draft) return
    // Check if any field changed by >20% — those must go through maker-checker
    const oversized = Object.keys(draft).find((k) => checkOver20Pct(k, draft[k]))
    if (oversized) {
      alert(
        `Risk limit change for "${oversized}" exceeds 20% — must go through maker-checker approval.`
      )
      return
    }
    await updateMutation.mutateAsync(draft)
    setDraft(null)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Risk Limits</CardTitle>
          <div className="text-xs text-muted-foreground">
            Last updated by {limits.updated_by} ·{" "}
            {new Date(limits.updated_at).toLocaleDateString()}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {FIELD_DEFS.map(({ key, label, hint, prefix, suffix }) => (
          <div key={key} className="flex items-center gap-4">
            <div className="w-56">
              <p className="text-sm font-medium">{label}</p>
              <p className="text-xs text-muted-foreground">{hint}</p>
            </div>
            <div className="flex items-center gap-1.5">
              {prefix && (
                <span className="text-sm text-muted-foreground">{prefix}</span>
              )}
              <input
                type="number"
                step="any"
                className="w-36 rounded-md border bg-background px-3 py-1.5 text-sm tabular-nums outline-none focus:ring-2 focus:ring-ring"
                value={displayValue(key, current[key])}
                onChange={(e) => handleChange(key, e.target.value)}
              />
              {suffix && (
                <span className="text-sm text-muted-foreground">{suffix}</span>
              )}
            </div>
          </div>
        ))}

        {draft && (
          <div className="flex gap-2 pt-2">
            <Button
              onClick={handleSave}
              disabled={updateMutation.isPending}
            >
              {updateMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
            <Button variant="outline" onClick={() => setDraft(null)}>
              Cancel
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
