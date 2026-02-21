import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface MultiplierRule {
  name: string
  multiplier: string
  condition: string
  status: "active" | "inactive"
}

const MULTIPLIER_RULES: MultiplierRule[] = [
  {
    name: "Payday",
    multiplier: "1.4×",
    condition: "Day 25–last of month, or day 1–3 of next month",
    status: "active",
  },
  {
    name: "Pre-holiday",
    multiplier: "1.2×",
    condition: "Day before IN_RBI_FX holiday",
    status: "inactive",
  },
  {
    name: "Post-holiday",
    multiplier: "0.6×",
    condition: "Day after IN_RBI_FX holiday",
    status: "inactive",
  },
]

export function MultiplierConfig() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Multiplier Engine</CardTitle>
          <span className="text-xs text-muted-foreground">
            Cap: 2.5× (system max)
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {MULTIPLIER_RULES.map((rule) => (
          <div
            key={rule.name}
            className="flex items-center gap-3 rounded-md border bg-card p-3"
          >
            <Badge
              variant={rule.status === "active" ? "default" : "secondary"}
              className="w-16 justify-center text-[10px]"
            >
              {rule.status === "active" ? "ACTIVE" : "INACTIVE"}
            </Badge>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">{rule.name}</p>
              <p className="text-xs text-muted-foreground">{rule.condition}</p>
            </div>
            <span className="shrink-0 font-mono text-sm font-bold">
              {rule.multiplier}
            </span>
          </div>
        ))}
        <p className="pt-1 text-xs text-muted-foreground">
          Multipliers are read-only — managed by the LiquidityAgent configuration.
          Modify via <code className="font-mono">ASPORA_*</code> environment variables.
        </p>
      </CardContent>
    </Card>
  )
}
