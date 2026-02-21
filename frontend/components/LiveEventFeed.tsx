"use client"

import { useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useTmsStore } from "@/store"
import { MOCK_LIVE_EVENTS } from "@/lib/mock"
import { Trash2 } from "lucide-react"

const EVENT_LABEL: Record<string, string> = {
  "forecast.daily.ready": "Forecast",
  "forecast.rda.shortfall": "Shortfall",
  "ops.nostro.balance.update": "Balance",
  "maker_checker.proposal.approved": "Approved",
  "ops.window.closing": "Window",
  "ops.fund.movement.request": "Transfer",
  "ops.transfer.confirmed": "Confirmed",
  "fx.deal.instruction": "Deal",
  "fx.exposure.update": "Exposure",
}

function eventBadgeVariant(type: string) {
  if (type.includes("shortfall") || type.includes("closing")) return "destructive" as const
  if (type.includes("approved") || type.includes("confirmed")) return "default" as const
  return "secondary" as const
}

export function LiveEventFeed() {
  const { liveEvents, appendEvent, clearEvents } = useTmsStore()

  // Seed with mock events on mount in mock mode
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_USE_MOCK !== "true") return
    if (liveEvents.length > 0) return
    // Replay mock events in reverse order so newest is first
    ;[...MOCK_LIVE_EVENTS].reverse().forEach((e) => appendEvent(e))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-muted-foreground">
            Live Event Feed
          </CardTitle>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={clearEvents}
            title="Clear events"
          >
            <Trash2 className="size-3" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto">
        {liveEvents.length === 0 ? (
          <p className="text-xs text-muted-foreground">No events yet.</p>
        ) : (
          <ul className="space-y-1.5">
            {liveEvents.map((evt) => (
              <li key={evt.id} className="flex items-start gap-2 text-xs">
                <Badge
                  variant={eventBadgeVariant(evt.type)}
                  className="mt-0.5 shrink-0 text-[10px]"
                >
                  {EVENT_LABEL[evt.type] ?? evt.type.split(".").pop()}
                </Badge>
                <span className="flex-1 truncate font-mono text-[11px] text-muted-foreground">
                  {evt.type}
                </span>
                <span className="shrink-0 text-[10px] text-muted-foreground">
                  {new Date(evt.timestamp).toLocaleTimeString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
