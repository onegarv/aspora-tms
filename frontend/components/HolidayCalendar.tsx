"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useHolidays } from "@/hooks/useAdmin"

const CALENDAR_COLORS: Record<string, string> = {
  IN_RBI_FX: "bg-orange-100 text-orange-700 border-orange-200",
  US_FED: "bg-blue-100 text-blue-700 border-blue-200",
  UK_BOE: "bg-purple-100 text-purple-700 border-purple-200",
  UAE_CBUAE: "bg-emerald-100 text-emerald-700 border-emerald-200",
}

export function HolidayCalendar() {
  const { data: holidays, isLoading } = useHolidays()

  if (isLoading) return <Skeleton className="h-48 rounded-xl" />

  const sorted = [...(holidays ?? [])].sort((a, b) =>
    a.date.localeCompare(b.date)
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Upcoming Bank Holidays</CardTitle>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <p className="text-sm text-muted-foreground">No upcoming holidays.</p>
        ) : (
          <div className="space-y-2">
            {sorted.map((h, i) => (
              <div
                key={i}
                className="flex items-center gap-3 rounded-md border bg-card p-2.5"
              >
                <div className="w-20 shrink-0 text-center">
                  <p className="text-xs font-semibold">
                    {new Date(h.date + "T00:00:00").toLocaleDateString("en-GB", {
                      day: "2-digit",
                      month: "short",
                    })}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {new Date(h.date + "T00:00:00").toLocaleDateString("en-GB", {
                      year: "numeric",
                    })}
                  </p>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate">{h.description}</p>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <span
                    className={`rounded border px-1.5 py-0.5 text-[10px] font-medium ${
                      CALENDAR_COLORS[h.calendar] ?? "bg-muted text-muted-foreground"
                    }`}
                  >
                    {h.calendar.replace("_", " ")}
                  </span>
                  <Badge
                    variant={h.impact === "FULL_CLOSE" ? "destructive" : "secondary"}
                    className="text-[10px]"
                  >
                    {h.impact === "FULL_CLOSE" ? "Closed" : "Partial"}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
