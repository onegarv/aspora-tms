"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useHolidays } from "@/hooks/useAdmin"

const CALENDAR_COLORS: Record<string, string> = {
  US_FEDWIRE: "bg-blue-100 text-blue-700 border-blue-200",
  UK_CHAPS: "bg-purple-100 text-purple-700 border-purple-200",
  IN_RBI_FX: "bg-orange-100 text-orange-700 border-orange-200",
  AE_BANKING: "bg-emerald-100 text-emerald-700 border-emerald-200",
}

export function HolidayCalendar() {
  const { data: holidays, isLoading } = useHolidays()

  if (isLoading) return <Skeleton className="h-48 rounded-xl" />

  // Flatten Record<date, calendar[]> → { date, calendar }[] sorted by date
  const rows = Object.entries(holidays ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([date, calendars]) =>
      calendars.map((calendar) => ({ date, calendar }))
    )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Upcoming Bank Holidays</CardTitle>
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No upcoming holidays.</p>
        ) : (
          <div className="space-y-2">
            {rows.map(({ date, calendar }, i) => (
              <div
                key={`${date}-${calendar}-${i}`}
                className="flex items-center gap-3 rounded-md border bg-card p-2.5"
              >
                <div className="w-20 shrink-0 text-center">
                  <p className="text-xs font-semibold">
                    {new Date(date + "T00:00:00").toLocaleDateString("en-GB", {
                      day: "2-digit",
                      month: "short",
                    })}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {new Date(date + "T00:00:00").toLocaleDateString("en-GB", {
                      year: "numeric",
                    })}
                  </p>
                </div>
                <div className="flex-1" />
                <Badge
                  className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-medium ${
                    CALENDAR_COLORS[calendar] ?? "bg-muted text-muted-foreground"
                  }`}
                  variant="outline"
                >
                  {calendar.replace(/_/g, " ")}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
