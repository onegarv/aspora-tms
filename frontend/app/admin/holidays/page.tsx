import { HolidayCalendar } from "@/components/HolidayCalendar"

export default function HolidaysPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Bank Holiday Calendar</h1>
        <p className="text-sm text-muted-foreground">
          Upcoming holidays affecting FX settlement windows
        </p>
      </div>
      <HolidayCalendar />
    </div>
  )
}
