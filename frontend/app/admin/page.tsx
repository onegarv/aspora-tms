import { RiskLimitForm } from "@/components/RiskLimitForm"
import { HolidayCalendar } from "@/components/HolidayCalendar"
import { MultiplierConfig } from "@/components/MultiplierConfig"

export default function AdminPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Admin Console</h1>
        <p className="text-sm text-muted-foreground">
          Risk limits, multiplier configuration, and holiday calendar
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <RiskLimitForm />
        <div className="space-y-4">
          <MultiplierConfig />
        </div>
      </div>

      <HolidayCalendar />
    </div>
  )
}
