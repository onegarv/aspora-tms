import { WindowStatusBar } from "@/components/WindowStatusBar"
import { BalanceGrid } from "@/components/BalanceGrid"
import { ForecastCard } from "@/components/ForecastCard"
import { ShortfallAlerts } from "@/components/ShortfallAlerts"
import { ExposureChart } from "@/components/ExposureChart"
import { DealBlotter } from "@/components/DealBlotter"
import { PnLChart } from "@/components/PnLChart"
import { LiveEventFeed } from "@/components/LiveEventFeed"
import { AgentSeeder } from "@/components/AgentSeeder"

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <AgentSeeder />
      <div>
        <h1 className="text-xl font-bold">Treasury Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Live overview — {new Date().toLocaleDateString("en-GB", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
        </p>
      </div>

      {/* Banking Window Status */}
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Banking Windows
        </h2>
        <WindowStatusBar />
      </section>

      {/* Nostro Balances */}
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Nostro Balances
        </h2>
        <BalanceGrid />
      </section>

      {/* Forecast + Shortfalls */}
      <div className="grid gap-4 lg:grid-cols-2">
        <ForecastCard />
        <ShortfallAlerts />
      </div>

      {/* Exposure + P&L */}
      <div className="grid gap-4 lg:grid-cols-2">
        <ExposureChart />
        <PnLChart />
      </div>

      {/* Deal Blotter + Live Feed */}
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <DealBlotter />
        </div>
        <LiveEventFeed />
      </div>
    </div>
  )
}
