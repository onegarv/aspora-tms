import { RiskLimitForm } from "@/components/RiskLimitForm"

export default function RiskLimitsPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Risk Limits</h1>
        <p className="text-sm text-muted-foreground">
          Treasury risk parameters — changes &gt; 20% require maker-checker approval
        </p>
      </div>
      <RiskLimitForm />
    </div>
  )
}
