"use client"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

const STATUS_OPTIONS = [
  { value: "all", label: "All statuses" },
  { value: "PENDING_CHECKER", label: "Pending Checker" },
  { value: "PENDING_MAKER", label: "Pending Maker" },
  { value: "APPROVED", label: "Approved" },
  { value: "REJECTED", label: "Rejected" },
  { value: "EXECUTED", label: "Executed" },
]

interface Props {
  value: string
  onChange: (value: string) => void
}

export function ProposalFilters({ value, onChange }: Props) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-muted-foreground">Filter:</label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-44">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {STATUS_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
