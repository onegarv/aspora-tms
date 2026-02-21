import type { ProposalAuditEntry } from "@/lib/types"
import { CheckCircle, Clock, XCircle, Zap, FileText } from "lucide-react"

const ACTION_ICON: Record<string, React.ReactNode> = {
  CREATED: <FileText className="size-3.5 text-muted-foreground" />,
  SUBMITTED_FOR_APPROVAL: <Clock className="size-3.5 text-blue-500" />,
  APPROVED: <CheckCircle className="size-3.5 text-green-500" />,
  REJECTED: <XCircle className="size-3.5 text-destructive" />,
  EXECUTED: <Zap className="size-3.5 text-purple-500" />,
}

export function AuditTrail({ entries }: { entries: ProposalAuditEntry[] }) {
  return (
    <ol className="relative ml-2 border-l border-border">
      {entries.map((entry, i) => (
        <li key={i} className="mb-4 ml-5">
          <span className="absolute -left-[9px] flex size-4 items-center justify-center rounded-full bg-background ring-2 ring-border">
            {ACTION_ICON[entry.action] ?? (
              <span className="size-1.5 rounded-full bg-muted-foreground" />
            )}
          </span>
          <div className="rounded-md border bg-card px-3 py-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold">
                {entry.action.replace(/_/g, " ")}
              </span>
              <span className="text-[10px] text-muted-foreground">
                {new Date(entry.timestamp).toLocaleString()}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">by {entry.actor}</p>
            {entry.notes && (
              <p className="mt-1 text-xs italic text-muted-foreground">
                &quot;{entry.notes}&quot;
              </p>
            )}
          </div>
        </li>
      ))}
    </ol>
  )
}
