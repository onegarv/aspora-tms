"use client"

import Decimal from "decimal.js"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import type { FundMovementProposal, ProposalStatus } from "@/lib/types"
import { ArrowRight } from "lucide-react"

function statusVariant(status: ProposalStatus) {
  if (status === "PENDING_CHECKER" || status === "PENDING_MAKER")
    return "default" as const
  if (status === "APPROVED") return "secondary" as const
  if (status === "EXECUTED") return "outline" as const
  return "destructive" as const
}

function fmt(v: string) {
  return new Decimal(v).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")
}

export function ProposalCard({ proposal }: { proposal: FundMovementProposal }) {
  const isPending =
    proposal.status === "PENDING_CHECKER" || proposal.status === "PENDING_MAKER"

  return (
    <Card className={isPending ? "border-primary/40" : ""}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-muted-foreground">
              {proposal.id}
            </span>
            {proposal.requires_dual_checker && (
              <Badge variant="outline" className="text-[10px]">
                Dual Checker
              </Badge>
            )}
          </div>
          <Badge variant={statusVariant(proposal.status)} className="text-[10px]">
            {proposal.status.replace("_", " ")}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Amount + route */}
        <div>
          <p className="text-xl font-bold tabular-nums">
            {proposal.currency} ${fmt(proposal.amount_usd)}
          </p>
          <p className="flex items-center gap-1 text-xs text-muted-foreground">
            {proposal.from_account}
            <ArrowRight className="size-3" />
            {proposal.to_account}
          </p>
        </div>

        {/* Notes */}
        {proposal.notes && (
          <p className="text-xs text-muted-foreground">{proposal.notes}</p>
        )}

        {/* Metadata */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            <span>Maker: {proposal.maker}</span>
            {proposal.checker && <span> · Checker: {proposal.checker}</span>}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              {new Date(proposal.created_at).toLocaleString()}
            </span>
            <Button variant="outline" size="xs" asChild>
              <Link href={`/approvals/${proposal.id}`}>View</Link>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
