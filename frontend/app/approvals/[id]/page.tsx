"use client"

import { useState } from "react"
import { use } from "react"
import Decimal from "decimal.js"
import { ArrowLeft, ArrowRight } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { AuditTrail } from "@/components/AuditTrail"
import { ApprovalModal } from "@/components/ApprovalModal"
import { useProposal } from "@/hooks/useProposals"
import type { ProposalStatus } from "@/lib/types"

function statusVariant(status: ProposalStatus) {
  if (status === "PENDING_CHECKER" || status === "PENDING_MAKER")
    return "default" as const
  if (status === "APPROVED") return "secondary" as const
  if (status === "EXECUTED") return "outline" as const
  return "destructive" as const
}

export default function ProposalDetailPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = use(params)
  const { data: proposal, isLoading } = useProposal(id)
  const [modalAction, setModalAction] = useState<"approve" | "reject" | null>(null)

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    )
  }

  if (!proposal) {
    return <p className="text-sm text-muted-foreground">Proposal not found.</p>
  }

  const fmt = (v: string) =>
    new Decimal(v).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")

  const isPending =
    proposal.status === "PENDING_CHECKER" || proposal.status === "PENDING_MAKER"

  return (
    <div className="space-y-6">
      {/* Back */}
      <Button variant="ghost" size="sm" asChild className="-ml-2">
        <Link href="/approvals">
          <ArrowLeft className="mr-1 size-4" />
          Back to Approvals
        </Link>
      </Button>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold font-mono">{proposal.id}</h1>
          <p className="text-sm text-muted-foreground">{proposal.type}</p>
        </div>
        <Badge variant={statusVariant(proposal.status)}>
          {proposal.status.replace(/_/g, " ")}
        </Badge>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Details card */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Transfer Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Amount</span>
              <span className="font-bold tabular-nums">
                {proposal.currency} ${fmt(proposal.amount_usd)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Route</span>
              <span className="flex items-center gap-1 font-mono text-xs">
                {proposal.from_account}
                <ArrowRight className="size-3" />
                {proposal.to_account}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Maker</span>
              <span>{proposal.maker}</span>
            </div>
            {proposal.checker && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Checker</span>
                <span>{proposal.checker}</span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Dual Checker</span>
              <span>{proposal.requires_dual_checker ? "Yes" : "No"}</span>
            </div>
            {proposal.notes && (
              <div>
                <span className="text-muted-foreground block mb-1">Notes</span>
                <p className="text-xs italic">&quot;{proposal.notes}&quot;</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Audit Trail */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Audit Trail</CardTitle>
          </CardHeader>
          <CardContent>
            <AuditTrail entries={proposal.audit_trail} />
          </CardContent>
        </Card>
      </div>

      {/* Action buttons */}
      {isPending && (
        <div className="flex gap-3">
          <Button onClick={() => setModalAction("approve")}>
            Approve
          </Button>
          <Button variant="destructive" onClick={() => setModalAction("reject")}>
            Reject
          </Button>
        </div>
      )}

      <ApprovalModal
        proposal={proposal}
        action={modalAction}
        onClose={() => setModalAction(null)}
      />
    </div>
  )
}
