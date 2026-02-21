"use client"

import { useState } from "react"
import { use } from "react"
import Decimal from "decimal.js"
import { ArrowLeft, ArrowRight, Zap } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { AuditTrail } from "@/components/AuditTrail"
import { ApprovalModal } from "@/components/ApprovalModal"
import { DispatchModal } from "@/components/DispatchModal"
import { useProposal } from "@/hooks/useProposals"
import type { ProposalStatus } from "@/lib/types"

function statusVariant(status: ProposalStatus) {
  if (status === "PENDING_CHECKER" || status === "PENDING_MAKER")
    return "default" as const
  if (status === "APPROVED" || status === "approved") return "secondary" as const
  if (status === "EXECUTED" || status === "executed") return "outline" as const
  if (status === "DISPATCHED" || status === "dispatched")
    return "secondary" as const
  return "destructive" as const
}

export default function ProposalDetailPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = use(params)
  const { data: proposal, isLoading } = useProposal(id)
  const [modalAction, setModalAction]   = useState<"approve" | "reject" | null>(null)
  const [showDispatch, setShowDispatch] = useState(false)

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
    proposal.status === "PENDING_CHECKER" || proposal.status === "PENDING_MAKER" ||
    proposal.status === "pending_approval"

  const isApproved =
    proposal.status === "APPROVED" || proposal.status === "approved"

  const isDispatched =
    proposal.status === "DISPATCHED" || proposal.status === "dispatched"

  const isGbp = proposal.currency === "GBP"

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
            {(isDispatched || proposal.settlement_ref) && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">ClearBank Payment ID</span>
                <span className="font-mono text-xs text-green-600 dark:text-green-400">
                  {proposal.settlement_ref}
                </span>
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
      <div className="flex flex-wrap gap-3">
        {isPending && (
          <>
            <Button onClick={() => setModalAction("approve")}>Approve</Button>
            <Button variant="destructive" onClick={() => setModalAction("reject")}>
              Reject
            </Button>
          </>
        )}

        {isApproved && isGbp && (
          <Button
            variant="outline"
            className="border-orange-400 text-orange-600 hover:bg-orange-50 dark:hover:bg-orange-950"
            onClick={() => setShowDispatch(true)}
          >
            <Zap className="mr-1.5 size-4" />
            Dispatch to ClearBank
          </Button>
        )}

        {isApproved && !isGbp && (
          <p className="text-xs text-muted-foreground self-center">
            ClearBank dispatch is only available for GBP proposals.
          </p>
        )}

        {isDispatched && (
          <Badge className="self-center bg-green-600 text-white text-xs px-3 py-1">
            Dispatched to ClearBank
          </Badge>
        )}
      </div>

      <ApprovalModal
        proposal={proposal}
        action={modalAction}
        onClose={() => setModalAction(null)}
      />

      <DispatchModal
        proposal={showDispatch ? proposal : null}
        onClose={() => setShowDispatch(false)}
      />
    </div>
  )
}
