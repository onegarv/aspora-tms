"use client"

import { useState } from "react"
import { ProposalCard } from "@/components/ProposalCard"
import { ProposalFilters } from "@/components/ProposalFilters"
import { ApprovalModal } from "@/components/ApprovalModal"
import { Skeleton } from "@/components/ui/skeleton"
import { useProposals } from "@/hooks/useProposals"
import type { FundMovementProposal } from "@/lib/types"

export default function ApprovalsPage() {
  const [statusFilter, setStatusFilter] = useState("all")
  const [selectedProposal, setSelectedProposal] =
    useState<FundMovementProposal | null>(null)
  const [modalAction, setModalAction] = useState<"approve" | "reject" | null>(null)

  const { data: proposals, isLoading } = useProposals(
    statusFilter === "all" ? undefined : statusFilter
  )

  const filtered =
    statusFilter === "all"
      ? (proposals ?? [])
      : (proposals ?? []).filter((p) => p.status === statusFilter)

  function openModal(p: FundMovementProposal, action: "approve" | "reject") {
    setSelectedProposal(p)
    setModalAction(action)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Maker-Checker Approvals</h1>
          <p className="text-sm text-muted-foreground">
            {filtered.filter(
              (p) =>
                p.status === "PENDING_CHECKER" || p.status === "PENDING_MAKER"
            ).length}{" "}
            pending action
          </p>
        </div>
        <ProposalFilters value={statusFilter} onChange={setStatusFilter} />
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <Skeleton key={i} className="h-36 rounded-xl" />
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((p) => (
            <div key={p.id} className="group relative">
              <ProposalCard proposal={p} />
              {(p.status === "PENDING_CHECKER" ||
                p.status === "PENDING_MAKER") && (
                <div className="absolute right-4 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-2">
                  <button
                    onClick={() => openModal(p, "approve")}
                    className="rounded-md bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => openModal(p, "reject")}
                    className="rounded-md bg-destructive px-3 py-1.5 text-xs font-medium text-white hover:bg-destructive/90"
                  >
                    Reject
                  </button>
                </div>
              )}
            </div>
          ))}
          {filtered.length === 0 && (
            <p className="text-sm text-muted-foreground">No proposals match the filter.</p>
          )}
        </div>
      )}

      <ApprovalModal
        proposal={selectedProposal}
        action={modalAction}
        onClose={() => {
          setSelectedProposal(null)
          setModalAction(null)
        }}
      />
    </div>
  )
}
