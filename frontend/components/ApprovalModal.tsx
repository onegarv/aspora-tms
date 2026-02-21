"use client"

import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { useApprove, useReject } from "@/hooks/useProposals"
import type { FundMovementProposal } from "@/lib/types"
import Decimal from "decimal.js"

interface Props {
  proposal: FundMovementProposal | null
  action: "approve" | "reject" | null
  onClose: () => void
}

export function ApprovalModal({ proposal, action, onClose }: Props) {
  const [notes, setNotes] = useState("")
  const approve = useApprove()
  const reject = useReject()

  if (!proposal || !action) return null

  const mutation = action === "approve" ? approve : reject
  const isApprove = action === "approve"

  async function handleSubmit() {
    await mutation.mutateAsync({ id: proposal!.id, notes: notes || undefined })
    setNotes("")
    onClose()
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isApprove ? "Approve" : "Reject"} Proposal
          </DialogTitle>
          <DialogDescription>
            {proposal.currency} ${new Decimal(proposal.amount_usd)
              .toFixed(2)
              .replace(/\B(?=(\d{3})+(?!\d))/g, ",")} · {proposal.id}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <p className="text-sm font-medium mb-1">Transfer route</p>
            <p className="text-sm text-muted-foreground">
              {proposal.from_account} → {proposal.to_account}
            </p>
          </div>

          <div>
            <label
              htmlFor="notes"
              className="mb-1 block text-sm font-medium"
            >
              Notes{isApprove ? " (optional)" : " (required for rejection)"}
            </label>
            <textarea
              id="notes"
              rows={3}
              className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
              placeholder={isApprove ? "Add a comment..." : "Reason for rejection..."}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant={isApprove ? "default" : "destructive"}
            onClick={handleSubmit}
            disabled={mutation.isPending || (!isApprove && !notes.trim())}
          >
            {mutation.isPending
              ? "Processing..."
              : isApprove
              ? "Approve"
              : "Reject"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
