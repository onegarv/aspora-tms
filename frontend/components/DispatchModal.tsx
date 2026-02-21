"use client"

import { useState } from "react"
import Decimal from "decimal.js"
import { CheckCircle2, AlertTriangle, Zap, FlaskConical } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useDispatch } from "@/hooks/useProposals"
import { useTmsStore } from "@/store"
import type { FundMovementProposal, DispatchResponse } from "@/lib/types"

interface Props {
  proposal: FundMovementProposal | null
  onClose: () => void
}

type ConfirmMode = "DEMO" | "LIVE"

function fmt(v: string) {
  return new Decimal(v).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")
}

export function DispatchModal({ proposal, onClose }: Props) {
  const [mode, setMode]           = useState<ConfirmMode>("DEMO")
  const [result, setResult]       = useState<DispatchResponse | null>(null)
  const [errorMsg, setErrorMsg]   = useState<string | null>(null)
  const currentUser               = useTmsStore((s) => s.currentUser)
  const dispatch                  = useDispatch()

  if (!proposal) return null

  const amount    = proposal.amount ?? proposal.amount_usd ?? "0"
  const isGbp     = proposal.currency === "GBP"
  const canDispatch = isGbp && proposal.status === "approved"

  async function handleDispatch() {
    setErrorMsg(null)
    try {
      const res = await dispatch.mutateAsync({
        proposal_id: proposal!.id,
        operator_id: currentUser.id,
        confirm:     mode,
      })
      setResult(res)
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Dispatch failed")
    }
  }

  function handleClose() {
    setResult(null)
    setErrorMsg(null)
    setMode("DEMO")
    onClose()
  }

  // ── Success state ─────────────────────────────────────────────────────────
  if (result) {
    return (
      <Dialog open onOpenChange={handleClose}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-green-600">
              <CheckCircle2 className="size-5" />
              {result.demo_mode ? "Demo Dispatch Complete" : "Transfer Dispatched"}
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-3 rounded-lg border bg-muted/30 p-4 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Payment ID</span>
              <span className="font-mono text-xs">{result.clearbank_payment_id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Amount</span>
              <span className="font-bold">{result.currency} {fmt(result.amount)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Rail</span>
              <span className="uppercase font-medium">{result.rail}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Mode</span>
              {result.demo_mode
                ? <Badge variant="outline" className="text-[10px]">DEMO — no funds moved</Badge>
                : <Badge className="text-[10px] bg-green-600">LIVE</Badge>
              }
            </div>
          </div>

          <p className="text-xs text-muted-foreground">{result.message}</p>

          <DialogFooter>
            <Button onClick={handleClose}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    )
  }

  // ── Main dispatch form ────────────────────────────────────────────────────
  return (
    <Dialog open onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Dispatch to ClearBank</DialogTitle>
          <DialogDescription>
            {proposal.currency} {fmt(amount)} via {proposal.rail?.toUpperCase() ?? "CHAPS/FPS"} ·{" "}
            <span className="font-mono text-xs">{proposal.id}</span>
          </DialogDescription>
        </DialogHeader>

        {/* Non-GBP warning */}
        {!isGbp && (
          <div className="flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-700">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>
              ClearBank only supports GBP. This proposal uses {proposal.currency} and
              cannot be dispatched via ClearBank.
            </span>
          </div>
        )}

        {/* Transfer summary */}
        {isGbp && (
          <div className="space-y-2 rounded-lg border bg-muted/30 p-4 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">From</span>
              <span className="font-mono text-xs">{proposal.source_account ?? proposal.from_account}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">To (Nostro)</span>
              <span className="font-mono text-xs">{proposal.destination_nostro ?? proposal.to_account}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Amount</span>
              <span className="font-bold">GBP {fmt(amount)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Rail</span>
              <span className="uppercase font-medium">{proposal.rail}</span>
            </div>
          </div>
        )}

        {/* Mode selector */}
        {isGbp && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Dispatch mode</p>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setMode("DEMO")}
                className={`flex flex-col items-center gap-1.5 rounded-lg border p-3 text-sm transition-colors ${
                  mode === "DEMO"
                    ? "border-primary bg-primary/5 font-semibold"
                    : "hover:bg-muted"
                }`}
              >
                <FlaskConical className="size-5 text-blue-500" />
                <span>DEMO</span>
                <span className="text-[10px] text-muted-foreground">No real funds move</span>
              </button>
              <button
                onClick={() => setMode("LIVE")}
                className={`flex flex-col items-center gap-1.5 rounded-lg border p-3 text-sm transition-colors ${
                  mode === "LIVE"
                    ? "border-red-500 bg-red-50 font-semibold dark:bg-red-950"
                    : "hover:bg-muted"
                }`}
              >
                <Zap className="size-5 text-red-500" />
                <span>LIVE</span>
                <span className="text-[10px] text-muted-foreground">Real ClearBank call</span>
              </button>
            </div>
          </div>
        )}

        {/* Live mode warning */}
        {isGbp && mode === "LIVE" && (
          <div className="flex items-start gap-2 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200 dark:border-red-700">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>
              <strong>LIVE mode</strong> — this will submit a real GBP {fmt(amount)} payment
              to ClearBank. The transfer cannot be reversed once submitted.
              Guardrails (amount cap £{process.env.NEXT_PUBLIC_MAX_DISPATCH_GBP ?? "500"},
              rate limit) are still enforced.
            </span>
          </div>
        )}

        {/* Error */}
        {errorMsg && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            variant={mode === "LIVE" ? "destructive" : "default"}
            onClick={handleDispatch}
            disabled={!isGbp || !canDispatch || dispatch.isPending}
          >
            {dispatch.isPending
              ? "Dispatching..."
              : mode === "DEMO"
              ? "Run Demo Dispatch"
              : "Dispatch LIVE to ClearBank"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
