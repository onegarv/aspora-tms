import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { apiFetch, apiPost } from "@/lib/api"
import type { FundMovementProposal, DispatchRequest, DispatchResponse } from "@/lib/types"

export function useProposals(statusFilter?: string) {
  return useQuery({
    queryKey: ["proposals", statusFilter],
    queryFn: () => {
      const qs = statusFilter ? `?status=${statusFilter}` : ""
      return apiFetch<FundMovementProposal[]>(`/api/v1/proposals${qs}`)
    },
    refetchInterval: 30_000,
  })
}

export function useProposal(id: string) {
  return useQuery({
    queryKey: ["proposals", id],
    queryFn: () => apiFetch<FundMovementProposal>(`/api/v1/proposals/${id}`),
    enabled: !!id,
  })
}

export function useApprove() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, notes }: { id: string; notes?: string }) =>
      apiPost<FundMovementProposal>(`/api/v1/proposals/${id}/approve`, {
        notes,
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["proposals"] })
    },
  })
}

export function useReject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, notes }: { id: string; notes?: string }) =>
      apiPost<FundMovementProposal>(`/api/v1/proposals/${id}/reject`, {
        notes,
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["proposals"] })
    },
  })
}

export function useDispatch() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      proposal_id,
      operator_id,
      confirm,
      purpose,
      category_purpose,
    }: { proposal_id: string } & DispatchRequest) =>
      apiPost<DispatchResponse>(`/api/v1/proposals/${proposal_id}/dispatch`, {
        operator_id,
        confirm,
        purpose:          purpose ?? "INTC",
        category_purpose: category_purpose ?? "CASH",
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["proposals"] })
    },
  })
}
