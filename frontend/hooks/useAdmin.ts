import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { apiFetch, apiPut } from "@/lib/api"
import type { RiskLimits, HolidayEntry } from "@/lib/types"

export function useRiskLimits() {
  return useQuery({
    queryKey: ["risk-limits"],
    queryFn: () => apiFetch<RiskLimits>("/api/v1/risk"),
  })
}

export function useUpdateRiskLimit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (updates: Partial<RiskLimits>) =>
      apiPut<RiskLimits>("/api/v1/risk", updates),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["risk-limits"] })
    },
  })
}

export function useHolidays() {
  return useQuery({
    queryKey: ["holidays"],
    queryFn: () => apiFetch<HolidayEntry[]>("/api/v1/holidays"),
    // Holidays don't change often
    staleTime: 10 * 60_000,
  })
}
