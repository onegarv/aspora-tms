import { useQuery } from "@tanstack/react-query"
import { apiFetch } from "@/lib/api"
import type {
  NostroBalance,
  ExposureStatus,
  BankingWindow,
  FxDeal,
  PnLEntry,
  DailyForecast,
  RDAShortfall,
} from "@/lib/types"

export function useBalances() {
  return useQuery({
    queryKey: ["balances"],
    queryFn: () => apiFetch<NostroBalance[]>("/api/v1/balances"),
    refetchInterval: 30_000,
  })
}

export function useExposure() {
  return useQuery({
    queryKey: ["exposure"],
    queryFn: () => apiFetch<ExposureStatus>("/api/v1/exposure"),
    refetchInterval: 60_000,
  })
}

export function useWindows() {
  return useQuery({
    queryKey: ["windows"],
    queryFn: () => apiFetch<BankingWindow[]>("/api/v1/windows"),
    refetchInterval: 30_000,
  })
}

export function useDeals() {
  return useQuery({
    queryKey: ["deals"],
    queryFn: () => apiFetch<FxDeal[]>("/api/v1/deals"),
    refetchInterval: 60_000,
  })
}

export function usePnL() {
  return useQuery({
    queryKey: ["pnl"],
    queryFn: () => apiFetch<PnLEntry[]>("/api/v1/pnl"),
    refetchInterval: 60_000,
  })
}

export function useForecast() {
  return useQuery({
    queryKey: ["forecast"],
    queryFn: () => apiFetch<DailyForecast>("/api/v1/forecast"),
    // Forecast is updated once daily — no aggressive polling needed
    staleTime: 5 * 60_000,
  })
}

export function useShortfalls() {
  return useQuery({
    queryKey: ["shortfalls"],
    queryFn: () => apiFetch<RDAShortfall[]>("/api/v1/forecast/shortfalls"),
    refetchInterval: 60_000,
  })
}
