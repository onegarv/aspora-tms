"""
Risk Limits router — GET /risk and PUT /risk

GET  returns the current effective risk limits (settings + any in-memory overrides).
PUT  stores overrides in app.state.risk_overrides (resets on restart; suitable for dev).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from api.auth import require_auth
from api.schemas import RiskLimitsResponse, RiskLimitsUpdate, _dec
from config.settings import settings

router = APIRouter(prefix="/risk", tags=["risk"])

_SYSTEM_START = datetime.now(timezone.utc).isoformat()


def _build_response(overrides: dict) -> RiskLimitsResponse:
    return RiskLimitsResponse(
        max_single_deal_usd=overrides.get(
            "max_single_deal_usd", _dec(settings.max_single_deal_usd)
        ),
        max_open_exposure_pct=overrides.get(
            "max_open_exposure_pct", _dec(settings.max_open_exposure_pct)
        ),
        stop_loss_paise=overrides.get(
            "stop_loss_paise", str(settings.stop_loss_paise)
        ),
        dual_checker_threshold_usd=overrides.get(
            "dual_checker_threshold_usd", _dec(settings.dual_checker_threshold_usd)
        ),
        prefunding_buffer_pct=overrides.get(
            "prefunding_buffer_pct", _dec(settings.prefunding_buffer_pct)
        ),
        updated_at=overrides.get("updated_at", _SYSTEM_START),
        updated_by=overrides.get("updated_by", "system"),
    )


@router.get("", response_model=RiskLimitsResponse, dependencies=[Depends(require_auth)])
async def get_risk_limits(request: Request) -> RiskLimitsResponse:
    overrides: dict = getattr(request.app.state, "risk_overrides", {})
    return _build_response(overrides)


@router.put("", response_model=RiskLimitsResponse, dependencies=[Depends(require_auth)])
async def update_risk_limits(body: RiskLimitsUpdate, request: Request) -> RiskLimitsResponse:
    if not hasattr(request.app.state, "risk_overrides"):
        request.app.state.risk_overrides = {}

    overrides: dict = request.app.state.risk_overrides
    for field, value in body.model_dump(exclude_none=True).items():
        overrides[field] = value

    overrides["updated_at"] = datetime.now(timezone.utc).isoformat()
    overrides["updated_by"] = "treasury_admin"  # TODO: extract from JWT claims

    return _build_response(overrides)
