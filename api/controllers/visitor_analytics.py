"""
REST API endpoints for visitor analytics.

Exposes computed visitor statistics with filtering capabilities.

Endpoints:
    GET /visitor-analytics - Get visitor statistics with optional filters
    GET /visitor-analytics/summary - Get aggregate summary of visitor analytics
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api import db

router = APIRouter()


@router.get("/visitor-analytics")
async def get_visitor_analytics(
    visitor_ip: str | None = Query(
        None,
        description="Filter by specific visitor IP address",
        alias="visitor_id",
    ),
    start_date: str | None = Query(
        None,
        description="Filter by start date (ISO8601 format, e.g., 2025-01-01T00:00:00Z)",
    ),
    end_date: str | None = Query(
        None,
        description="Filter by end date (ISO8601 format, e.g., 2025-01-31T23:59:59Z)",
    ),
    recurring_only: bool | None = Query(
        None,
        description="Filter to show only recurring visitors (true) or non-recurring (false)",
        alias="segment",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of records to return",
    ),
) -> dict[str, Any]:
    """
    Get visitor statistics with optional filtering.

    Query Parameters:
        - visitor_id: Filter by specific visitor IP address
        - start_date: Filter by period start date (ISO8601)
        - end_date: Filter by period end date (ISO8601)
        - segment: Filter by visitor segment (true = recurring, false = non-recurring)
        - limit: Maximum number of records (default: 100, max: 1000)

    Returns:
        - visitors: List of visitor statistics records
        - count: Number of records returned
        - filters: Applied filter values
    """
    try:
        stats = await db.fetch_visitor_stats(
            visitor_ip=visitor_ip,
            start_date=start_date,
            end_date=end_date,
            is_recurring=recurring_only,
            limit=limit,
        )

        return {
            "visitors": stats,
            "count": len(stats),
            "filters": {
                "visitor_id": visitor_ip,
                "start_date": start_date,
                "end_date": end_date,
                "segment": (
                    "recurring"
                    if recurring_only is True
                    else ("non-recurring" if recurring_only is False else None)
                ),
                "limit": limit,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch visitor analytics: {e}"
        ) from e


@router.get("/visitor-analytics/summary")
async def get_visitor_analytics_summary(
    start_date: str | None = Query(
        None,
        description="Filter by start date (ISO8601 format)",
    ),
    end_date: str | None = Query(
        None,
        description="Filter by end date (ISO8601 format)",
    ),
) -> dict[str, Any]:
    """
    Get aggregate summary of visitor analytics.

    Query Parameters:
        - start_date: Filter by period start date (ISO8601)
        - end_date: Filter by period end date (ISO8601)

    Returns:
        - unique_visitors: Count of unique visitor IPs
        - total_visits: Total number of visits across all visitors
        - avg_session_duration_seconds: Average session duration
        - total_time_spent_seconds: Total time spent by all visitors
        - recurring_visitors: Count of visitors with multiple visits
        - avg_visit_frequency_per_day: Average visits per day per visitor
    """
    try:
        summary = await db.get_visitor_analytics_summary(
            start_date=start_date,
            end_date=end_date,
        )

        return {
            "summary": summary,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch analytics summary: {e}"
        ) from e


@router.get("/visitor-analytics/{visitor_id}")
async def get_visitor_analytics_by_id(
    visitor_id: str,
    start_date: str | None = Query(None, description="Filter by start date (ISO8601)"),
    end_date: str | None = Query(None, description="Filter by end date (ISO8601)"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """
    Get analytics for a specific visitor by IP address.

    Path Parameters:
        - visitor_id: The visitor's IP address

    Query Parameters:
        - start_date: Filter by period start date (ISO8601)
        - end_date: Filter by period end date (ISO8601)
        - limit: Maximum number of records (default: 100)

    Returns:
        - visitor_id: The queried visitor IP
        - records: List of analytics records for this visitor
        - count: Number of records found
    """
    try:
        stats = await db.fetch_visitor_stats(
            visitor_ip=visitor_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        if not stats:
            return {
                "visitor_id": visitor_id,
                "records": [],
                "count": 0,
                "message": "No analytics data found for this visitor",
            }

        return {
            "visitor_id": visitor_id,
            "records": stats,
            "count": len(stats),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch visitor analytics: {e}"
        ) from e

