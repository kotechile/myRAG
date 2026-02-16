"""
output_structures.py - Output data structures for RAG system

This module defines the output data structures used throughout the RAG system,
including table columns, table data, metrics, and insights.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TableColumn:
    """Represents a table column configuration"""
    id: str
    label: str
    width: str
    field: str
    sortable: bool = False
    searchable: bool = False
    filterable: bool = False
    editable: bool = False
    type: str = "String"
    sort_field: Optional[str] = None
    filter_field: Optional[str] = None
    filter_options: Optional[List[str]] = None


@dataclass
class TrendingTopicsTableData:
    """Data structure for trending topics table"""
    id: str
    rank: int
    topic: str
    viral_score: int
    viral_score_display: str
    monetization_score: int
    monetization_score_display: str
    affiliate_score: int
    affiliate_score_display: str
    revenue_range: str
    opportunity_level: str
    search_volume: str
    competition: str
    primary_keywords: str
    selected: bool = False


@dataclass
class ContentOpportunitiesTableData:
    """Data structure for content opportunities table"""
    id: str
    rank: int
    title: str
    format: str
    format_with_icon: str
    monetization_score: int
    monetization_score_display: str
    affiliate_score: int
    affiliate_score_display: str
    revenue_range: str
    engagement_potential: str
    difficulty: str
    difficulty_display: str
    time_investment: str
    selected: bool = False


@dataclass
class OverviewMetrics:
    """Overview metrics for the analysis"""
    total_topics: int
    total_opportunities: int
    filtered_topics: int
    filtered_opportunities: int
    selected_topics: int
    selected_opportunities: int
    confidence_score: int
    ready_for_phase2: bool
    selection_progress: int
    topic_selection_rate: int
    opportunity_selection_rate: int
    high_potential_selections: bool
    high_monetization_selections: bool
    high_affiliate_selections: bool
    total_estimated_revenue: float
    avg_monetization_score: float
    avg_affiliate_score: float


@dataclass
class MainTopicAnalysis:
    """Main topic analysis data"""
    current_interest: int
    trend_direction: str
    momentum: int
    peak_interest: int
    recommendation: str
    growth_potential: str


@dataclass
class GeographicHotspot:
    """Geographic hotspot data"""
    country: str
    interest_score: int
    growth_rate: float
    market_size: str


@dataclass
class SeasonalPattern:
    """Seasonal pattern data"""
    has_pattern: bool
    peak_months: List[str]
    next_peak: Optional[str]


@dataclass
class RelatedQueries:
    """Related queries data"""
    top_queries: List[str]
    rising_queries: List[str]


@dataclass
class ActionableInsight:
    """Actionable insight data"""
    id: str
    type: str
    priority: str
    title: str
    description: str
    action: str
    timing: str


@dataclass
class PyTrendsInsights:
    """PyTrends insights data structure"""
    available: bool
    main_topic_analysis: MainTopicAnalysis
    geographic_hotspots: List[GeographicHotspot]
    seasonal_patterns: SeasonalPattern
    related_queries: RelatedQueries
    actionable_insights: List[ActionableInsight]


class OutputStructures:
    """Main class containing all output structures"""
    
    # Table Columns
    @staticmethod
    def get_trending_topics_table_columns() -> List[TableColumn]:
        """Get trending topics table columns configuration"""
        return [
            TableColumn(
                id="selected",
                label="Select",
                width="4%",
                field="selected",
                editable=True,
                type="Boolean"
            ),
            TableColumn(
                id="rank",
                label="#",
                width="3%",
                field="rank",
                sortable=True,
                type="String"
            ),
            TableColumn(
                id="topic",
                label="Trending Topic",
                width="18%",
                field="topic",
                sortable=True,
                searchable=True,
                type="String"
            ),
            TableColumn(
                id="viral_score_display",
                label="Viral Potential",
                width="10%",
                field="viral_score_display",
                sortable=True,
                sort_field="viral_score",
                type="String"
            ),
            TableColumn(
                id="monetization_score_display",
                label="Monetization",
                width="10%",
                field="monetization_score_display",
                sortable=True,
                sort_field="monetization_score",
                type="String"
            ),
            TableColumn(
                id="affiliate_score_display",
                label="Affiliate Score",
                width="10%",
                field="affiliate_score_display",
                sortable=True,
                sort_field="affiliate_score",
                type="String"
            ),
            TableColumn(
                id="revenue_range",
                label="Revenue Range",
                width="11%",
                field="revenue_range",
                type="String"
            ),
            TableColumn(
                id="opportunity_level",
                label="Opportunity",
                width="9%",
                field="opportunity_level",
                sortable=True,
                sort_field="opportunity_score",
                type="String"
            ),
            TableColumn(
                id="search_volume",
                label="Search Volume",
                width="8%",
                field="search_volume",
                filterable=True,
                filter_options=["All", "High", "Medium", "Low"],
                type="String"
            ),
            TableColumn(
                id="competition",
                label="Competition",
                width="8%",
                field="competition",
                filterable=True,
                filter_options=["All", "Low", "Medium", "High"],
                type="String"
            ),
            TableColumn(
                id="primary_keywords",
                label="Top Keywords",
                width="9%",
                field="primary_keywords",
                type="String"
            )
        ]
    
    @staticmethod
    def get_content_opportunities_table_columns() -> List[TableColumn]:
        """Get content opportunities table columns configuration"""
        return [
            TableColumn(
                id="selected",
                label="Select",
                width="4%",
                field="selected",
                editable=True,
                type="Boolean"
            ),
            TableColumn(
                id="rank",
                label="#",
                width="3%",
                field="rank",
                sortable=True,
                type="String"
            ),
            TableColumn(
                id="title",
                label="Content Opportunity",
                width="20%",
                field="title",
                sortable=True,
                searchable=True,
                type="String"
            ),
            TableColumn(
                id="format_with_icon",
                label="Content Type",
                width="10%",
                field="format_with_icon",
                filterable=True,
                filter_field="format",
                type="String"
            ),
            TableColumn(
                id="monetization_score_display",
                label="Monetization",
                width="10%",
                field="monetization_score_display",
                sortable=True,
                sort_field="monetization_score",
                type="String"
            ),
            TableColumn(
                id="affiliate_score_display",
                label="Affiliate Score",
                width="10%",
                field="affiliate_score_display",
                sortable=True,
                sort_field="affiliate_score",
                type="String"
            ),
            TableColumn(
                id="revenue_range",
                label="Revenue Range",
                width="11%",
                field="revenue_range",
                type="String"
            ),
            TableColumn(
                id="engagement_potential",
                label="Engagement",
                width="8%",
                field="engagement_potential",
                filterable=True,
                filter_options=["All", "High", "Medium", "Low"],
                type="String"
            ),
            TableColumn(
                id="difficulty_display",
                label="Difficulty",
                width="10%",
                field="difficulty_display",
                sortable=True,
                sort_field="difficulty",
                type="String"
            ),
            TableColumn(
                id="time_investment",
                label="Time Investment",
                width="10%",
                field="time_investment",
                filterable=True,
                filter_options=["All", "1-2 weeks", "2-3 weeks", "3-4 weeks", "1+ month"],
                type="String"
            )
        ]
    
    @staticmethod
    def create_empty_overview_metrics() -> OverviewMetrics:
        """Create empty overview metrics"""
        return OverviewMetrics(
            total_topics=0,
            total_opportunities=0,
            filtered_topics=0,
            filtered_opportunities=0,
            selected_topics=0,
            selected_opportunities=0,
            confidence_score=0,
            ready_for_phase2=False,
            selection_progress=0,
            topic_selection_rate=0,
            opportunity_selection_rate=0,
            high_potential_selections=False,
            high_monetization_selections=False,
            high_affiliate_selections=False,
            total_estimated_revenue=0.0,
            avg_monetization_score=0.0,
            avg_affiliate_score=0.0
        )
    
    @staticmethod
    def create_empty_pytrends_insights() -> PyTrendsInsights:
        """Create empty PyTrends insights"""
        return PyTrendsInsights(
            available=False,
            main_topic_analysis=MainTopicAnalysis(
                current_interest=0,
                trend_direction="unknown",
                momentum=0,
                peak_interest=0,
                recommendation="PyTrends data not available",
                growth_potential="unknown"
            ),
            geographic_hotspots=[],
            seasonal_patterns=SeasonalPattern(
                has_pattern=False,
                peak_months=[],
                next_peak=None
            ),
            related_queries=RelatedQueries(
                top_queries=[],
                rising_queries=[]
            ),
            actionable_insights=[]
        )
    
    @staticmethod
    def create_trending_topics_table_data(
        topics: List[Dict[str, Any]]
    ) -> List[TrendingTopicsTableData]:
        """Create trending topics table data from raw topics"""
        table_data = []
        for i, topic in enumerate(topics):
            table_data.append(TrendingTopicsTableData(
                id=topic.get("id", f"topic_{i}"),
                rank=i + 1,
                topic=topic.get("topic", ""),
                viral_score=topic.get("viral_score", 0),
                viral_score_display=topic.get("viral_score_display", "0%"),
                monetization_score=topic.get("monetization_score", 0),
                monetization_score_display=topic.get("monetization_score_display", "0%"),
                affiliate_score=topic.get("affiliate_score", 0),
                affiliate_score_display=topic.get("affiliate_score_display", "0%"),
                revenue_range=topic.get("revenue_range", "$0 - $0"),
                opportunity_level=topic.get("opportunity_level", "Unknown"),
                search_volume=topic.get("search_volume", "Unknown"),
                competition=topic.get("competition", "Unknown"),
                primary_keywords=topic.get("primary_keywords", ""),
                selected=topic.get("selected", False)
            ))
        return table_data
    
    @staticmethod
    def create_content_opportunities_table_data(
        opportunities: List[Dict[str, Any]]
    ) -> List[ContentOpportunitiesTableData]:
        """Create content opportunities table data from raw opportunities"""
        table_data = []
        for i, opportunity in enumerate(opportunities):
            table_data.append(ContentOpportunitiesTableData(
                id=opportunity.get("id", f"opportunity_{i}"),
                rank=i + 1,
                title=opportunity.get("title", ""),
                format=opportunity.get("format", ""),
                format_with_icon=opportunity.get("format_with_icon", ""),
                monetization_score=opportunity.get("monetization_score", 0),
                monetization_score_display=opportunity.get("monetization_score_display", "0%"),
                affiliate_score=opportunity.get("affiliate_score", 0),
                affiliate_score_display=opportunity.get("affiliate_score_display", "0%"),
                revenue_range=opportunity.get("revenue_range", "$0 - $0"),
                engagement_potential=opportunity.get("engagement_potential", "Unknown"),
                difficulty=opportunity.get("difficulty", "Unknown"),
                difficulty_display=opportunity.get("difficulty_display", "Unknown"),
                time_investment=opportunity.get("time_investment", "Unknown"),
                selected=opportunity.get("selected", False)
            ))
        return table_data


# Convenience functions for easy access
def get_trending_topics_table_columns() -> List[TableColumn]:
    """Get trending topics table columns"""
    return OutputStructures.get_trending_topics_table_columns()


def get_content_opportunities_table_columns() -> List[TableColumn]:
    """Get content opportunities table columns"""
    return OutputStructures.get_content_opportunities_table_columns()


def get_overview_metrics() -> OverviewMetrics:
    """Get overview metrics"""
    return OutputStructures.create_empty_overview_metrics()


def get_pytrends_insights() -> PyTrendsInsights:
    """Get PyTrends insights"""
    return OutputStructures.create_empty_pytrends_insights()


def get_trending_topics_table_data(topics: List[Dict[str, Any]]) -> List[TrendingTopicsTableData]:
    """Get trending topics table data"""
    return OutputStructures.create_trending_topics_table_data(topics)


def get_content_opportunities_table_data(opportunities: List[Dict[str, Any]]) -> List[ContentOpportunitiesTableData]:
    """Get content opportunities table data"""
    return OutputStructures.create_content_opportunities_table_data(opportunities)








