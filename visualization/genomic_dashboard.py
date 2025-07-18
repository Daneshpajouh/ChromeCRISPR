"""
Genomic Dashboard
Interactive visualization dashboard for gene editing data using Streamlit and Plotly
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class GenomicDashboard:
    """
    Comprehensive genomic dashboard for gene editing data visualization
    Provides interactive charts and analytics for CRISPR, Prime Editing, and Base Editing domains
    """
    
    def __init__(self):
        """Initialize the dashboard with data loaders and visualizers"""
        self.data_loader = None  # Will be set by external data source
        self.plotter = PlotlyVisualizer()
        
        # Dashboard configuration
        st.set_page_config(
            page_title="GeneX Phase 1 - Genomic Dashboard",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .domain-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_main_dashboard(self, data: Dict[str, Any]):
        """Render the main dashboard with comprehensive analytics"""
        # Header
        st.markdown('<h1 class="main-header">🧬 GeneX Phase 1 - Genomic Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar for filters
        self._render_sidebar(data)
        
        # Main content area
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_overview_metrics(data, 'crispr')
        with col2:
            self._render_overview_metrics(data, 'prime_editing')
        with col3:
            self._render_overview_metrics(data, 'base_editing')
        with col4:
            self._render_overview_metrics(data, 'overall')
        
        # Domain-specific tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 CRISPR", "🎯 Prime Editing", "🧬 Base Editing"])
        
        with tab1:
            self._render_overview_analytics(data)
        
        with tab2:
            self._render_domain_analytics(data, 'crispr')
        
        with tab3:
            self._render_domain_analytics(data, 'prime_editing')
        
        with tab4:
            self._render_domain_analytics(data, 'base_editing')

    def _render_sidebar(self, data: Dict[str, Any]):
        """Render sidebar with filters and controls"""
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Date range filter
        st.sidebar.subheader("📅 Date Range")
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Organism filter
        st.sidebar.subheader("🧬 Organism Filter")
        organisms = self._get_available_organisms(data)
        selected_organisms = st.sidebar.multiselect(
            "Select organisms",
            options=organisms,
            default=organisms[:3] if organisms else []
        )
        
        # Quality threshold
        st.sidebar.subheader("📈 Quality Threshold")
        quality_threshold = st.sidebar.slider(
            "Minimum quality score",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        # Therapeutic focus
        st.sidebar.subheader("💊 Therapeutic Focus")
        therapeutic_only = st.sidebar.checkbox("Show therapeutic studies only", value=False)
        
        # Store filters in session state
        st.session_state.filters = {
            'date_range': date_range,
            'organisms': selected_organisms,
            'quality_threshold': quality_threshold,
            'therapeutic_only': therapeutic_only
        }

    def _render_overview_metrics(self, data: Dict[str, Any], domain: str):
        """Render overview metrics for a domain"""
        if domain == 'overall':
            title = "📊 Overall"
            metrics = self._calculate_overall_metrics(data)
        else:
            title = f"🔬 {domain.replace('_', ' ').title()}"
            metrics = self._get_domain_metrics(data, domain)
        
        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<h3>{title}</h3>', unsafe_allow_html=True)
        
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Experiments", f"{metrics.get('total_experiments', 0):,}")
                st.metric("Compliance", f"{metrics.get('compliance_rate', 0):.1f}%")
            with col2:
                st.metric("Human Studies", f"{metrics.get('human_studies', 0):,}")
                st.metric("Therapeutic", f"{metrics.get('therapeutic_rate', 0):.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_overview_analytics(self, data: Dict[str, Any]):
        """Render overview analytics with cross-domain comparisons"""
        st.header("📊 Cross-Domain Analytics")
        
        # Efficiency comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Efficiency Comparison")
            efficiency_data = self._prepare_efficiency_comparison(data)
            if efficiency_data is not None:
                fig = px.box(
                    efficiency_data,
                    x='domain',
                    y='efficiency',
                    color='organism_type',
                    title="Efficiency Distribution by Domain and Organism"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Quality Distribution")
            quality_data = self._prepare_quality_distribution(data)
            if quality_data is not None:
                fig = px.pie(
                    quality_data,
                    values='count',
                    names='quality_level',
                    title="Data Quality Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Temporal trends
        st.subheader("📅 Temporal Trends")
        temporal_data = self._prepare_temporal_data(data)
        if temporal_data is not None:
            fig = px.line(
                temporal_data,
                x='date',
                y='count',
                color='domain',
                title="Publication Trends Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Organism distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Organism Distribution")
            organism_data = self._prepare_organism_distribution(data)
            if organism_data is not None:
                fig = px.bar(
                    organism_data,
                    x='organism',
                    y='count',
                    color='domain',
                    title="Organism Distribution by Domain"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💊 Therapeutic Focus")
            therapeutic_data = self._prepare_therapeutic_data(data)
            if therapeutic_data is not None:
                fig = px.bar(
                    therapeutic_data,
                    x='domain',
                    y='therapeutic_rate',
                    title="Therapeutic Study Rate by Domain"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_domain_analytics(self, data: Dict[str, Any], domain: str):
        """Render domain-specific analytics"""
        st.header(f"🔬 {domain.replace('_', ' ').title()} Analytics")
        
        # Domain-specific metrics
        metrics = self._get_domain_metrics(data, domain)
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Studies", f"{metrics.get('total_experiments', 0):,}")
            with col2:
                st.metric("Avg Efficiency", f"{metrics.get('avg_efficiency', 0):.1f}%")
            with col3:
                st.metric("Human Studies", f"{metrics.get('human_studies', 0):,}")
            with col4:
                st.metric("Therapeutic Rate", f"{metrics.get('therapeutic_rate', 0):.1f}%")
        
        # Domain-specific charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Efficiency Analysis")
            efficiency_data = self._prepare_domain_efficiency_data(data, domain)
            if efficiency_data is not None:
                fig = px.histogram(
                    efficiency_data,
                    x='efficiency',
                    nbins=20,
                    title=f"{domain.title()} Efficiency Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Quality Assessment")
            quality_data = self._prepare_domain_quality_data(data, domain)
            if quality_data is not None:
                fig = px.bar(
                    quality_data,
                    x='quality_level',
                    y='count',
                    title=f"{domain.title()} Data Quality Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Advanced analytics
        st.subheader("🔍 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Target Gene Analysis")
            gene_data = self._prepare_target_gene_data(data, domain)
            if gene_data is not None:
                fig = px.bar(
                    gene_data.head(10),
                    x='gene',
                    y='count',
                    title=f"Top 10 Target Genes in {domain.title()}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Publication Trends")
            trend_data = self._prepare_domain_trends(data, domain)
            if trend_data is not None:
                fig = px.line(
                    trend_data,
                    x='year',
                    y='publications',
                    title=f"{domain.title()} Publication Trends"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _get_available_organisms(self, data: Dict[str, Any]) -> List[str]:
        """Get list of available organisms from data"""
        organisms = set()
        for domain_data in data.values():
            if isinstance(domain_data, dict) and 'experiments' in domain_data:
                for exp in domain_data['experiments']:
                    if 'organism' in exp:
                        organisms.add(exp['organism'])
        return sorted(list(organisms))

    def _calculate_overall_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics across all domains"""
        total_experiments = 0
        total_human = 0
        total_therapeutic = 0
        compliance_rates = []
        
        for domain_data in data.values():
            if isinstance(domain_data, dict):
                total_experiments += domain_data.get('total_experiments', 0)
                total_human += domain_data.get('human_studies', 0)
                total_therapeutic += domain_data.get('therapeutic_studies', 0)
                compliance_rates.append(domain_data.get('compliance_rate', 0))
        
        return {
            'total_experiments': total_experiments,
            'human_studies': total_human,
            'therapeutic_studies': total_therapeutic,
            'compliance_rate': np.mean(compliance_rates) if compliance_rates else 0,
            'therapeutic_rate': (total_therapeutic / total_experiments * 100) if total_experiments > 0 else 0
        }

    def _get_domain_metrics(self, data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Get metrics for a specific domain"""
        if domain in data:
            return data[domain]
        return {}

    def _prepare_efficiency_comparison(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare efficiency comparison data"""
        records = []
        for domain, domain_data in data.items():
            if isinstance(domain_data, dict) and 'experiments' in domain_data:
                for exp in domain_data['experiments']:
                    if 'efficiency' in exp:
                        records.append({
                            'domain': domain.replace('_', ' ').title(),
                            'efficiency': exp['efficiency'],
                            'organism_type': exp.get('organism_type', 'unknown')
                        })
        
        return pd.DataFrame(records) if records else None

    def _prepare_quality_distribution(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare quality distribution data"""
        quality_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for domain_data in data.values():
            if isinstance(domain_data, dict) and 'quality_distribution' in domain_data:
                for level, count in domain_data['quality_distribution'].items():
                    quality_counts[level] += count
        
        records = [{'quality_level': level, 'count': count} 
                  for level, count in quality_counts.items() if count > 0]
        
        return pd.DataFrame(records) if records else None

    def _prepare_temporal_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare temporal data for trend analysis"""
        records = []
        for domain, domain_data in data.items():
            if isinstance(domain_data, dict) and 'experiments' in domain_data:
                for exp in domain_data['experiments']:
                    if 'year' in exp:
                        records.append({
                            'date': datetime(exp['year'], 1, 1),
                            'count': 1,
                            'domain': domain.replace('_', ' ').title()
                        })
        
        if records:
            df = pd.DataFrame(records)
            return df.groupby(['date', 'domain']).sum().reset_index()
        return None

    def _prepare_organism_distribution(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare organism distribution data"""
        organism_counts = {}
        
        for domain, domain_data in data.items():
            if isinstance(domain_data, dict) and 'organism_distribution' in domain_data:
                for organism, count in domain_data['organism_distribution'].items():
                    key = (domain.replace('_', ' ').title(), organism)
                    organism_counts[key] = organism_counts.get(key, 0) + count
        
        records = [{'domain': domain, 'organism': organism, 'count': count}
                  for (domain, organism), count in organism_counts.items()]
        
        return pd.DataFrame(records) if records else None

    def _prepare_therapeutic_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare therapeutic data"""
        records = []
        for domain, domain_data in data.items():
            if isinstance(domain_data, dict):
                records.append({
                    'domain': domain.replace('_', ' ').title(),
                    'therapeutic_rate': domain_data.get('therapeutic_relevance_rate', 0)
                })
        
        return pd.DataFrame(records) if records else None

    def _prepare_domain_efficiency_data(self, data: Dict[str, Any], domain: str) -> Optional[pd.DataFrame]:
        """Prepare efficiency data for a specific domain"""
        if domain not in data or 'experiments' not in data[domain]:
            return None
        
        records = []
        for exp in data[domain]['experiments']:
            if 'efficiency' in exp:
                records.append({'efficiency': exp['efficiency']})
        
        return pd.DataFrame(records) if records else None

    def _prepare_domain_quality_data(self, data: Dict[str, Any], domain: str) -> Optional[pd.DataFrame]:
        """Prepare quality data for a specific domain"""
        if domain not in data or 'quality_distribution' not in data[domain]:
            return None
        
        records = [{'quality_level': level, 'count': count}
                  for level, count in data[domain]['quality_distribution'].items()]
        
        return pd.DataFrame(records) if records else None

    def _prepare_target_gene_data(self, data: Dict[str, Any], domain: str) -> Optional[pd.DataFrame]:
        """Prepare target gene data for a specific domain"""
        if domain not in data or 'experiments' not in data[domain]:
            return None
        
        gene_counts = {}
        for exp in data[domain]['experiments']:
            if 'target_gene' in exp:
                gene = exp['target_gene']
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
        
        records = [{'gene': gene, 'count': count}
                  for gene, count in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)]
        
        return pd.DataFrame(records) if records else None

    def _prepare_domain_trends(self, data: Dict[str, Any], domain: str) -> Optional[pd.DataFrame]:
        """Prepare publication trends for a specific domain"""
        if domain not in data or 'experiments' not in data[domain]:
            return None
        
        year_counts = {}
        for exp in data[domain]['experiments']:
            if 'year' in exp:
                year = exp['year']
                year_counts[year] = year_counts.get(year, 0) + 1
        
        records = [{'year': year, 'publications': count}
                  for year, count in sorted(year_counts.items())]
        
        return pd.DataFrame(records) if records else None


class PlotlyVisualizer:
    """Helper class for creating Plotly visualizations"""
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        self.color_scheme = px.colors.qualitative.Set3
        self.template = "plotly_white"
    
    def create_efficiency_trend(self, data: pd.DataFrame, domain: str) -> go.Figure:
        """Create efficiency trend visualization"""
        fig = px.line(
            data,
            x='date',
            y='efficiency',
            color='organism',
            title=f'{domain} Efficiency Trends',
            template=self.template
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Efficiency (%)",
            hovermode='x unified'
        )
        return fig
    
    def create_quality_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Create quality distribution visualization"""
        fig = px.pie(
            data,
            values='count',
            names='quality_level',
            title="Data Quality Distribution",
            template=self.template
        )
        return fig
    
    def create_organism_comparison(self, data: pd.DataFrame) -> go.Figure:
        """Create organism comparison visualization"""
        fig = px.bar(
            data,
            x='organism',
            y='count',
            color='domain',
            title="Organism Distribution by Domain",
            template=self.template
        )
        fig.update_layout(
            xaxis_title="Organism",
            yaxis_title="Number of Studies",
            barmode='group'
        )
        return fig


def run_dashboard(data: Dict[str, Any]):
    """Run the genomic dashboard"""
    dashboard = GenomicDashboard()
    dashboard.render_main_dashboard(data)


if __name__ == "__main__":
    # Example usage
    sample_data = {
        'crispr': {
            'total_experiments': 381,
            'compliance_rate': 88.71,
            'human_studies': 150,
            'therapeutic_rate': 45.2,
            'experiments': []
        }
    }
    
    run_dashboard(sample_data)
