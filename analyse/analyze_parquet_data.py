"""
Parquet Data Analysis Tool
Comprehensive analysis and manipulation of parquet files
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("Error: pyarrow not installed. Please run: pip install pyarrow")
    sys.exit(1)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not installed. Visualization features will be limited.")
    print("Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


class ParquetAnalyzer:
    """Comprehensive parquet file analyzer"""

    def __init__(self):
        self.df = None
        self.metadata = None
        self.file_path = None

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load a parquet file"""
        try:
            self.file_path = file_path
            self.df = pd.read_parquet(file_path)

            # Get metadata
            parquet_file = pq.ParquetFile(file_path)
            self.metadata = parquet_file.metadata

            return self.df
        except Exception as e:
            print(f"Error loading parquet file: {e}")
            return None

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a parquet file"""
        df = self.load_parquet(file_path)
        if df is None:
            return None

        analysis = {
            'file_info': self.get_file_info(file_path),
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'null_counts': df.isnull().sum().to_dict(),
            'basic_stats': self.get_basic_stats(df),
            'metadata': self.get_metadata_info()
        }

        return analysis

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        path = Path(file_path)
        stats = path.stat()

        return {
            'name': path.name,
            'size_mb': stats.st_size / 1024**2,
            'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
        }

    def get_metadata_info(self) -> Dict[str, Any]:
        """Extract parquet metadata information"""
        if self.metadata is None:
            return {}

        return {
            'num_rows': self.metadata.num_rows,
            'num_columns': self.metadata.num_columns,
            'num_row_groups': self.metadata.num_row_groups,
            'format_version': str(self.metadata.format_version),
            'created_by': self.metadata.created_by if self.metadata.created_by else 'Unknown',
            'compression': self.get_compression_info()
        }

    def get_compression_info(self) -> str:
        """Get compression codec used"""
        if self.metadata and self.metadata.row_group(0).column(0).compression:
            return self.metadata.row_group(0).column(0).compression
        return 'UNCOMPRESSED'

    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}

        stats = numeric_df.describe().to_dict()

        # Add additional statistics
        for col in numeric_df.columns:
            stats[col]['skew'] = numeric_df[col].skew()
            stats[col]['kurtosis'] = numeric_df[col].kurtosis()
            stats[col]['unique'] = numeric_df[col].nunique()

        return stats

    def convert_to_csv(self, input_file: str, output_file: str):
        """Convert parquet to CSV"""
        df = self.load_parquet(input_file)
        if df is not None:
            df.to_csv(output_file, index=False)
            print(f"Successfully converted to CSV: {output_file}")
            print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    def convert_to_json(self, input_file: str, output_file: str, orient: str = 'records'):
        """Convert parquet to JSON"""
        df = self.load_parquet(input_file)
        if df is not None:
            # Handle datetime columns
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

            df.to_json(output_file, orient=orient, indent=2)
            print(f"Successfully converted to JSON: {output_file}")
            print(f"Format: {orient}, Rows: {len(df)}")

    def merge_parquet_files(self, file_list: List[str], output_file: str):
        """Merge multiple parquet files"""
        dfs = []
        for file in file_list:
            df = pd.read_parquet(file.strip())
            dfs.append(df)
            print(f"Loaded {file}: {len(df)} rows")

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_parquet(output_file, compression='snappy')
        print(f"Merged {len(dfs)} files into {output_file}")
        print(f"Total rows: {len(merged_df)}, Columns: {len(merged_df.columns)}")

    def generate_stats_report(self, file_path: str, report_type: int = 1) -> str:
        """Generate detailed statistics report"""
        df = self.load_parquet(file_path)
        if df is None:
            return "Error loading file"

        report = []
        report.append("=" * 80)
        report.append(f"PARQUET FILE ANALYSIS REPORT")
        report.append(f"File: {file_path}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        # File information
        file_info = self.get_file_info(file_path)
        report.append("\n## FILE INFORMATION")
        report.append(f"Size: {file_info['size_mb']:.2f} MB")
        report.append(f"Modified: {file_info['modified']}")

        # Data shape
        report.append("\n## DATA SHAPE")
        report.append(f"Rows: {df.shape[0]:,}")
        report.append(f"Columns: {df.shape[1]}")
        report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Column information
        report.append("\n## COLUMN INFORMATION")
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            unique = df[col].nunique()
            report.append(f"  - {col}: {dtype} | Nulls: {nulls:,} | Unique: {unique:,}")

        if report_type >= 1:
            # Basic statistics
            report.append("\n## NUMERIC STATISTICS")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats = numeric_df.describe()
                report.append(stats.to_string())

        if report_type >= 2:
            # Percentiles
            report.append("\n## PERCENTILES")
            for col in numeric_df.columns:
                percentiles = df[col].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                report.append(f"\n{col}:")
                for p, v in percentiles.items():
                    report.append(f"  {p*100:.0f}%: {v:.4f}")

        if report_type >= 3:
            # Data quality
            report.append("\n## DATA QUALITY")
            report.append(f"Total Null Values: {df.isnull().sum().sum():,}")
            report.append(f"Duplicate Rows: {df.duplicated().sum():,}")

            # Missing data by column
            report.append("\nMissing Data Percentage:")
            missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            for col, pct in missing[missing > 0].items():
                report.append(f"  - {col}: {pct:.2f}%")

        if report_type >= 4:
            # Correlation analysis for numeric columns
            if len(numeric_df.columns) > 1:
                report.append("\n## CORRELATION MATRIX (Top 10)")
                corr = numeric_df.corr()
                # Get top correlations
                corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        corr_pairs.append((
                            corr.columns[i],
                            corr.columns[j],
                            corr.iloc[i, j]
                        ))
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr_val in corr_pairs[:10]:
                    report.append(f"  {col1} <-> {col2}: {corr_val:.4f}")

        return "\n".join(report)

    def visualize_data(self, file_path: str, viz_type: int = 1):
        """Create visualizations of the data"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Please install with: pip install plotly")
            return

        df = self.load_parquet(file_path)
        if df is None:
            return

        if viz_type == 1:
            self.create_distribution_plots(df)
        elif viz_type == 2:
            self.create_correlation_heatmap(df)
        elif viz_type == 3:
            self.create_time_series_plots(df)
        elif viz_type == 4:
            self.create_scatter_plots(df)
        elif viz_type == 5:
            self.create_dashboard(df)

    def create_distribution_plots(self, df: pd.DataFrame):
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("No numeric columns found for distribution plots")
            return

        n_cols = min(len(numeric_cols), 9)  # Limit to 9 subplots
        n_rows = (n_cols + 2) // 3
        n_cols_subplot = min(3, n_cols)

        fig = make_subplots(
            rows=n_rows, cols=n_cols_subplot,
            subplot_titles=list(numeric_cols[:n_cols])
        )

        for i, col in enumerate(numeric_cols[:n_cols]):
            row = i // 3 + 1
            col_idx = i % 3 + 1

            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_idx
            )

        fig.update_layout(
            title="Distribution Plots",
            height=300 * n_rows,
            showlegend=False
        )

        fig.write_html("distribution_plots.html")
        print("Distribution plots saved to: distribution_plots.html")
        fig.show()

    def create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            print("Need at least 2 numeric columns for correlation heatmap")
            return

        corr = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Correlation Heatmap",
            width=800,
            height=800
        )

        fig.write_html("correlation_heatmap.html")
        print("Correlation heatmap saved to: correlation_heatmap.html")
        fig.show()

    def create_time_series_plots(self, df: pd.DataFrame):
        """Create time series plots if date columns exist"""
        # Find date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Try to parse date columns
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols = [col]
                        break
                    except:
                        pass

        if len(date_cols) == 0:
            print("No date columns found for time series plots")
            return

        date_col = date_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5

        fig = make_subplots(
            rows=len(numeric_cols), cols=1,
            subplot_titles=list(numeric_cols),
            shared_xaxes=True
        )

        for i, col in enumerate(numeric_cols):
            fig.add_trace(
                go.Scatter(x=df[date_col], y=df[col], mode='lines', name=col),
                row=i+1, col=1
            )

        fig.update_layout(
            title="Time Series Plots",
            height=200 * len(numeric_cols),
            showlegend=False
        )

        fig.write_html("time_series_plots.html")
        print("Time series plots saved to: time_series_plots.html")
        fig.show()

    def create_scatter_plots(self, df: pd.DataFrame):
        """Create scatter plot matrix"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5

        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for scatter plots")
            return

        fig = px.scatter_matrix(
            df[numeric_cols],
            dimensions=numeric_cols,
            title="Scatter Plot Matrix"
        )

        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=800, width=800)

        fig.write_html("scatter_plots.html")
        print("Scatter plots saved to: scatter_plots.html")
        fig.show()

    def create_dashboard(self, df: pd.DataFrame):
        """Create interactive dashboard"""
        print("Creating interactive dashboard...")

        # This would create a more complex dashboard
        # For now, create a simple summary dashboard
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("No numeric columns for dashboard")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Data Overview", "Top Correlations",
                          "Missing Data", "Distribution Sample"),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )

        # Data overview table
        overview_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{len(df):,}"],
            ["Total Columns", f"{len(df.columns)}"],
            ["Numeric Columns", f"{len(numeric_cols)}"],
            ["Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"]
        ]

        fig.add_trace(
            go.Table(
                cells=dict(values=list(zip(*overview_data)))
            ),
            row=1, col=1
        )

        # Top correlations
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            corr_values = []
            corr_labels = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_values.append(abs(corr.iloc[i, j]))
                    corr_labels.append(f"{corr.columns[i][:10]}-{corr.columns[j][:10]}")

            top_5_idx = np.argsort(corr_values)[-5:]
            fig.add_trace(
                go.Bar(x=[corr_labels[i] for i in top_5_idx],
                      y=[corr_values[i] for i in top_5_idx]),
                row=1, col=2
            )

        # Missing data
        missing = df.isnull().sum().sort_values(ascending=False)[:10]
        fig.add_trace(
            go.Bar(x=list(missing.index), y=list(missing.values)),
            row=2, col=1
        )

        # Sample distribution
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Histogram(x=df[numeric_cols[0]], nbinsx=30),
                row=2, col=2
            )

        fig.update_layout(
            title="Parquet Data Dashboard",
            height=800,
            showlegend=False
        )

        fig.write_html("dashboard.html")
        print("Dashboard saved to: dashboard.html")
        fig.show()

    def query_data(self, file_path: str, query: str, output_file: Optional[str] = None):
        """Query parquet data with pandas query syntax"""
        df = self.load_parquet(file_path)
        if df is None:
            return

        try:
            result = df.query(query)
            print(f"Query returned {len(result)} rows out of {len(df)} total rows")

            if len(result) > 0:
                print("\nFirst 10 rows of results:")
                print(result.head(10).to_string())

                if output_file:
                    if output_file.endswith('.parquet'):
                        result.to_parquet(output_file)
                    elif output_file.endswith('.csv'):
                        result.to_csv(output_file, index=False)
                    else:
                        result.to_parquet(output_file + '.parquet')
                    print(f"\nResults saved to: {output_file}")
            else:
                print("No rows matched the query")

        except Exception as e:
            print(f"Error executing query: {e}")
            print("Make sure to use valid pandas query syntax")


def main():
    parser = argparse.ArgumentParser(description='Parquet Data Analysis Tool')
    parser.add_argument('--file', type=str, help='Single parquet file to analyze')
    parser.add_argument('--files', type=str, help='Comma-separated list of files to merge')
    parser.add_argument('--directory', type=str, help='Directory containing parquet files')
    parser.add_argument('--action', type=str, required=True,
                       choices=['analyze', 'analyze-all', 'to-csv', 'to-json',
                               'merge', 'stats', 'visualize', 'query'],
                       help='Action to perform')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument('--orient', type=str, default='records',
                       choices=['records', 'table', 'split', 'index'],
                       help='JSON orientation')
    parser.add_argument('--report-type', type=int, default=1,
                       help='Report type (1-4, higher = more detailed)')
    parser.add_argument('--viz-type', type=int, default=1,
                       help='Visualization type (1-5)')
    parser.add_argument('--query', type=str, help='Query string for filtering data')
    parser.add_argument('--save', type=str, help='Save report to file')

    args = parser.parse_args()

    analyzer = ParquetAnalyzer()

    if args.action == 'analyze':
        if args.file:
            result = analyzer.analyze_file(args.file)
            if result:
                print(json.dumps(result, indent=2, default=str))

    elif args.action == 'analyze-all':
        directory = args.directory or '.'
        parquet_files = list(Path(directory).glob('*.parquet'))

        print(f"Found {len(parquet_files)} parquet files in {directory}")
        for file in parquet_files:
            print(f"\n{'='*60}")
            print(f"Analyzing: {file.name}")
            print('='*60)
            result = analyzer.analyze_file(str(file))
            if result:
                print(f"Shape: {result['shape']}")
                print(f"Columns: {len(result['columns'])}")
                print(f"Memory: {result['memory_usage']:.2f} MB")
                print(f"File Size: {result['file_info']['size_mb']:.2f} MB")

    elif args.action == 'to-csv':
        if args.file and args.output:
            analyzer.convert_to_csv(args.file, args.output)

    elif args.action == 'to-json':
        if args.file and args.output:
            analyzer.convert_to_json(args.file, args.output, args.orient)

    elif args.action == 'merge':
        if args.files and args.output:
            file_list = args.files.split(',')
            analyzer.merge_parquet_files(file_list, args.output)

    elif args.action == 'stats':
        if args.file:
            report = analyzer.generate_stats_report(args.file, args.report_type)
            print(report)

            if args.save:
                with open(args.save, 'w') as f:
                    f.write(report)
                print(f"\nReport saved to: {args.save}")

    elif args.action == 'visualize':
        if args.file:
            analyzer.visualize_data(args.file, args.viz_type)

    elif args.action == 'query':
        if args.file and args.query:
            analyzer.query_data(args.file, args.query, args.output)


if __name__ == "__main__":
    main()