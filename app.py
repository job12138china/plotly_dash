# -*- coding: utf-8 -*-
"""
产品名称: 鸢尾花多维数据探索器（生产级）
架构师: 顶级数据可视化架构师
日期: 2026-01-21
依赖库: pandas, plotly, dash
环境兼容: Render / Heroku / 本地
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from plotly.subplots import make_subplots


# ==============================================================================
# 1. 配置类（视觉设计规范）
# ==============================================================================
class VisualConfig:
    """集中式美学和排版配置"""
    BG_COLOR = "#FFFFFF"
    DASH_BG = "#FAEDDA"

    COLOR_MAP = {
        "Iris-setosa": "#6CA9A3",
        "Iris-versicolor": "#E69F00",
        "Iris-virginica": "#5D6D7E"
    }

    FONTS = {
        "title": "Times New Roman, serif",
        "ui": "Arial, sans-serif",
        "code": "Courier New, monospace"
    }


# ==============================================================================
# 2. 数据管理器
# ==============================================================================
class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._load_data()

    def _load_data(self):
        """加载并验证数据"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"数据文件未找到: {self.file_path}")

        df = pd.read_csv(self.file_path)
        required_cols = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("数据集缺少必需列")

        df.columns = [c.replace('Iris-', '') if 'Iris' in c else c for c in df.columns]
        return df


# ==============================================================================
# 3. 图表架构师
# ==============================================================================
class ChartArchitect:
    def __init__(self, df):
        self.df = df
        self.dimensions = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
        self.dim_labels = {
            'SepalLength': '花萼长度',
            'SepalWidth': '花萼宽度',
            'PetalLength': '花瓣长度',
            'PetalWidth': '花瓣宽度'
        }

    def build_scatter_matrix(self):
        """构建4x4散点图矩阵（SPLOM）"""
        corr = self.df[self.dimensions].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()

        n_dims = len(self.dimensions)
        fig = make_subplots(
            rows=n_dims,
            cols=n_dims,
            subplot_titles=[f"{self.dim_labels[self.dimensions[i]]} vs {self.dim_labels[self.dimensions[j]]}"
                            if i != j else f"{self.dim_labels[self.dimensions[i]]} 分布"
                            for i in range(n_dims) for j in range(n_dims)],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )

        species = self.df['Name'].unique()

        for i, dim_y in enumerate(self.dimensions):
            for j, dim_x in enumerate(self.dimensions):
                row = i + 1
                col = j + 1

                if i == j:
                    # 对角线：直方图
                    for spec in species:
                        df_spec = self.df[self.df['Name'] == spec]
                        fig.add_trace(
                            go.Histogram(
                                x=df_spec[dim_x],
                                name=spec,
                                marker_color=VisualConfig.COLOR_MAP.get(spec, "#666666"),
                                opacity=0.7,
                                showlegend=(i == 0 and j == 0),
                                legendgroup=spec
                            ),
                            row=row, col=col
                        )
                else:
                    # 非对角线：散点图
                    for spec in species:
                        df_spec = self.df[self.df['Name'] == spec]
                        fig.add_trace(
                            go.Scatter(
                                x=df_spec[dim_x],
                                y=df_spec[dim_y],
                                mode='markers',
                                name=spec,
                                marker=dict(
                                    color=VisualConfig.COLOR_MAP.get(spec, "#666666"),
                                    size=6,
                                    opacity=0.7
                                ),
                                showlegend=(i == 0 and j == 1),
                                legendgroup=spec
                            ),
                            row=row, col=col
                        )

                fig.update_xaxes(
                    title_text=self.dim_labels[dim_x] if i == n_dims - 1 else "",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#F0F0F0",
                    linecolor="#333333",
                    tickfont=dict(family=VisualConfig.FONTS["code"], size=9),
                    row=row, col=col
                )
                fig.update_yaxes(
                    title_text=self.dim_labels[dim_y] if j == 0 else "",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#F0F0F0",
                    linecolor="#333333",
                    tickfont=dict(family=VisualConfig.FONTS["code"], size=9),
                    row=row, col=col
                )

        fig.update_layout(
            title=dict(
                text=f"<b>鸢尾花形态矩阵</b><br><span style='font-size:14px; font-family:{VisualConfig.FONTS['ui']}'>洞察: 最大维度相关性 = {max_corr:.2f}</span>",
                font=dict(family=VisualConfig.FONTS["title"], size=24)
            ),
            height=1000,
            width=1200,
            plot_bgcolor=VisualConfig.BG_COLOR,
            paper_bgcolor=VisualConfig.BG_COLOR,
            legend=dict(
                title=dict(text="物种", font=dict(family=VisualConfig.FONTS["title"])),
                font=dict(family=VisualConfig.FONTS["code"]),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E5E5E5",
                borderwidth=1,
                x=1.02,
                y=1
            ),
            margin=dict(l=60, r=150, t=100, b=60)
        )

        return fig


# ==============================================================================
# 4. Dash应用构建
# ==============================================================================
def create_dash_app(df, fig):
    """创建Dash应用实例"""
    app = Dash(__name__)

    app.layout = html.Div(
        style={'backgroundColor': VisualConfig.DASH_BG, 'padding': '40px', 'fontFamily': VisualConfig.FONTS['ui']},
        children=[
            html.H1("鸢尾花数据集分析产品",
                    style={'textAlign': 'center', 'fontFamily': VisualConfig.FONTS['title'], 'color': '#2C3E50'}),
            html.Hr(style={'borderColor': '#888888'}),

            html.Div([
                dcc.Graph(figure=fig)
            ], style={'backgroundColor': '#FFFFFF', 'padding': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                      'marginBottom': '30px'}),

            html.Div([
                html.H3("原始数据检查（可排序 & 可过滤）", style={'fontFamily': VisualConfig.FONTS['title']}),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'fontFamily': VisualConfig.FONTS['code'],
                        'textAlign': 'left',
                        'padding': '10px'
                    },
                    style_header={
                        'backgroundColor': '#E5E5E5',
                        'fontWeight': 'bold',
                        'fontFamily': VisualConfig.FONTS['ui']
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                    ],
                    filter_action="native",
                    sort_action="native"
                )
            ], style={'backgroundColor': '#FFFFFF', 'padding': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        ])

    return app


# ==============================================================================
# Gunicorn入口（生产环境）
# ==============================================================================
def create_app():
    """应用工厂函数 - 供Gunicorn/Render使用"""
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "iris.csv")

    manager = DataManager(DATA_PATH)
    architect = ChartArchitect(manager.df)
    figure = architect.build_scatter_matrix()

    app = create_dash_app(manager.df, figure)
    return app.server


# Gunicorn使用此对象
server = create_app()

# ==============================================================================
# 本地开发入口（仅本地调试）
# ==============================================================================
if __name__ == "__main__":
    print(">>> 启动本地开发服务器...")

    # 本地专属：ngrok隧道（生产环境不执行）
    try:
        from pyngrok import conf, ngrok

        conf.get_default().ngrok_path = r"E:\ngrok\ngrok.exe"
        ngrok.kill()
        public_url = ngrok.connect(8050).public_url
        print(f">>> [远程隧道] {public_url}")
    except ImportError:
        print(">>> [本地模式] pyngrok未安装，仅本地访问")
    except Exception as e:
        print(f">>> [警告] Ngrok初始化失败: {e}")

    # 数据加载
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "iris.csv")
    manager = DataManager(DATA_PATH)
    architect = ChartArchitect(manager.df)
    figure = architect.build_scatter_matrix()

    # 本地专属：导出静态资源
    try:
        export_dir = os.path.join(os.getcwd(), "deliverables")
        os.makedirs(export_dir, exist_ok=True)

        html_path = os.path.join(export_dir, "iris_matrix.html")
        figure.write_html(html_path)
        print(f">>> [导出] HTML已生成: {html_path}")

        # 尝试SVG（需要kaleido）
        try:
            svg_path = os.path.join(export_dir, "iris_matrix.svg")
            figure.write_image(svg_path)
            print(f">>> [导出] SVG已生成: {svg_path}")
        except Exception:
            print(">>> [跳过] SVG导出需要kaleido")
    except Exception as e:
        print(f">>> [警告] 资源导出失败: {e}")

    # 启动本地服务器
    app = create_dash_app(manager.df, figure)
    print(">>> [就绪] http://localhost:8050")
    app.run(debug=True, port=8050)