# -*- coding: utf-8 -*-
"""
产品名称: 鸢尾花多维数据探索器（工业级）
架构师: 顶级数据可视化架构师
日期: 2026-01-18
依赖库: pandas, plotly, dash, pyngrok, kaleido
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from pyngrok import conf, ngrok
import io


# ==============================================================================
# 1. 配置类（视觉设计规范）
# ==============================================================================
class VisualConfig:
    """
    集中式美学和排版配置。
    执行"浅色模式策略"和"排版策略"。
    """
    # 画布颜色
    BG_COLOR = "#FFFFFF"  # 纯白色用于图表
    DASH_BG = "#FAEDDA"  # 温暖米色用于仪表板背景

    # 逻辑调色板（低饱和度，高区分度）
    # Setosa（叶绿色），Versicolor（花卉色），Virginica（深色）
    COLOR_MAP = {
        "Iris-setosa": "#6CA9A3",  # 柔和青色
        "Iris-versicolor": "#E69F00",  # 赭黄色
        "Iris-virginica": "#5D6D7E"  # 板岩蓝
    }

    # 排版策略
    FONTS = {
        "title": "Times New Roman, serif",  # 学术/正式
        "ui": "Arial, sans-serif",  # 屏幕/现代
        "code": "Courier New, monospace"  # 数据/注释
    }

    # 导出设置
    EXPORT_DIR = os.path.join(os.getcwd(), "deliverables")


# ==============================================================================
# 2. 数据管理器（鲁棒数据握手）
# ==============================================================================
class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._load_data()

    def _load_data(self):
        """
        加载数据并进行严格验证。
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[严重错误] 数据文件未找到: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
            # 基础验证
            required_cols = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("数据集缺少必需列。")

            # 清理列名以便显示
            df.columns = [c.replace('Iris-', '') if 'Iris' in c else c for c in df.columns]
            return df
        except Exception as e:
            raise RuntimeError(f"数据导入失败: {str(e)}")

    def get_correlations(self):
        """
        自动洞察：计算数值型列的相关系数。
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()


# ==============================================================================
# 3. 图表架构师（核心可视化产品）
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
        """
        构建4x4散点图矩阵（SPLOM）。
        每个子图拥有独立的坐标系。
        """
        # 计算洞察（最强相关性）用于副标题
        corr = self.df[self.dimensions].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()

        # 创建子图网格 - 使用make_subplots实现独立坐标系
        from plotly.subplots import make_subplots

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

        # 为每个物种准备颜色
        species = self.df['Name'].unique()

        # 填充每个子图
        for i, dim_y in enumerate(self.dimensions):
            for j, dim_x in enumerate(self.dimensions):
                row = i + 1
                col = j + 1

                if i == j:
                    # 对角线：绘制直方图
                    for spec in species:
                        df_spec = self.df[self.df['Name'] == spec]
                        fig.add_trace(
                            go.Histogram(
                                x=df_spec[dim_x],
                                name=spec,
                                marker_color=VisualConfig.COLOR_MAP.get(spec, "#666666"),
                                opacity=0.7,
                                showlegend=(i == 0 and j == 0),  # 只在第一个子图显示图例
                                legendgroup=spec
                            ),
                            row=row, col=col
                        )
                else:
                    # 非对角线：绘制散点图
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
                                showlegend=(i == 0 and j == 1),  # 只在第一行第二列显示图例
                                legendgroup=spec
                            ),
                            row=row, col=col
                        )

                # 设置每个子图的轴标签
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

        # 全局布局设置
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

    def export_assets(self, fig, filename="iris_matrix"):
        """
        后期制作资源协议: SVG & HTML导出
        """
        if not os.path.exists(VisualConfig.EXPORT_DIR):
            os.makedirs(VisualConfig.EXPORT_DIR)

        # 1. 交互式输出
        html_path = os.path.join(VisualConfig.EXPORT_DIR, f"{filename}.html")
        fig.write_html(html_path)
        print(f"   [资源] 交互式HTML已生成: {html_path}")

        # 2. 矢量图输出（用于PS/AI）
        try:
            svg_path = os.path.join(VisualConfig.EXPORT_DIR, f"{filename}.svg")
            fig.write_image(svg_path)
            print(f"   [资源] 矢量SVG已生成: {svg_path}")
        except Exception as e:
            print(f"   [警告] SVG导出跳过（缺少Kaleido?）: {e}")


# ==============================================================================
# 4. 产品交付（Dash应用 & Ngrok）
# ==============================================================================
def launch_product(df, fig):
    """
    将可视化封装到Dash应用中，包含数据表组件。
    """
    app = Dash(__name__)

    # 布局设计（温暖米色背景）
    app.layout = html.Div(
        style={'backgroundColor': VisualConfig.DASH_BG, 'padding': '40px', 'fontFamily': VisualConfig.FONTS['ui']},
        children=[

            # 标题栏
            html.H1("鸢尾花数据集分析产品",
                    style={'textAlign': 'center', 'fontFamily': VisualConfig.FONTS['title'], 'color': '#2C3E50'}),
            html.Hr(style={'borderColor': '#888888'}),

            # 模块1：可视化矩阵
            html.Div([
                dcc.Graph(figure=fig)
            ], style={'backgroundColor': '#FFFFFF', 'padding': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                      'marginBottom': '30px'}),

            # 模块2：数据组件（列表）
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
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    filter_action="native",
                    sort_action="native"
                )
            ], style={'backgroundColor': '#FFFFFF', 'padding': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        ])

    return app


# ==============================================================================
# Gunicorn入口
# ==============================================================================
def create_app():
    """应用工厂函数 - 供Gunicorn使用"""
    # 数据握手（使用相对路径）
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "iris.csv")
    manager = DataManager(DATA_PATH)

    # 构建图表
    architect = ChartArchitect(manager.df)
    figure = architect.build_scatter_matrix()

    # 启动应用
    app = launch_product(manager.df, figure)
    return app.server  # 返回Flask server实例


# Gunicorn会使用这个server对象
server = create_app()


# ==============================================================================
# 主执行流程（仅用于本地调试）
# ==============================================================================
if __name__ == "__main__":
    print(">>> 初始化可视化产品架构...")

    # 1. 配置Ngrok（远程共享协议）
    conf.get_default().ngrok_path = r"E:\ngrok\ngrok.exe"

    try:
        ngrok.kill()  # 清理旧进程
        public_url = ngrok.connect(8050).public_url
        print(f">>> [远程协议] 隧道已建立: {public_url}")
    except Exception as e:
        print(f">>> [警告] Ngrok初始化失败（检查路径/令牌）: {e}")
        print("    继续使用本地部署...")

    # 2. 数据握手（使用相对路径）
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "iris.csv")
    manager = DataManager(DATA_PATH)

    # 3. 构建图表
    architect = ChartArchitect(manager.df)
    figure = architect.build_scatter_matrix()
    architect.export_assets(figure)

    # 4. 启动应用
    app = launch_product(manager.df, figure)
    print(">>> [产品就绪] 在端口8050提供服务...")
    app.run(debug=False, port=8050)