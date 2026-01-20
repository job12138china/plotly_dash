import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import numpy as np


# ==============================================================================
# 1. 配置中心 (Configuration) - 单一事实来源
# ==============================================================================
class VisualConfig:
    """
    视觉设计规范与系统配置
    """
    # 文件路径配置
    DATA_PATH = r"C:\Users\Peter\Desktop\比赛\数学建模\美赛加油\plotly_examples-master\data\earthquakes.csv"
    SAVE_DIR = os.getcwd()  # 默认保存至当前脚本目录

    # 调色板策略：使用 perceptually uniform 的颜色，高震级用暖色
    COLOR_SCALE = "Inferno_r"

    # 字体工程：中西文混排策略
    FONT_FAMILY_CN = "Microsoft YaHei"
    FONT_FAMILY_EN = "Arial"

    # 布局规范
    BG_COLOR = "#FAFAFA"  # 极淡的暖灰，比纯白更护眼
    CARD_COLOR = "#FFFFFF"  # 卡片背景纯白
    TEXT_COLOR = "#2C3E50"  # 深岩灰，替代纯黑

    # 地图风格
    MAP_STYLE = "carto-positron"  # 极简浅色地图，突出数据点


# ==============================================================================
# 2. 数据工厂 (Data Factory) - 健壮的数据处理层
# ==============================================================================
class EarthquakeDataLoader:
    """
    负责数据的读取、清洗与特征工程
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.clean_data = None

    def load_and_clean(self):
        """
        读取并清洗数据，处理混合的时间格式
        """
        try:
            # 1. 读取数据
            print(f"正在读取数据: {self.file_path} ...")
            df = pd.read_csv(self.file_path)

            # 2. 时间格式标准化 (处理混合格式)
            # 数据中包含 '01/02/1965' 和 '1975-02-23T02:58:41.000Z'
            # 使用 errors='coerce' 将无法解析的转为 NaT，然后清洗
            df['Datetime'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')

            # 3. 剔除无效时间数据
            initial_count = len(df)
            df = df.dropna(subset=['Datetime'])
            if len(df) < initial_count:
                print(f"警告: 剔除了 {initial_count - len(df)} 条时间格式错误的记录。")

            # 4. 特征提取：提取年份用于Dash的时间轴滑块
            df['Year'] = df['Datetime'].dt.year

            # 5. 排序
            df = df.sort_values(by='Datetime')

            self.clean_data = df
            print("数据清洗完成。")
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"错误: 找不到文件，请检查路径: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"数据处理过程中发生未知错误: {e}")


# ==============================================================================
# 3. 视觉构建器 (Visual Architect) - 封装绘图逻辑
# ==============================================================================
class ChartFactory:
    """
    生产符合设计规范的 Plotly 图表对象
    """

    @staticmethod
    def create_global_map(df):
        """
        构建全球震级分布地图（使用新版scatter_map）
        设计意图：使用气泡大小映射震级，颜色映射深度或强度，让视觉重心集中在环太平洋地震带。
        """
        # 自动洞察：找到最大震级
        max_quake = df.loc[df['Magnitude'].idxmax()]

        # 修复：使用新版 scatter_map 替代 scatter_mapbox
        fig = px.scatter_map(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Magnitude",
            size="Magnitude",
            hover_name="Date",
            hover_data=["Magnitude", "Latitude", "Longitude"],
            color_continuous_scale=VisualConfig.COLOR_SCALE,
            size_max=15,  # 控制最大气泡尺寸，防止遮挡
            zoom=1,
            map_style=VisualConfig.MAP_STYLE,
            title=f"全球地震震级分布 ({df['Year'].min()}-{df['Year'].max()})"
        )

        # 注入自动洞察：标注最大地震
        fig.add_annotation(
            x=0.5, y=0.95,  # 相对坐标
            text=f"历史最强震级: {max_quake['Magnitude']} (年份: {max_quake['Year']})",
            showarrow=False,
            font=dict(family=VisualConfig.FONT_FAMILY_CN, size=14, color="#C0392B"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#C0392B",
            borderwidth=1
        )

        ChartFactory._apply_theme(fig)
        return fig

    @staticmethod
    def create_magnitude_hist(df):
        """
        构建震级分布直方图
        设计意图：展示地震发生的频率与强度的关系（通常遵循幂律分布）。
        """
        fig = px.histogram(
            df,
            x="Magnitude",
            nbins=30,
            title="震级频率分布统计",
            color_discrete_sequence=["#34495E"]  # 使用稳重的深色
        )

        # 添加平均线
        mean_mag = df['Magnitude'].mean()
        fig.add_vline(x=mean_mag, line_width=2, line_dash="dash", line_color="#E74C3C",
                      annotation_text=f"平均震级: {mean_mag:.2f}")

        ChartFactory._apply_theme(fig)
        fig.update_layout(xaxis_title="震级 (Magnitude)", yaxis_title="发生频次 (Count)")
        return fig

    @staticmethod
    def _apply_theme(fig):
        """
        统一应用设计规范（字体、背景、边距）
        """
        fig.update_layout(
            font=dict(family=f"{VisualConfig.FONT_FAMILY_CN}, {VisualConfig.FONT_FAMILY_EN}", size=12),
            paper_bgcolor=VisualConfig.CARD_COLOR,
            plot_bgcolor=VisualConfig.CARD_COLOR,
            font_color=VisualConfig.TEXT_COLOR,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_colorbar=dict(title="震级")
        )


# ==============================================================================
# 4. Dash 应用编排 (App Orchestration) - 交互层
# ==============================================================================
def launch_dashboard(df):
    """
    启动 Dash 数据可视化产品
    """
    app = Dash(__name__)

    # 获取年份范围
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())

    # 布局设计 (采用 CSS Flexbox 思想)
    app.layout = html.Div(
        style={'backgroundColor': VisualConfig.BG_COLOR, 'fontFamily': VisualConfig.FONT_FAMILY_CN, 'padding': '20px'},
        children=[

            # --- 标题栏 ---
            html.Div([
                html.H1("全球地震数据可视化分析平台", style={'textAlign': 'center', 'color': VisualConfig.TEXT_COLOR}),
                html.P("Global Earthquake Analysis System | Powered by Python Dash",
                       style={'textAlign': 'center', 'color': '#7F8C8D'}),
            ], style={'marginBottom': '30px'}),

            # --- 控制区 (Controls) ---
            html.Div([
                html.Label("请选择观测时间跨度 (Year Range):", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='year-slider',
                    min=min_year,
                    max=max_year,
                    value=[min_year, max_year],
                    marks={str(year): str(year) for year in range(min_year, max_year + 1, 5)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'backgroundColor': VisualConfig.CARD_COLOR, 'padding': '20px', 'borderRadius': '10px',
                      'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),

            # --- 图表区 (Charts) ---
            html.Div([
                # 左侧：地图
                html.Div([
                    dcc.Graph(id='map-graph')
                ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

                # 右侧：统计图
                html.Div([
                    dcc.Graph(id='hist-graph'),
                    html.Div(id='stats-card',
                             style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#ECF0F1',
                                    'borderRadius': '5px'})
                ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'marginTop': '20px'})
        ])

    # --- 回调逻辑 (Interaction Logic) ---
    @app.callback(
        [Output('map-graph', 'figure'),
         Output('hist-graph', 'figure'),
         Output('stats-card', 'children')],
        [Input('year-slider', 'value')]
    )
    def update_charts(year_range):
        # 1. 数据切片
        filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

        # 边界情况处理：如果筛选为空
        if filtered_df.empty:
            return {}, {}, "所选范围内无数据"

        # 2. 生成图表
        map_fig = ChartFactory.create_global_map(filtered_df)
        hist_fig = ChartFactory.create_magnitude_hist(filtered_df)

        # 3. 生成统计卡片
        stats_info = [
            html.H4("当前区间统计概要"),
            html.P(f"地震总次数: {len(filtered_df)} 次"),
            html.P(f"最大震级: {filtered_df['Magnitude'].max()}"),
            html.P(f"平均震级: {filtered_df['Magnitude'].mean():.2f}")
        ]

        return map_fig, hist_fig, stats_info

    print(">>> 系统启动中: 请访问 http://127.0.0.1:8050/ 查看可视化产品")
    print(">>> 如需公网访问，请在终端运行: ngrok http 8050")

    # 修复：使用新版 app.run 替代 app.run_server
    app.run(debug=True, use_reloader=False)


# ==============================================================================
# 5. 主程序入口 (Main Execution)
# ==============================================================================
if __name__ == "__main__":
    # 实例化加载器
    loader = EarthquakeDataLoader(VisualConfig.DATA_PATH)

    try:
        # 加载数据
        df_clean = loader.load_and_clean()

        # 导出静态 HTML 报告 (交付思维)
        # 为那些不方便运行 Dash 服务器的团队成员准备
        print("正在生成静态 HTML 报告快照...")
        static_fig = ChartFactory.create_global_map(df_clean)
        static_output_path = os.path.join(VisualConfig.SAVE_DIR, "earthquake_global_snapshot.html")
        static_fig.write_html(static_output_path)
        print(f"静态报告已生成: {static_output_path}")

        # 启动交互式 Dash 应用
        launch_dashboard(df_clean)

    except Exception as e:
        print(f"程序执行失败: {e}")