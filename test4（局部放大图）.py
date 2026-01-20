import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# ==============================================================================
# 1. 架构配置 (Configuration)
# ==============================================================================
class AppConfig:
    """
    视觉规范与全局参数配置
    """
    # 路径配置 - 修正为新路径
    DATA_PATH = r"C:\Users\Peter\Desktop\Dow Jones Industrial Average.csv"

    # 调色板 (Deep Logical Palette)
    COLOR_BULLISH = "rgba(46, 204, 113, 0.4)"  # 翡翠绿，带透明度
    COLOR_BEARISH = "rgba(231, 76, 60, 0.4)"  # 珊瑚红，带透明度
    COLOR_MA_LINE = "#2C3E50"  # 深岩灰
    COLOR_PRICE_LINE = "#95A5A6"  # 混凝土灰

    # 字体配置 (中西文混排)
    FONT_FAMILY = "Microsoft YaHei, Arial, sans-serif"

    # 布局背景
    BG_COLOR = "#FAEDDA"  # 暖米色
    CARD_COLOR = "#FFFFFF"


# ==============================================================================
# 2. 数据引擎 (Data Engine)
# ==============================================================================
def load_and_process_data(file_path):
    """
    读取CSV文件 - 适配新的数据格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 找不到文件 {file_path}")

    # 读取CSV - 根据图片，第一行是表头
    df = pd.read_csv(file_path)
    print(f"✓ 成功加载数据，共 {len(df)} 条记录")
    print(f"✓ 列名: {df.columns.tolist()}")

    return df


def clean_data(df):
    """
    清洗数据，计算辅助列
    """
    # 1. 规范化列名 (去除前后空格)
    df.columns = [c.strip() for c in df.columns]

    # 2. 根据图片数据结构，列应该是：
    # 列A是空(Unnamed)，列B是日期，列C是Dow Jones值，列D是1 year moving average
    # 但read_csv会自动识别第一行为表头

    # 检查列名并重命名
    print(f"原始列名: {df.columns.tolist()}")

    # 假设结构：第一列可能是Unnamed，第二列是日期相关
    # 更稳健的方法：通过位置索引
    if len(df.columns) >= 3:
        # 重命名列以便后续使用
        new_columns = {
            df.columns[0]: 'Date',  # 第一列：日期
            df.columns[1]: 'Value',  # 第二列：道琼斯指数
            df.columns[2]: 'MA'  # 第三列：移动平均
        }
        df = df.rename(columns=new_columns)
    else:
        raise ValueError("CSV格式错误：列数少于3列")

    # 3. 转换日期
    df['Date'] = pd.to_datetime(df['Date'])

    # 4. 确保数值列是数字类型
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['MA'] = pd.to_numeric(df['MA'], errors='coerce')

    # 5. 删除缺失值
    df = df.dropna(subset=['Date', 'Value', 'MA'])

    # 6. 核心逻辑：计算用于填充颜色的"上方"和"下方"序列
    df['Upper_Bound'] = np.maximum(df['Value'], df['MA'])
    df['Lower_Bound'] = np.minimum(df['Value'], df['MA'])

    print(f"✓ 数据清洗完成，有效记录: {len(df)} 条")
    print(f"✓ 日期范围: {df['Date'].min()} 至 {df['Date'].max()}")

    return df.sort_values('Date').reset_index(drop=True)


# ==============================================================================
# 3. 视觉构建工厂 (Chart Factory)
# ==============================================================================
def create_financial_chart(df, y_range=None):
    """
    构建差异面积图 (Difference Chart)
    """
    fig = go.Figure()

    # 1. 绘制基准线 (移动平均线)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA'],
        mode='lines',
        line=dict(color=AppConfig.COLOR_MA_LINE, width=1.5),
        name='1年移动平均',
        hoverinfo='skip'
    ))

    # 2. 绘制"多头"区域 (Price > MA)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Upper_Bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=AppConfig.COLOR_BULLISH,
        name='牛市区域 (高于均线)',
        hoverinfo='skip'
    ))

    # 3. 重新绘制MA作为填充基准
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 4. 绘制"空头"区域 (Price < MA)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Lower_Bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=AppConfig.COLOR_BEARISH,
        name='熊市区域 (低于均线)',
        hoverinfo='skip'
    ))

    # 5. 绘制实际价格线 (覆盖在最上层)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Value'],
        mode='lines',
        line=dict(color=AppConfig.COLOR_PRICE_LINE, width=1),
        name='道琼斯指数'
    ))

    # 6. 美学修饰
    fig.update_layout(
        title="<b>道琼斯工业平均指数：市场情绪分析</b><br>" +
              "<span style='font-size:12px;color:grey'>与1年移动平均线的差异对比</span>",
        title_font=dict(family=AppConfig.FONT_FAMILY, size=20, color="#2C3E50"),
        paper_bgcolor=AppConfig.CARD_COLOR,
        plot_bgcolor=AppConfig.CARD_COLOR,
        font=dict(family=AppConfig.FONT_FAMILY),
        margin=dict(l=60, r=40, t=80, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F0F0F0",
            title="日期"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F0F0F0",
            title="指数点位",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    # 7. 响应 Y 轴控制
    if y_range and len(y_range) == 2 and y_range[0] < y_range[1]:
        fig.update_layout(yaxis_range=y_range)

    return fig


# ==============================================================================
# 4. Dash 应用编排 (Application Orchestration)
# ==============================================================================
def create_app(df):
    app = Dash(__name__)

    # 计算初始统计量
    min_val = int(min(df['Value'].min(), df['MA'].min()) * 0.9)
    max_val = int(max(df['Value'].max(), df['MA'].max()) * 1.1)
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    app.layout = html.Div(style={
        'backgroundColor': AppConfig.BG_COLOR,
        'minHeight': '100vh',
        'padding': '30px',
        'fontFamily': AppConfig.FONT_FAMILY
    }, children=[

        # --- 仪表盘容器 ---
        html.Div(style={
            'backgroundColor': AppConfig.CARD_COLOR,
            'borderRadius': '15px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.05)',
            'padding': '30px',
            'maxWidth': '1200px',
            'margin': '0 auto'
        }, children=[

            # 1. 标题区
            html.H2("金融市场趋势分析器",
                    style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '10px'}),
            html.P("基于道琼斯工业平均指数移动平均线的交互式差异图",
                   style={'textAlign': 'center', 'color': '#7F8C8D', 'marginBottom': '30px'}),

            # 2. 控制面板
            html.Div(style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'marginBottom': '20px',
                'padding': '15px',
                'backgroundColor': '#F8F9F9',
                'borderRadius': '8px'
            }, children=[

                # Y轴控制
                html.Div([
                    html.Label("Y轴范围 (价格):", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Input(id='y-min', type='number', value=min_val,
                              style={'width': '80px', 'marginRight': '5px'}),
                    html.Span("-", style={'marginRight': '5px'}),
                    dcc.Input(id='y-max', type='number', value=max_val,
                              style={'width': '80px'}),
                ]),

                # 提示信息
                html.Div(
                    html.Span("💡 使用下方滑块控制时间范围 (X轴)",
                              style={'color': '#16A085', 'fontSize': '14px'})
                )
            ]),

            # 3. 主图表
            dcc.Graph(id='main-chart'),

            # 4. X轴时间滑块
            html.Div(style={'padding': '0 40px 20px 40px'}, children=[
                html.Label("时间范围导航:",
                           style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
                dcc.RangeSlider(
                    id='date-slider',
                    min=min_date.timestamp(),
                    max=max_date.timestamp(),
                    value=[min_date.timestamp(), max_date.timestamp()],
                    marks={
                        int(pd.Timestamp(f"{year}-01-01").timestamp()): str(year)
                        for year in range(min_date.year, max_date.year + 2)
                    },
                    step=24 * 60 * 60  # 1天步长
                )
            ])
        ])
    ])

    # --- 回调逻辑 ---
    @app.callback(
        Output('main-chart', 'figure'),
        [Input('date-slider', 'value'),
         Input('y-min', 'value'),
         Input('y-max', 'value')]
    )
    def update_chart(date_range_timestamps, y_min, y_max):
        # 1. 解析时间范围
        start_date = pd.Timestamp.fromtimestamp(date_range_timestamps[0])
        end_date = pd.Timestamp.fromtimestamp(date_range_timestamps[1])

        # 2. 过滤数据
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df_filtered = df.loc[mask]

        # 3. 验证 Y 轴输入有效性
        y_range = None
        if y_min is not None and y_max is not None and y_min < y_max:
            y_range = [y_min, y_max]

        # 4. 生成图表
        return create_financial_chart(df_filtered, y_range)

    return app


# ==============================================================================
# 5. 执行入口
# ==============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print(">>> 道琼斯指数可视化分析器启动中...")
    print("=" * 70)

    try:
        # 1. 加载数据
        print("\n[步骤 1/3] 加载数据...")
        raw_df = load_and_process_data(AppConfig.DATA_PATH)

        # 2. 清洗数据
        print("\n[步骤 2/3] 清洗数据...")
        clean_df = clean_data(raw_df)

        # 3. 启动应用
        print("\n[步骤 3/3] 启动Web应用...")
        app = create_app(clean_df)

        print("\n" + "=" * 70)
        print(">>> 应用已就绪")
        print("=" * 70)
        print("   本地访问: http://127.0.0.1:8050/")
        print("   如需公网访问，请在终端运行: ngrok http 8050")
        print("\n   按 Ctrl+C 停止服务器")
        print("=" * 70 + "\n")

        # 修复：使用 app.run 替代 app.run_server
        app.run(debug=True, use_reloader=False)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("   请确保CSV文件存在于指定路径")
    except Exception as e:
        print(f"\n❌ 应用启动失败: {e}")
        import traceback

        traceback.print_exc()