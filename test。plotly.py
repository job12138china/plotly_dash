import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import os

# ==========================================
# 1. 启动设置 (Launch Config)
# ==========================================
if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        try:
            from pyngrok import ngrok, conf

            # --- 【你的关键修正】 ---
            # 强制指定 ngrok.exe 的绝对路径，绕过网络下载
            # 注意：前面的 'r' 表示原始字符串，防止反斜杠被转义
            conf.get_default().ngrok_path = r"E:\ngrok\ngrok.exe"

            # 杀掉所有残留进程
            ngrok.kill()

            # 建立隧道
            tunnel = ngrok.connect(8050)

            print("\n" + "=" * 60)
            print(f"   >>> 交付链接 (发给队友): {tunnel.public_url} <<<")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"Ngrok 启动失败: {e}")
            print("请检查 ngrok.exe 路径是否正确")

# ==========================================
# 2. 设计规范 (Design System)
# ==========================================
COLOR_BG = '#FAEDDA'
COLOR_MAIN = '#3C2F80'
COLOR_ACCENT = '#26B6C6'
FONT_PRIMARY = "Times New Roman"
FONT_FUNC = "Arial"
FONT_CODE = "Courier New"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# ==========================================
# 3. 核心逻辑 (Simulation Logic)
# ==========================================
def generate_simulation_data(amplitude, frequency, decay):
    t = np.linspace(0, 10, 500)
    y = amplitude * np.exp(-decay * t) * np.cos(frequency * t)
    return t, y


# ==========================================
# 4. 界面布局 (Layout)
# ==========================================
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Model Sensitivity Cockpit",
                        style={'font-family': FONT_PRIMARY, 'color': COLOR_MAIN, 'font-weight': 'bold',
                               'margin-top': '20px'}),
                width=12)
    ]),
    html.Hr(style={'border-top': f'2px solid {COLOR_MAIN}'}),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Controls", style={'font-family': FONT_PRIMARY, 'color': COLOR_MAIN}),
                html.Label("Amplitude (A)", style={'font-family': FONT_FUNC}),
                dcc.Slider(min=1, max=10, step=0.1, value=5, id='slider-amp'),
                html.Label("Frequency (ω)", style={'font-family': FONT_FUNC}),
                dcc.Slider(min=1, max=20, step=0.5, value=10, id='slider-freq'),
                html.Label("Decay (λ)", style={'font-family': FONT_FUNC}),
                dcc.Slider(min=0, max=1, step=0.1, value=0.2, id='slider-decay'),
            ], style={'background-color': 'white', 'padding': '20px', 'border-radius': '10px'})
        ], width=4),
        dbc.Col([
            dcc.Graph(id='main-graph', style={'height': '600px'})
        ], width=8)
    ])
], fluid=True, style={'background-color': COLOR_BG, 'min-height': '100vh', 'padding': '20px'})


# ==========================================
# 5. 回调 (Callbacks)
# ==========================================
@callback(
    Output('main-graph', 'figure'),
    [Input('slider-amp', 'value'), Input('slider-freq', 'value'), Input('slider-decay', 'value')]
)
def update_graph(amp, freq, decay):
    t, y = generate_simulation_data(amp, freq, decay)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', line=dict(color=COLOR_MAIN, width=2), fill='tozeroy',
                             fillcolor='rgba(60, 47, 128, 0.1)'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.5)',
        font=dict(family=FONT_PRIMARY, color=COLOR_MAIN),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(family=FONT_FUNC)),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(family=FONT_FUNC))
    )
    return fig


# ==========================================
# 6. 正式运行
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8050, use_reloader=False)