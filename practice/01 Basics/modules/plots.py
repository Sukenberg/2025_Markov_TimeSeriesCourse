import numpy as np
import pandas as pd

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
import cv2

def plot_ts(ts_set: np.ndarray, plot_title: str = 'Input Time Series Set'):
    """
    Plot the time series set

    Parameters
    ----------
    ts_set: time series set with shape (ts_number, ts_length)
    plot_title: title of plot
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=18, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=16, color='black'),
                     linewidth=1,
                     tickwidth=1)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=18, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=16, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1)

    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top'},
                      title_font=dict(size=18, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=16, color='black')),
                      width=1000,
                      height=400
                      )

    fig.show(renderer="browser")


def display_image_plotly(img, contour, edge_coordinates, center):
    """
    Функция для отображения изображения с контурами и координатами используя Plotly
    """
    # Конвертируем BGR в RGB для Plotly
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = go.Figure()

    # Добавляем изображение
    fig.add_trace(go.Image(z=img_rgb))

    # Добавляем линии от центра к краям
    for i in range(len(edge_coordinates)):
        fig.add_trace(go.Scatter(
            x=[center[0], edge_coordinates[i][0]],
            y=[center[1], edge_coordinates[i][1]],
            mode='lines',
            line=dict(color='magenta', width=2),
            showlegend=False
        ))

    # Добавляем центр
    fig.add_trace(go.Scatter(
        x=[center[0]],
        y=[center[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        showlegend=False
    ))

    fig.update_layout(
        title='Image with Contours and Lines',
        width=600,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor='x')
    )

    fig.show(renderer="browser")


def plot_ts_plotly(ts, title="Time Series"):
    """
    Функция для отображения временного ряда используя Plotly
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(len(ts)),
        y=ts,
        mode='lines',
        line=dict(width=3, color='blue')
    ))

    fig.update_xaxes(
        showgrid=False,
        title='Angle (degrees)',
        title_font=dict(size=18, color='black'),
        linecolor='#000',
        ticks="outside",
        tickfont=dict(size=16, color='black'),
        linewidth=1,
        tickwidth=1
    )

    fig.update_yaxes(
        showgrid=False,
        title='Distance',
        title_font=dict(size=18, color='black'),
        linecolor='#000',
        ticks="outside",
        tickfont=dict(size=16, color='black'),
        zeroline=False,
        linewidth=1,
        tickwidth=1
    )

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top'},
        title_font=dict(size=18, color='black'),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor='rgba(0,0,0,0)',
        width=1000,
        height=400
    )

    fig.show(renderer="browser")