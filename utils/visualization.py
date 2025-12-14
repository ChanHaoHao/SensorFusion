import plotly.graph_objects as go
import numpy as np

PCD_SCENE=dict(
    xaxis=dict(visible=False,range=[0,70]),
    yaxis=dict(visible=False,range=[-40,40]),
    zaxis=dict(visible=False,),
    aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
    aspectratio=dict(x=1, y=1, z=0.05),
)

PCD_CAM_VIEW = dict(
    up=dict(x=0, y=0, z=1),
    eye=dict(x=-0.8, y=0, z=0.25)
)

def create_rgb_image(image):
    W, H = image.size[0], image.size[1]  # (W, H)
    
    rgb_fig = go.Figure(go.Image(z=image))

    rgb_fig.update_xaxes(range=[0, 1242], constrain='range', visible=False, fixedrange=True)
    rgb_fig.update_yaxes(range=[375, 0], constrain='range', visible=False, scaleanchor='x', fixedrange=True)

    rgb_fig.update_layout(
        width=W,
        height=H,
        margin=dict(l=0, r=0, t=0, b=0),  # remove all margins
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    
    return rgb_fig

def rgb_add_2dtrace(fig, x, y, color, marker_size=3, mode='markers', name=None, opacity=0.8):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name,
            marker=dict(color=color, size=marker_size, opacity=opacity),
        )
    )
    
    return fig

def create_pc_figure(xyz, mode='markers', marker_size=1, marker_opacity=0.8, color_scale="blackbody"):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode=mode,
                marker=dict(
                    size=marker_size,
                    opacity=marker_opacity,
                    color=xyz[:, 0],     # you can also use reflectance: points[:, 3]
                    colorscale=color_scale,
                ),
            )
        ]
    )
    
    fig.update_layout(
        template="plotly_dark",
        scene=PCD_SCENE,
        scene_camera=PCD_CAM_VIEW,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
    )
    
    return fig

def pc_add_3dtrace(fig, xyz, color, mode='markers', marker_size=1, marker_opacity=0.8, name=None):
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode=mode,
            name=name,
            marker=dict(
                size=marker_size,
                opacity=marker_opacity,
                color=color,
            ),
        )
    )
    
    return fig

def plot_boxes_plotly(fig, corners_list, color="green", name="box"):
    traces = []
    for corners in corners_list:
        c = corners  # (8,3)

        # edges (12 edges)
        edges = [
            [0,1],[1,2],[2,3],[3,0],  # bottom rectangle
            [4,5],[5,6],[6,7],[7,4],  # top rectangle
            [0,4],[1,5],[2,6],[3,7],  # vertical lines
        ]

        xs, ys, zs = [], [], []
        for s,e in edges:
            xs += [c[s,0], c[e,0], None]
            ys += [c[s,1], c[e,1], None]
            zs += [c[s,2], c[e,2], None]

        trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color=color, width=5),
        )
        fig.add_trace(trace)

    return fig

def plot_boxes_on_image_plotly(fig, corners_uv_list, color="red", name="box"):
    """
    image: H x W x 3 (RGB) numpy array
    corners_uv_list: iterable of (8,2) arrays, each [u, v] for the 8 box corners
                     in the **same index order** as your 3D function:
                         0-3: bottom rectangle (0-1-2-3)
                         4-7: top rectangle (4-5-6-7)
    """

    # Edges as in your 3D version
    edges = [
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ]

    # Ensure iterable of arrays
    corners_uv_list = [np.asarray(c) for c in corners_uv_list]

    for corners_uv in corners_uv_list:
        # corners_uv: (8,2) -> [u,v]
        xs, ys = [], []
        for s, e in edges:
            xs += [corners_uv[s, 0], corners_uv[e, 0], None]
            ys += [corners_uv[s, 1], corners_uv[e, 1], None]

        trace = go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=color, width=2),
        )
        fig.add_trace(trace)

    return fig