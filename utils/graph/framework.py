from __future__ import annotations

import dash
import dash_cytoscape as cyto
import dash_html_components as html

cyto.load_extra_layouts()
app = dash.Dash("Concept Visualization")

def display_graph(elements: list):
    app.layout = html.Div([
        cyto.Cytoscape(
            id='concept-visualization',
            style={'width': '100vw', 'height': '100vh'},
            layout={
                'name': 'concentric',
                'animate': True,
            },
            elements=elements
        )
    ])
    app.run_server(debug=True)