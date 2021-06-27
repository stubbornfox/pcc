from __future__ import annotations

import dash
import dash_cytoscape as cyto
import dash_html_components as html

cyto.load_extra_layouts()
app = dash.Dash("Concept Visualization")

def display_graph(elements: list, root_id: str):
    app.layout = html.Div([
        cyto.Cytoscape(
            id='concept-visualization',
            style={'width': '100vw', 'height': '100vh'},
            layout={
                'name': 'breadthfirst',
                'animate': True,
                # Here you can put the node that should be the focus of the
                # graph. It will be placed at the top, with all it's concepts
                # being placed directly below it.
                'roots': f'[id="{root_id}"]',
            },
            elements=elements
        )
    ])
    app.run_server(debug=False)

def display_decision_tree(elements: list, root_id: str):
    app.layout = html.Div([
        cyto.Cytoscape(
            id='decision-tree-visualization',
            style={'width': '100vw', 'height': '100vh'},
            layout={
                'name': 'cose',
                'animate': True,
                'roots': f'[id="{root_id}"]',

            },
            elements=elements,
            stylesheet=[
                {
                    'selector': 'edge',
                    'style': {
                        'label': 'data(weight)'
                    }
                },
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(weight)'
                    }
                },
            ]
        )
    ])
    app.run_server(debug=False)