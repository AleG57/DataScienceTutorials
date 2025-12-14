# Import des librairies pour le parcours de graphes
# Parcours de graphes
from pyvis.network import Network
import networkx as nx
import plotly.graph_objects as go

def graph():
    # Init
    G = nx.Graph() # Création du graphe
    
    with open("src/Nodes.txt","r") as file: # Ajout des noeuds
        Nodes = [line.strip() for line in file.readlines()]
    
    with open("src/Edges.txt", "r") as file: # Ajout des arrêtes
        Edges = [
            tuple(part.strip().strip('"') for part in line.strip().split(",")) for line in file.readlines() # FORMAT "[parent]", "[child]"
        ]
    
    with open("src/Description.txt") as file:
        descr = [
            tuple(part.strip().strip('"') for part in line.strip().split(",")) for line in file.readlines() # FORMAT "[node]", "[description]"
        ]
    
    for node in Nodes:
        if node == "Modèle":
            G.add_node(node, title="Bonjour 👋")  # Tooltip spécial pour "Modèle"
        else:
            G.add_node(node)  # Les autres sans tooltip
    
    
    hover_texts = [f"{label} : {desc[-1]}" for label, desc in zip(Nodes, descr)]
    
    
    # Graphe
    G.add_nodes_from(Nodes)
    G.add_edges_from(Edges)
    
    # Style
    # Positionnement des noeuds (layout)
    pos = nx.spring_layout(
                            G, 
                            seed=42,
                            k=0.25
                          )
    
    # Création des arêtes
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Création des noeuds
    node_x, node_y, text, size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)
        size.append(G.degree[node] * 6 + 10)
    
    # Style du graphe
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        text=text,
        textposition='top center',
        hoverinfo='text',
        hovertext=hover_texts,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[G.degree[n] for n in G.nodes()],
            size=size,
            colorbar=dict(
                thickness=15,
                title='Degré du noeud',
                xanchor='left'
            ),
            line_width=2
        )
    )
    
    # Création du layout interactif
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Carte modèles statistiques, de ML et DL',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Visualisation interactive avec Plotly + NetworkX",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )
    
    # Sauvegarde ou affichage
    map = fig.write_html("Models.html", auto_open=True)

if __name__ == "__main__":
    graph()