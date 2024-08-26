import numpy  as np
import plotly.io as pio
import plotly.graph_objects as go
import numpy  as np
import random

from numpy.core.defchararray import isnumeric


def read_tsp_file(filename: str):
    coordinates = np.array([])
    with open(filename, 'r') as file:
        count = 0
        while True:
            line = file.readline()
            count += 1
            if 'EOF' in line:
                break
            parts = line.split(' ')
            if len(parts) == 3 and isnumeric(parts[0]):
                index_str, x_str, y_str = line.split(" ")
                x = float(x_str)
                y = float(y_str)
                if len(coordinates) == 0:
                    coordinates = np.array([[x, y]])
                else:
                    coordinates = np.append(coordinates, [[x, y]], axis=0)
    return coordinates

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Initial Seed
def seed_function(distance_matrix):
    seed     = [[], float('inf')]
    sequence = random.sample(list(range(1, distance_matrix.shape[0]+1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0]  = sequence
    seed[1]  = distance_calc(distance_matrix, seed)
    return seed

# Function: Build Coordinates
def build_coordinates(distance_matrix):  
    a           = distance_matrix[0,:].reshape(distance_matrix.shape[0], 1)
    b           = distance_matrix[:,0].reshape(1, distance_matrix.shape[0])
    m           = (1/2)*(a**2 + b**2 - distance_matrix**2)
    w, u        = np.linalg.eig(np.matmul(m.T, m))
    s           = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    return coordinates

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Build Coordinates
def build_coordinates(distance_matrix):
    a           = distance_matrix[0,:].reshape(distance_matrix.shape[0], 1)
    b           = distance_matrix[:,0].reshape(1, distance_matrix.shape[0])
    m           = (1/2)*(a**2 + b**2 - distance_matrix**2)
    w, u        = np.linalg.eig(np.matmul(m.T, m))
    s           = (np.diag(np.sort(w)[::-1]))**(1/2)
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    return coordinates

############################################################################

# Function: Solution Plot
def plot_tour(coordinates, city_tour = [], view = 'browser', size = 10):
    if (coordinates.shape[0] == coordinates.shape[1]):
      coordinates = build_coordinates(coordinates)
    if (view == 'browser' ):
        pio.renderers.default = 'browser'
    if (len(city_tour) > 0):
        xy = np.zeros((len(city_tour), 2))
        for i in range(0, len(city_tour)):
            if (i < len(city_tour)):
                xy[i, 0] = coordinates[city_tour[i]-1, 0]
                xy[i, 1] = coordinates[city_tour[i]-1, 1]
            else:
                xy[i, 0] = coordinates[city_tour[0]-1, 0]
                xy[i, 1] = coordinates[city_tour[0]-1, 1]
    else:
        xy = np.zeros((coordinates.shape[0], 2))
        for i in range(0, coordinates.shape[0]):
            xy[i, 0] = coordinates[i, 0]
            xy[i, 1] = coordinates[i, 1]
    data = []
    Xe   = []
    Ye   = []
    ids  = [ 'id: '+ str(i+1)+'<br>'+'x: '+str(round(coordinates[i,0], 2))+'<br>'+'y: '+str(round(coordinates[i,1], 2))  for i in range(0, coordinates.shape[0])]
    if (len(city_tour) > 0):
        id0  = 'id: '+str(city_tour[0])+'<br>'+'x: '+str(round(xy[0,0], 2)) +'<br>'+'y: '+str(round(xy[0,1], 2))
    else:
        id0 = 'id: '+str(1)+'<br>'+'x: '+str(round(xy[0,0], 2)) +'<br>'+'y: '+str(round(xy[0,1], 2))
    if (len(city_tour) > 0):
        for i in range(0, xy.shape[0]-1):
            Xe.append(xy[i,0])
            Xe.append(xy[i+1,0])
            Xe.append(None)
            Ye.append(xy[i,1])
            Ye.append(xy[i+1,1])
            Ye.append(None)
        e_trace = go.Scatter(x         = Xe[2:],
                             y         = Ye[2:],
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 1)', width = 0.50, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        data.append(e_trace)
    n_trace = go.Scatter(x         = coordinates[0:, -2],
                         y         = coordinates[0:, -1],
                         opacity   = 1,
                         mode      = 'markers+text',
                         marker    = dict(symbol = 'circle-dot', size = size, color = 'rgba(46, 138, 199, 1)'),
                         hoverinfo = 'text',
                         hovertext = ids[0:],
                         name      = ''
                         )
    data.append(n_trace)
    m_trace = go.Scatter(x         = xy[0:1, -2],
                         y         = xy[0:1, -1],
                         opacity   = 1,
                         mode      = 'markers+text',
                         marker    = dict(symbol = 'square-dot', size = size, color = 'rgba(247, 138, 54, 1)'),
                         hoverinfo = 'text',
                         hovertext = id0,
                         name      = ''
                         )
    data.append(m_trace)
    layout  = go.Layout(showlegend   = False,
                        hovermode    = 'closest',
                        margin       = dict(b = 10, l = 5, r = 5, t = 10),
                        plot_bgcolor = 'rgb(235, 235, 235)',
                        xaxis        = dict(  showgrid       = True,
                                              zeroline       = True,
                                              showticklabels = True,
                                              tickmode       = 'array',
                                           ),
                        yaxis        = dict(  showgrid       = True,
                                              zeroline       = True,
                                              showticklabels = True,
                                              tickmode       = 'array',
                                            )
                        )
    fig = go.Figure(data = data, layout = layout)
    if (len(city_tour) > 0):
        fig.add_annotation(
                           x          = Xe[1]*1.00,  # to x
                           y          = Ye[1]*1.00,  # to y
                           ax         = Xe[0]*1.00,  # from x
                           ay         = Ye[0]*1.00,  # from y
                           xref       = 'x',
                           yref       = 'y',
                           axref      = 'x',
                           ayref      = 'y',
                           text       = '',
                           showarrow  = True,
                           arrowhead  = 3,
                           arrowsize  = 1.5,
                           arrowwidth = 2,
                           arrowcolor = 'red',
                           opacity    = 1
                       )
    fig.update_traces(textfont_size = 10, textfont_color = 'rgb(255, 255, 255)')
    fig.write_image('result/result.jpg')
    fig.show()
    return

