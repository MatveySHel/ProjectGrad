import numpy as np
import random
import matplotlib.pyplot as plot



def differentiable_function(x):
    function = (x[0])**3
    return function
#np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2) + \
           #np.cos(x[1]) * np.exp((1 - np.sin(x[0])) ** 2) + (x[0] - x[1]) ** 2



def gradient(x, dx):
    grad = []
    for i in range(len(x)):
        a = x.copy()
        a[i]=a[i]+dx
        grad.append((differentiable_function(a)-differentiable_function(x))/dx)
    return grad



def gradient_search(x0, dx, center, radius, extr_type, d=0.1, epsilon = 10**(-4)):
  if extr_type == 'min':
    k=-1
  else:
    k=1
  x0 = np.array(x0)
  n=0
  while sum(np.abs(np.array(gradient(x=list(x0), dx=dx))))>=epsilon:
    if sum((x0 + np.array(gradient(x=list(x0), dx=dx))*k*d - np.array(center))**2)<=radius**2:
        x0 = x0 + np.array(gradient(x=list(x0), dx=dx))*k*d
        if d>0.01:
            d = d*0.95
        n=n+1
        if n == 10000:
            break
    else:
        x0=[]
        break
  return list(x0)



def silvestr_criterion(x0, dx=0.0001, epsilon = 0.001):
  A = derivative_2(x0, dx)
  if len(A)==1:
    if abs(A[0][0])<epsilon:
      return "Not an extremum"
    elif A[0][0]>0:
      return "min"
    else:
      return "min"
  elif A[0][0]>0 and abs(A[0][0])>epsilon:
    for i in range(2,len(A)+1):
      minor_det = np.linalg.det(np.array(A)[:i,:i])
      if minor_det<0 or abs(minor_det)<epsilon:
        return "Not an extremum"
        break
    return "min"
  elif A[0][0]<0 and abs(A[0][0])>epsilon:
    for i in range(2,len(A)+1):
      minor_det =  np.linalg.det(np.array(A)[:i,:i])
      if i == 2:
        if minor_det<0 or abs(minor_det)<epsilon:
          return "Not an extremum"
          break
        else:
          prev_minor_det = minor_det
      elif prev_minor_det*minor_det>0 or abs(minor_det)<epsilon:
        return "Not an extremum"
        break
      else:
        prev_minor_det = minor_det
    return "max"
  else:
    return "Not an extremum"




def get_grid(grid_step):
    samples = np.arange(-radius, radius, grid_step)
    Ox, Oy = np.meshgrid(samples, samples)
    return Ox, Oy, differentiable_function([Ox,Oy])



def draw_chart(points, grid):
    grid_x, grid_y, grid_z = grid
    plot.rcParams.update({
        'figure.figsize': (4, 4),
        'figure.dpi': 200,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4
    })
    ax = plot.figure().add_subplot(111, projection='3d')
    for point in points:
        ax.scatter(point[0], point[1], point[-1], color='red')
    ax.plot_surface(grid_x, grid_y, grid_z, rstride=5, cstride=5, alpha=0.7)
    plot.show()



def derivative_2(x, dx):
    gesse = []
    for i in range(len(x)):
        a = x.copy()
        a[i] = a[i] + dx
        gesse.append(list((np.array(gradient(a, dx)) - np.array(gradient(x, dx))) / dx))
    return gesse



if __name__ == '__main__':


    random.seed(1113)
    radius = 8
    center = [0]
    epsilon = 0.000001
    dx = 0.0001
    d = 1
    start_points_qnt = 20
    type='min'
    try:
        differentiable_function(center)
        start_points = []
        n = 0
        while len(start_points) != start_points_qnt:
            x0 = [random.uniform(center[i] - radius, center[i] + radius) for i in range(len(center))]
            if sum((np.array(x0) - np.array(center)) ** 2) <= radius ** 2:
                start_points.append(x0)

        extra_points = []
        extra0_points = []
        for j in range(len(start_points)):
            result = gradient_search(start_points[j], dx=dx, center=center, radius=radius, extr_type=type, d=0.1, epsilon=10 ** (-4))
            if len(result)!=0:
                z=0
                if len(extra0_points)>0:
                    for u in range(len(result)):
                        if abs(result[u]-extra0_points[-1][u])<=10**(-4):
                            z+=1
                    if z!=len(result):
                        extra0_points.append(list(result))
                else:

                       extra0_points.append(list(result))

        for t in range(len(extra0_points)):
            if silvestr_criterion(extra0_points[t], dx=0.0001, epsilon = 0.1)!="Not an extremum":
                extra_points.append(extra0_points[t])
        for t in range(len(extra_points)):
            extra_points[t].append(differentiable_function(extra_points[t]))
        print(extra_points)

        if len(center)==2:
            draw_chart(extra_points, get_grid(0.05))

        if len(center)==1:
            plot.plot([i for i in list(np.arange(-10-radius, radius+10,0.01))], [differentiable_function([j]) for j in list(np.arange(-10-radius, radius+10,0.01))])
            for s in range(len(extra_points)):
                plot.scatter(extra_points[s][0], extra_points[s][1], c='red')
            plot.show()

    except:
        print('center coordinate is not exist')




#functions for tests:

def gradient_test1(x, dx):
    grad = []
    for i in range(len(x)):
        a = x.copy()
        a[i]=a[i]+dx
        grad.append(round((((a[0]**2+3*a[0]+5)-(x[0]**2+3*x[0]+5))/dx),4))
    return grad


def gradient_test2(x, dx):
    grad = []
    for i in range(len(x)):
        a = x.copy()
        a[i]=a[i]+dx
        grad.append(round((((a[0]**2+a[1]**2+5)-(x[0]**2+x[1]**2+5))/dx),4))
    return grad



def differentiable_function_test(x):
    function = x[0]**2+x[1]**2
    return function


def gradient_search_test(x0, dx, center, radius, extr_type, d=0.1, epsilon = 10**(-4)):
  if extr_type == 'min':
    k=-1
  else:
    k=1
  x0 = np.array(x0)
  n=0
  while sum(np.abs(np.array(gradient_test1(x=list(x0), dx=dx))))>=epsilon:
    if sum((x0 + np.array(gradient_test1(x=list(x0), dx=dx))*k*d - np.array(center))**2)<=radius**2:
        x0 = x0 + np.array(gradient_test1(x=list(x0), dx=dx))*k*d
        if d>0.01:
            d = d*0.95
        n=n+1
        if n == 10000:
            break
    else:
        x0=[]
        break
  return round(list(x0)[0],3)



def gradient_test3(x, dx):
    grad = []
    for i in range(len(x)):
        a = x.copy()
        a[i]=a[i]+dx
        grad.append(round(((a[0])**2+(a[1])**2+a[2]**2)-((x[0])**2+(x[1])**2+x[2]**2)/dx,3))
    return grad


def derivative_2_test(x, dx):
    gesse = []
    for i in range(len(x)):
        a = x.copy()
        a[i] = a[i] + dx
        gesse.append(list((np.array(gradient_test3(a, dx)) - np.array(gradient_test3(x, dx))) / dx))
    for i in range(len(gesse)):
        for j in range(len(gesse[i])):
            gesse[i][j]=round(gesse[i][j],3)
    return gesse