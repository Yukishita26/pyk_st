import numpy as np
import matplotlib.pyplot as plt

step = lambda x:1*(x>=0)
def gen_rect_fn(a=0, b=1):
    return lambda x:1*(a<=x)*(x<=b)
f = lambda x: 3*x*(x<=1/2) + (-3*x+3)*(x>=1/2)
def gousei(*fns):
    if len(fns)==0:return lambda x:x
    else:return lambda x:fns[0](gousei(*fns[1:])(x))
def plot_show(fn, a=0, b=1, n=100):
    lin = np.linspace(a, b, n)
    plt.plot(lin, fn(lin))
    plt.show()

def nth_comp_fn(fn,n):return gousei(*([fn]*n))

if __name__=="__main__":
    plot_show(f)
    plot_show(gousei(step,nth_comp_fn(f,5)), a=-0.1, b=1.1, n=1000)
