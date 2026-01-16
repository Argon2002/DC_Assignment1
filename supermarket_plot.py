import json 
import numpy as np
import matplotlib.pyplot as plt


# <========= Theoretical CDF =========>

def theoretical_cdf(lambd,d,max_x):
    xs = np.arange(0, max_x + 1)
    if(d == 1):
        ys = lambd ** xs
    else:
        ys = lambd ** ((d**xs -1) / (d - 1))
        
    return (xs,ys)    


# <========= Loading experimental CDFs from json files  =========>
         
def load_experimental(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    
    xs = sorted(int(k) for k in data.keys())
    ys = [data[str(x)] for x in xs]
    
    return (xs,ys)             
         
         
 
# <========= Plotting both theoretical and experimental datas  =========>

def plot_combined(d_values, lambd, experimental_files):
    plt.figure(figsize=(7,5))

    for d in d_values:
        max_x = 20 

        # experimental
        if d in experimental_files:
            xs_e, ys_e = load_experimental(experimental_files[d])
            if xs_e:
                max_x = max(xs_e)
            plt.plot(xs_e, ys_e, marker='o', label=f"Sim d={d}")
            
            
        # theoretical
        xs_t, ys_t = theoretical_cdf(lambd, d, max_x=max_x)
        plt.plot(xs_t, ys_t, '--', label=f"Theory d={d}")    
    
    plt.xlabel("Queue length")
    plt.ylabel("Fraction of queues with at least that size")
    plt.title(f"Supermarket Model (λ={lambd})")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.xlim(0,14)
    plt.show()


