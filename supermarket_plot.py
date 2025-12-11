import json 
import numpy as np
import matplotlib.pyplot as plt


# <========= Theoretical Part Of CDF =========>

def theoretical_cdf(lambd,d,max_t=20):
    xs = np.arange(0, max_t + 1)
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
        # theoretical
        xs_t, ys_t = theoretical_cdf(lambd, d)
        plt.plot(xs_t, ys_t, '--', label=f"Theory d={d}")

        # experimental
        if d in experimental_files:
            xs_e, ys_e = load_experimental(experimental_files[d])
            plt.plot(xs_e, ys_e, marker='o', label=f"Sim d={d}")

    plt.xlabel("Queue length x")
    plt.ylabel("P(queue length ≥ x)")
    plt.title(f"Supermarket Model (λ={lambd})")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.show()


# <========= Example Usage =========>

if __name__ == "__main__":
    lambd = 0.8
    d_values = [1, 2, 5]

    experimental_files = {
        1: "d1.json",
        2: "d2.json",
        5: "d5.json",
    }

    plot_combined(d_values, lambd, experimental_files) 
         