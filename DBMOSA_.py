import os
import copy
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


os.chdir("~/PB2")


def y_1(x):
    """ MOP objective function 1.
    Args: 
        x (float): Decision variable value.
    Returns:
        x**2 (float): Objective function value.
    """
    return x**2


def y_2(x):
    """ MOP objective function 2.
    Args: 
        x (float): Decision variable value.
    Returns:
        (x-2)**2 (float): Objective function value.
    """
    return (x-2)**2


def x_prime(x):
    """ Generate neighbouring xprime from x.
    Args: 
        x (float): Input x.
    Returns:
        xprime (float): Neighbour of x.
    """
    r1 = np.random.uniform(0, 1)
    xprime = x + (0.5 - r1) * min([3, 10**5 - x, x + 10**5])
    
    return xprime


def reheated_geo(T):
    """ Geometric reheating schedule (b=1.2).
    Args:
        T (float): Current temperature.
    Returns:
        T (float): Increased (reheated) temperature.
    """
    T = float(T*1.2)
    
    return T


def cooled_geo(T):
    """ Geometric cooling schedule (a=0.9).
    Args:
        T (float): Current temperature.
    Returns:
        T (float): Decreased (cooled) temperature.
    """
    T = float(T*0.9)
    
    return T


def reheated_lin(T):
    """ Linear reheating schedule (b=0.1).
    Args:
        T (float): Current temperature.
    Returns:
        T (float): Increased (reheated) temperature.
    """
    T = float(T + 0.1)
    
    return T


def cooled_lin(T):
    """ Geometric cooling schedule (a=0.08).
    Args:
        T (float): Current temperature.
    Returns:
        T (float): Decreased (cooled) temperature.
    """
    T = float(T + 0.08)
    
    return T


def dominance_num(test_point, archive):
    """ Gives number of solutions in an archive (A) dominated by a test point
    (x or xprime for DBMOSA).
    Args:
        test_point (tuple): Point to check.
        archive (list): A to search through for points dominated.
    Returns:
        dominance_sum (int): Number of solutions in A dominated by test_point.
    """
    y1_Tp, y2_Tp = test_point[0], test_point[1]
    dominance_list = []
    for i in archive:
        y1_Ap, y2_Ap = i[0], i[1]
        # case 1
        if (y1_Ap <= y1_Tp) and (y2_Ap < y2_Tp):
            dominance_list.append(1)
        # case 2
        elif (y1_Ap < y1_Tp) and (y2_Ap <= y2_Tp):
            dominance_list.append(1)
        # case 3
        elif (y1_Ap < y1_Tp) and (y2_Ap < y2_Tp):
            dominance_list.append(1)
        else:
            dominance_list.append(0)
            
    dominance_sum = sum(dominance_list)
        
    return dominance_sum


def remove_dominated_solns(new_point, archive, decision_vars):  
    """ Gives new archive (A) of which solutions dominated by new_point have 
    been removed.
    Args:
        new_point (tuple): Point recently added, used to compare to other A 
        points.
        archive (list): A to search through for points dominated.
        decision_vars (list): Coresponding decision vars for archive pairs. 
        Included as argument to ensure removal of dominated solutions from 
        both the archive list and decision variable list.
    Returns:
        new_A (list): A w/ test_point and w/o solutions that 
        test_point dominates.
        decision_vars (list): Decision variables w/ test_point decision 
        variable and and w/o solutions that test_point decision variable 
        dominates.
    """ 
    y1_Tp, y2_Tp = new_point[0], new_point[1]
    dominance_list = []
    # signs flipped compared to dominance_num() (elimnate dominated points)
    for i in archive: 
        y1_Ap, y2_Ap = i[0], i[1]
        # case 1
        if (y1_Ap >= y1_Tp) and (y2_Ap > y2_Tp):
            dominance_list.append(1)
        # case 2
        elif (y1_Ap > y1_Tp) and (y2_Ap >= y2_Tp):
            dominance_list.append(1)
        # case 3
        elif (y1_Ap > y1_Tp) and (y2_Ap > y2_Tp):
            dominance_list.append(1)
        else:
            dominance_list.append(0)
    
    # get list indices having 1 as value            
    dominated_soln_inds = [i for i, x in enumerate(dominance_list) if x==1]
    # new A excludes dominated indices
    new_A = [i for j, i in enumerate(archive) if j not in dominated_soln_inds]
    # do the same for decision variable list
    new_decision_vars = [i for j, i in enumerate(decision_vars) if j not in \
                         dominated_soln_inds]
        
    return (new_A, new_decision_vars)


def archive_metrics(x_fn_tup, xprime_fn_tup, A):
    """ Provides cardinality metrics for current archive required by DBMOSA 
    algorithm.
    Args: 
        x_fn_tup (tuple): Function values y1 and y2 based on x.
        xprime_fn_tup (tuple): Function values y1 and y2 based on xprime.
        A (list): Current archive.
    Returns:
        (A_tilde_card, A_tilde_x_card, A_tilde_xprime_card) (tuple))
        A_tilde_card (int): Cardinality of A tilde, where A tilde is the 
        archive containing existing solutions, a current solution (x) and a 
        neighbouring solution (xprime) of x. i.e The number solutions in 
        A_tilde.       
        A_tilde_x_card (int): Cardinality of A tilde x, i.e number solutions 
        in A tilde dominated by x.
        A_tilde_xprime_card (int): Cardinality of A tilde xprime, i.e number 
        solutions in A tilde dominated by xprime.
    """
    A_tilde = copy.deepcopy(A) # A_tilde is A but with neighbouring soln incl.
    A_tilde.append(xprime_fn_tup)
    
    # # in A_tilde
    A_tilde_card = len(set(A_tilde))
    # # in A dominated by x
    A_tilde_x_card = dominance_num(x_fn_tup, A_tilde) 
    # # in A dominated by xprime
    A_tilde_xprime_card = dominance_num(xprime_fn_tup, A_tilde) 
    
    return (A_tilde_card, A_tilde_x_card, A_tilde_xprime_card)


def save_df(archive, decision_vars, prefix):
    """ Returns and saves timestamped df as .csv in 'cwd/dfs' for inspection.
    Args: 
        archive (list): A to search through for points dominated.
        decision_vars (list): Coresponding decision vars for archive pairs. 
        prefix (str): Name to give file as prefix, useful to identify runs 
        from different approaches.
    Returns:
        record_df (pandas DataFrame): DataFrame containing DBMOSA results.
    """
    y1s = [p[0] for p in archive]
    y2s = [p[1] for p in archive]
    record_df = pd.DataFrame.from_dict({
                "Decision variables": decision_vars, 
                "Archive - y1": y1s,
                "Archive - y2": y2s})
    now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
    record_df.to_csv("dfs/"+prefix+"_{time}.csv".format(time=now))
    print(record_df)
    
    return record_df


def create_plots(archive, decision_vars, record_df):
    """ Renders plots for DBMOSA results visualisation.
    Args: 
        archive (list): A to search through for points dominated.
        decision_vars (list): Coresponding decision vars for archive pairs.
        record_df (pandas DataFrame): DataFrame containing DBMOSA results.
    """   
    # Plot 3 (2D) - 'Approximate Pareto front in objective space'
    plt.figure(3)
    xs = [*range(0, 5)] # decision var range (adapt) ('zooms in')
    y1s = [y_1(x) for x in xs]
    y2s = [y_2(x) for x in xs]
    plt.plot(y1s, y2s, color="blue", label="(y1, y2)")
    y1s_p_approx = [p[0] for p in archive]
    y2s_p_approx = [p[1] for p in archive]
    plt.scatter(y1s_p_approx, y2s_p_approx, color="red", 
                label="Approximated Pareto front")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Approximated Pareto front (objective space)")
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot 4 (2D) - 'True Pareto front in objective space'
    plt.figure(4)
    xs = list(np.round(np.arange(0, 4, .1), 1)) # decision var range (adapt) 
    #                                             ('zooms in')
    y1s = [y_1(x) for x in xs]
    y2s = [y_2(x) for x in xs]
    plt.plot(y1s, y2s, color="blue", label="(y1, y2)")
    # now also plot true pareto front, as can be seen, y1 and y2 BOTH decrease
    # until +- 4.2 then increase and hence prior to this is the true pareto 
    # front
    inc_ind = -1 # find increasing point in y2s list
    for i, e in enumerate(y2s):
        if y2s[i+1] > e: 
            inc_ind = i
            break
        
    y1s_p_true = y1s[:inc_ind+1]
    y2s_p_true = y2s[:inc_ind+1]
    plt.scatter(y1s_p_true, y2s_p_true, color="red", 
                label="True Pareto front")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("True Pareto front (objective space)")
    plt.legend(loc="lower right")
    plt.show()

    # Plot 1 (3D) - opens in browser - Approx Pareto in decision space
    fig1 = px.scatter_3d(record_df, x="Archive - y1", 
                        y="Archive - y2", 
                        z="Decision variables", 
                        color="Decision variables",
                        color_continuous_scale="viridis",
                        opacity=0.9,
                        labels={"Archive - y1": "y1", 
                                "Archive - y2": "y2", 
                                "Decision variables": "x"},
                        title="Approximated Pareto optimal set in decision "+\
                            "space with y1 and y2 values")
    fig1.show()
    
    # Plot 2 (3D) - opens in browser - True Pareto in decision space
    xs_p_true = xs[:inc_ind+1]
    true_pareto_df = pd.DataFrame.from_dict({"y1": y1s_p_true,
                                             "y2": y2s_p_true,
                                             "x": xs_p_true})
    fig2 = px.scatter_3d(true_pareto_df, 
                        x="y1", 
                        y="y2", 
                        z="x", 
                        color="x",
                        color_continuous_scale="viridis",
                        opacity=0.9,
                        title="Pareto optimal set in decision space with y1"+\
                            " and y2 values")
    fig2.show()
    
    print("Plots rendered [using IPython/QT and browser (3D plots)].")
    

def DBMOSA(xinit, imax, cmax, dmax, T, reheated, cooled):
    """
    Args:
        xinit (float): Starting x value (to give initial solution).
        imax (int): Max epochs.
        cmax (int): Max acceptances.
        dmax (int): Max rejections.
        T (int): Starting temperature.
        reheated (function): Function for reheating schedule.
        cooled (function): Function for cooling schedule.       
    Returns:
        (A, decision_vars) (tuple)
        A (list): Final archive of (y1, y2) pairs.
        decision_vars (list): Final decision variables for each pair in 
        archive.        
    """
    x = xinit # STEP 0
    x_fn_tup = (y_1(x), y_2(x))
    A = [x_fn_tup]
    i = 0
    c = 0
    d = 0
    t = 0
    go_to_step2 = False # used to bypass/access break/continue statements
    decision_vars = []
    decision_vars.append(x)
    while True:
        if i == imax: # STEP 1
            
            return (A, decision_vars)
        
        while True:
            if d == dmax: # STEP 2
                T = reheated(T)
                i += 1
                c = 0
                d = 0
                break # break, then to continue1 (back to step 1)
    
            else:
                while True:
                    if c == cmax:  # STEP 3
                        T = cooled(T)
                        i += 1
                        c = 0
                        d = 0
                        break # brk, next brk then continue1 (back to step 1)
                        
                    else:
                        xprime = x_prime(x) # STEP 4
                        xprime_fn_tup = (y_1(xprime), y_2(xprime))                    
                        A_tilde_card, \
                        A_tilde_x_card, \
                        A_tilde_xprime_card = archive_metrics(x_fn_tup, 
                                                       xprime_fn_tup, 
                                                       A)
                        delta_E = (A_tilde_xprime_card \
                                   - A_tilde_x_card)/A_tilde_card
                        
                        rand_unif = np.random.uniform(0, 1)
                        prob_xprime = min([1, math.exp(-((delta_E)/T))])
                        
                        if rand_unif > prob_xprime: # STEP 5
                            t += 1
                            d += 1
                            go_to_step2 = True
                            break # break then continue (back to step 2)                          
                            
                        else:
                            x_fn_tup = xprime_fn_tup # STEP 6
                            c += 1 # STEP 7
                            
                            if A_tilde_x_card == 0: # STEP 8
                                A.append(x_fn_tup) # A=AU{x} (add to arch)
                                decision_vars.append(xprime) # capture dvars
                                # remove items in A dominated by x
                                # and remove cores. dvars
                                A, decision_vars = remove_dominated_solns(
                                    x_fn_tup, A, decision_vars
                                    )
                            t += 1 # STEP 9
                            continue # (back to step 3)
                            
                if go_to_step2: # True if need to return to step 2
                    go_to_step2 = False # reset variable
                    continue # go to step 2
            
                break
            
        continue # continue1 (back to step 1) 
               

if __name__ == "__main__":
# =============================================================================
#       Approach 1 - base approach:
#     * Starting temperature = 1.4 (accept all)
#     * imax (epoch length): 10 (static)
#     * Cooling/reheating schedule: a=0.9 (geometric) / b=1.2 (geometric)
#     * cmax (max acceptances): 3
#     * dmax (max rejections): 5
#     * Search termination criterion: Reach imax (max epochs)
# =============================================================================
    # archive, decision_vars = DBMOSA(xinit=0, imax=10, cmax=3, dmax=5, T=1.4, 
    #                                 reheated=reheated_geo, cooled=cooled_geo)
    # print(archive, "\n", decision_vars)
    # record_df = save_df(archive, decision_vars, prefix="APP1")
    # create_plots(archive, decision_vars, record_df)
    
# =============================================================================
#       Approach 2 - double epochs:
#     * Starting temperature = 1.4 (accept all)
#     * imax (epoch length): 20 (static)
#     * Cooling/reheating schedule: a=0.9 (geometric) / b=1.2 (geometric)
#     * cmax (max acceptances): 3
#     * dmax (max rejections): 5
#     * Search termination criterion: Reach imax (max epochs)
# =============================================================================
    # archive, decision_vars = DBMOSA(xinit=0, imax=20, cmax=3, dmax=5, T=1.4, 
    #                                 reheated=reheated_geo, cooled=cooled_geo)
    # print(archive, "\n", decision_vars)
    # record_df = save_df(archive, decision_vars, prefix="APP2")
    # create_plots(archive, decision_vars, record_df)

# =============================================================================
#       Approach 3 - different cooling/reheating schedules:
#     * Starting temperature = 1.4 (accept all)
#     * imax (epoch length): 20 (static)
#     * Cooling/reheating schedule: a=0.08 (linear) / b=0.1 (linear)
#     * cmax (max acceptances): 3
#     * dmax (max rejections): 5
#     * Search termination criterion: Reach imax (max epochs)
# =============================================================================
    archive, decision_vars = DBMOSA(xinit=0, imax=20, cmax=3, dmax=5, T=1.4, 
                                    reheated=reheated_lin, cooled=cooled_lin)
    print(archive, "\n", decision_vars)
    record_df = save_df(archive, decision_vars, prefix="APP3")
    create_plots(archive, decision_vars, record_df)
    
    # # dfs for inspection
    # app1 = pd.read_csv("dfs/APP1_03-09-2021-003953.csv")
    # app2 = pd.read_csv("dfs/APP2_03-09-2021-004322.csv")
    # app3 = pd.read_csv("dfs/APP3_03-09-2021-004937.csv")
    