# python3 -i baseline_plot.py --location './sigma' --dataset tinyimagenet

##################### THIS USES sigma = 1.2 not sigma = 1.0 because that calculation is missing for some reason

import numpy as np
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use(['seaborn-paper', './paper.mplstyle'])

import argparse

def save_figs(fn,types=('.pdf', '.png')):
    fig = plt.gcf()
    fig.tight_layout()
    for t in types:
        fig.savefig(fn+t, bbox_inches='tight')

plt.savefig = save_figs

#def at_radius(df, radius, col):
#    return (df[col] >= radius).mean()
    
def secondary_condition_radius(df, radius, col, key='cohen', median='no'):
    z = df[df[key] >= radius]
    #return (100*(z[col] - z['cohen'])/z['cohen']).mean() #(100*(df[col] - df[key]) / df[key]).mean()
    if median == 'yes':
        return (z[col] - z['cohen']).median()
    if median == 'max':
        return (z[col] - z['cohen']).max()        
    else:
       return (100*(z[col] - z['cohen'])/z['cohen']).mean() #(100*(df[col] - df[key]) / df[key]).mean()
       
def at_radius(df, radius, col, offset=False):
    if offset != False:
        length = df.shape[0] + df['rej'].iloc[-1]
        return ((df[col] >= radius).sum() ) / length
    return (df[col] >= radius).mean()
    
def at_radius_rolling(df, radius, col, key='E0', mean=True, percentage=False):
    #if offset != False:
    #    length = df.shape[0] + df['rej'].iloc[-1]
    #    return ((df[col] >= radius).sum() ) / length
    key = (df[key] >= np.max((0, radius - 0.075))) & (df[key] <= np.min((1.0, radius + 0.075)))
    if percentage:
        val = (100*(df[col][key] - df['cohen'][key]) / df['cohen'][key])
        if mean:
            val = val.replace(-np.inf, np.nan).dropna()
            #print(val, val.mean())
            #val = val.replace([np.inf, -np.inf], np.nan, inplace=True).dropna(how="all")
            return val.mean()
        else:
            return val.median()
    else:
        val = (df[col][key] - df['cohen'][key])
        if mean:
            return val.mean()
        else:
            return val.median()

    
    
def improvement_distribution(df, key, reference_val, reference_key='E0', operator='mean'):
    locs = (((df[reference_key] < (reference_val+0.05)) & (df[reference_key] > (reference_val-0.05)) & (df['cohen'] > 0)) == True)
    #print(reference_val, locs.sum(), (100*(df[key][locs] - df['cohen'][locs])/df['cohen'][locs]), df['cohen'][locs])
    if locs.sum() == 0:
        return 0
    if operator == 'mean':
        return (100*(df[key][locs] - df['cohen'][locs])/df['cohen'][locs]).mean()    
    else:
        return (100*(df[key][locs] - df['cohen'][locs])/df['cohen'][locs]).median()    
        

parser = argparse.ArgumentParser(description='Certifying examples as per Smoothing')
parser.add_argument('--location', type=str, default='')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--flag', type=str, default='medians')

args = parser.parse_args()

#os.chdir(args.location)
#cdir = os.getcwd()

#os.chdir('./' + args.dataset)

print('#################', os.getcwd())

# So I want to plot CR vs Technique 
df_5 = pd.read_csv('c10-0.5', delimiter='\t')
df_5.columns = df_5.columns.str.strip()
df_5['approx'] = df_5[['sim', 'd_sim', 'mod']].max(axis=1)
df_5['actual'] = df_5[['f_sim', 'f_d_sim', 'f_mod']].max(axis=1)

df_1 = pd.read_csv('c10-1.0', delimiter='\t')
df_1.columns = df_1.columns.str.strip()
df_1['approx'] = df_1[['sim', 'd_sim', 'mod']].max(axis=1)
df_1['actual'] = df_1[['f_sim', 'f_d_sim', 'f_mod']].max(axis=1)

#os.chdir('./new')

offset_flag = True

radii = np.linspace(0, 1.75, 100)


multi_class_5 = np.asarray([at_radius(df_5, rad, 'c_mu', offset=offset_flag) for rad in radii])
multi_class_1 = np.asarray([at_radius(df_1, rad, 'c_mu', offset=offset_flag) for rad in radii])


c_n_5 = np.asarray([at_radius(df_5, rad, 'cohen', offset=offset_flag) for rad in radii])
approx_n_5 = np.asarray([at_radius(df_5, rad, 'sim', offset=offset_flag) for rad in radii])
approx_d_5 = np.asarray([at_radius(df_5, rad, 'd_sim', offset=offset_flag) for rad in radii])
approx_m_5 = np.asarray([at_radius(df_5, rad, 'mod', offset=offset_flag) for rad in radii])
full_n_5 = np.asarray([at_radius(df_5, rad, 'f_sim', offset=offset_flag) for rad in radii])
full_d_5 = np.asarray([at_radius(df_5, rad, 'f_d_sim', offset=offset_flag) for rad in radii])
full_m_5 = np.asarray([at_radius(df_5, rad, 'f_mod', offset=offset_flag) for rad in radii])
approx_b_5 = np.asarray([at_radius(df_5, rad, 'approx', offset=offset_flag) for rad in radii])
full_b_5 = np.asarray([at_radius(df_5, rad, 'actual', offset=offset_flag) for rad in radii])

c_n_1 = np.asarray([at_radius(df_1, rad, 'cohen', offset=offset_flag) for rad in radii])
approx_n_1 = np.asarray([at_radius(df_1, rad, 'sim', offset=offset_flag) for rad in radii])
approx_d_1 = np.asarray([at_radius(df_1, rad, 'd_sim', offset=offset_flag) for rad in radii])
approx_m_1 = np.asarray([at_radius(df_1, rad, 'mod', offset=offset_flag) for rad in radii])
full_n_1 = np.asarray([at_radius(df_1, rad, 'f_sim', offset=offset_flag) for rad in radii])
full_d_1 = np.asarray([at_radius(df_1, rad, 'f_d_sim', offset=offset_flag) for rad in radii])
full_m_1 = np.asarray([at_radius(df_1, rad, 'f_mod', offset=offset_flag) for rad in radii])
approx_b_1 = np.asarray([at_radius(df_1, rad, 'approx', offset=offset_flag) for rad in radii])
full_b_1 = np.asarray([at_radius(df_1, rad, 'actual', offset=offset_flag) for rad in radii])

keys = ['b-', 'b--', 'b-.', 'b:', 'r-', 'r--', 'r-.', 'r:']
key = iter(keys)

plt.clf()
plt.plot(radii, c_n_5, next(key), label='Cohen $\sigma = 0.5$')
plt.plot(radii, approx_n_5, next(key), label='Single $\sigma = 0.5$')
plt.plot(radii, approx_d_5, next(key), label='Double $\sigma = 0.5$')
plt.plot(radii, approx_m_5, next(key), label='Boundary $\sigma = 0.5$')
plt.plot(radii, c_n_1, next(key), label='Cohen $\sigma = 1.0$')
plt.plot(radii, approx_n_1, next(key), label='Single $\sigma = 1.0$')
plt.plot(radii, approx_d_1, next(key), label='Double $\sigma = 1.0$')
plt.plot(radii, approx_m_1, next(key), label='Boundary $\sigma = 1.0$')
plt.xlabel('R')
plt.ylabel('Certified Proportion')
plt.legend(loc='upper right')
plt.savefig('approx_radii')

keys = ['b-', 'b--', 'b-.', 'b:', 'r-', 'r--', 'r-.', 'r:']
key = iter(keys)

plt.clf()
plt.plot(radii, multi_class_5, next(key), label='MACER $\sigma = 0.5$')
plt.plot(radii, approx_n_5, next(key), label='Single $\sigma = 0.5$')
plt.plot(radii, approx_d_5, next(key), label='Double $\sigma = 0.5$')
plt.plot(radii, approx_m_5, next(key), label='Boundary $\sigma = 0.5$')
plt.plot(radii, multi_class_1, next(key), label='MACER $\sigma = 1.0$')
plt.plot(radii, approx_n_1, next(key), label='Single $\sigma = 1.0$')
plt.plot(radii, approx_d_1, next(key), label='Double $\sigma = 1.0$')
plt.plot(radii, approx_m_1, next(key), label='Boundary $\sigma = 1.0$')
plt.xlabel('R')
plt.ylabel('Certified Proportion')
plt.legend(loc='upper right')
plt.savefig('macer_radii')


keys = ['b-.', 'g-.', 'r-.', 'b-', 'g-', 'r-']
key = iter(keys)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.set_aspect(2.5)
ax.plot(radii, multi_class_5, next(key), label='Cohen $\sigma = 0.5$')
ax.plot(radii, approx_b_5, next(key), label='Best Approx $\sigma = 0.5$')
ax.plot(radii, full_b_5, next(key), label='Best Full $\sigma = 0.5$')
ax.plot(radii, multi_class_1, next(key), label='Cohen $\sigma = 1.0$')
ax.plot(radii, approx_b_1, next(key), label='Best Approx $\sigma = 1.0$')
ax.plot(radii, full_b_1, next(key), label='Best Full $\sigma = 1.0$')

#aa = plt.figure().get_size_inches()
#print(aa[1], 0.75*aa[0], aa[0])
#plt.figure().set_figheight(aa[1])
#plt.figure().set_figwidth(aa[0]*0.75)
plt.xlabel('R')
plt.ylabel('Certified Proportion')

plt.legend(loc='upper right')

plt.savefig('best_macer_radii')





keys = ['b-', 'b--', 'b-.', 'b:', 'r-', 'r--', 'r-.', 'r:']
key = iter(keys)

plt.clf()
plt.plot(radii, c_n_5, next(key), label='Cohen $\sigma = 0.5$')
plt.plot(radii, full_n_5, next(key), label='Single $\sigma = 0.5$')
plt.plot(radii, full_d_5, next(key), label='Double $\sigma = 0.5$')
plt.plot(radii, full_m_5, next(key), label='Boundary $\sigma = 0.5$')
plt.plot(radii, c_n_1, next(key), label='Cohen $\sigma = 1.0$')
plt.plot(radii, full_n_1, next(key), label='Single $\sigma = 1.0$')
plt.plot(radii, full_d_1, next(key), label='Double $\sigma = 1.0$')
plt.plot(radii, full_m_1, next(key), label='Boundary $\sigma = 1.0$')
plt.xlabel('R')
plt.ylabel('Certified Proportion')
plt.legend(loc='upper right')
plt.savefig('full_radii')


keys = ['b-.', 'g-.', 'r-.', 'b-', 'g-', 'r-']
key = iter(keys)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect(2.5)
ax.plot(radii, c_n_5, next(key), label='Cohen $\sigma = 0.5$')
ax.plot(radii, approx_b_5, next(key), label='Best Approx $\sigma = 0.5$')
ax.plot(radii, full_b_5, next(key), label='Best Full $\sigma = 0.5$')
ax.plot(radii, c_n_1, next(key), label='Cohen $\sigma = 1.0$')
ax.plot(radii, approx_b_1, next(key), label='Best Approx $\sigma = 1.0$')
ax.plot(radii, full_b_1, next(key), label='Best Full $\sigma = 1.0$')

#aa = plt.figure().get_size_inches()
#print(aa[1], 0.75*aa[0], aa[0])
#plt.figure().set_figheight(aa[1])
#plt.figure().set_figwidth(aa[0]*0.75)
plt.xlabel('R')
if args.dataset == 'mnist':
    plt.ylabel('Certified Proportion')
if args.dataset == 'tinyimagenet':
    plt.legend(loc='upper right')

plt.savefig('best_radii')

with open('best_' + args.dataset + '.npy', 'wb') as f:
    np.save(f, radii)
    np.save(f, c_n_5)
    np.save(f, approx_b_5)
    np.save(f, full_b_5)
    np.save(f, c_n_1)
    np.save(f, approx_b_1)
    np.save(f, full_b_1)


######################################################################3

def process_for_key(df_5, df_1, radii, key='cohen', median=False, pm=False):
    if pm:
        if median == 'yes':
            median=True
        elif median == 'max':
            return
        else:
            median=False
        p_approx_n_5 = np.asarray([at_radius_rolling(df_5, rad, 'sim', key=key, percentage=True, mean=~median) for rad in radii]) 
        p_approx_d_5 = np.asarray([at_radius_rolling(df_5, rad, 'd_sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_approx_m_5 = np.asarray([at_radius_rolling(df_5, rad, 'mod', key=key, percentage=True, mean=~median) for rad in radii])

        p_full_n_5 = np.asarray([at_radius_rolling(df_5, rad, 'f_sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_full_d_5 = np.asarray([at_radius_rolling(df_5, rad, 'f_d_sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_full_m_5 = np.asarray([at_radius_rolling(df_5, rad, 'f_mod', key=key, percentage=True, mean=~median) for rad in radii])


        p_approx_n_1 = np.asarray([at_radius_rolling(df_1, rad, 'sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_approx_d_1 = np.asarray([at_radius_rolling(df_1, rad, 'd_sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_approx_m_1 = np.asarray([at_radius_rolling(df_1, rad, 'mod', key=key, percentage=True, mean=~median) for rad in radii])

        p_full_n_1 = np.asarray([at_radius_rolling(df_1, rad, 'f_sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_full_d_1 = np.asarray([at_radius_rolling(df_1, rad, 'f_d_sim', key=key, percentage=True, mean=~median) for rad in radii])
        p_full_m_1 = np.asarray([at_radius_rolling(df_1, rad, 'f_mod', key=key, percentage=True, mean=~median) for rad in radii])
       
    else:
        p_approx_n_5 = np.asarray([secondary_condition_radius(df_5, rad, 'sim', key=key, median=median) for rad in radii])
        p_approx_d_5 = np.asarray([secondary_condition_radius(df_5, rad, 'd_sim', key=key, median=median) for rad in radii])
        p_approx_m_5 = np.asarray([secondary_condition_radius(df_5, rad, 'mod', key=key, median=median) for rad in radii])

        p_full_n_5 = np.asarray([secondary_condition_radius(df_5, rad, 'f_sim', key=key, median=median) for rad in radii])
        p_full_d_5 = np.asarray([secondary_condition_radius(df_5, rad, 'f_d_sim', key=key, median=median) for rad in radii])
        p_full_m_5 = np.asarray([secondary_condition_radius(df_5, rad, 'f_mod', key=key, median=median) for rad in radii])


        p_approx_n_1 = np.asarray([secondary_condition_radius(df_1, rad, 'sim', key=key, median=median) for rad in radii])
        p_approx_d_1 = np.asarray([secondary_condition_radius(df_1, rad, 'd_sim', key=key, median=median) for rad in radii])
        p_approx_m_1 = np.asarray([secondary_condition_radius(df_1, rad, 'mod', key=key, median=median) for rad in radii])

        p_full_n_1 = np.asarray([secondary_condition_radius(df_1, rad, 'f_sim', key=key, median=median) for rad in radii])
        p_full_d_1 = np.asarray([secondary_condition_radius(df_1, rad, 'f_d_sim', key=key, median=median) for rad in radii])
        p_full_m_1 = np.asarray([secondary_condition_radius(df_1, rad, 'f_mod', key=key, median=median) for rad in radii])

    keys = ['g-', 'g-.', 'g:', 'r-', 'r-.', 'r:']
    key_i = iter(keys)

    plt.clf()
    plt.plot(radii, p_approx_n_5, next(key_i), label='Single (Appox)')
    plt.plot(radii, p_approx_d_5, next(key_i), label='Double (Appox)')
    plt.plot(radii, p_approx_m_5, next(key_i), label='Boundary (Appox)')
    plt.plot(radii, p_full_n_5, next(key_i), label='Single (Full)')
    plt.plot(radii, p_full_d_5, next(key_i), label='Double (Full)')
    plt.plot(radii, p_full_m_5, next(key_i), label='Boundary (Full)')   

    ylabel_key = 'Median Improvement'        
    if pm:
        if key=='cohen':
            plt.xlabel('$r_1$')# \pm 0.075$')
        elif key=='E0':
            plt.xlabel('$E_0$')# \pm 0.075$')             
        else:
            plt.xlabel('R')  
        suffix = '_pm'  
        if median:
            median = 'yes'
        else:
            median = 'no'
        ylabel_key = 'Median Percentage Improvement'                     
    else:
        if key=='cohen':
            plt.xlabel('$r_1 > x$')
        elif key=='E0':
            plt.xlabel('$E_0 > x$')             
        else:
            plt.xlabel('R')
        suffix = ''
    if median == 'yes':
        plt.ylabel(ylabel_key)
        #plt.legend(loc='upper right')
        plt.savefig('median_radii_5_' + key + suffix)    
    elif median == 'max':
        plt.ylabel('Median Improvement')
        #plt.legend(loc='upper right')
        plt.savefig('max_radii_5_' + key + suffix)            
    else:
        plt.ylabel('Average Percentage Improvement')
        #plt.legend(loc='upper right')
        plt.ylim(0, 30)
        plt.savefig('percent_radii_5_' + key + suffix)

    keys = ['g-', 'g-.', 'g:', 'r-', 'r-.', 'r:']
    key_i = iter(keys)

    plt.clf()
    plt.plot(radii, p_approx_n_1, next(key_i), label='Single (Approx)')
    plt.plot(radii, p_approx_d_1, next(key_i), label='Double (Approx)')
    plt.plot(radii, p_approx_m_1, next(key_i), label='Boundary (Approx)')
    plt.plot(radii, p_full_n_1, next(key_i), label='Single (Full)')
    plt.plot(radii, p_full_d_1, next(key_i), label='Double (Full)')
    plt.plot(radii, p_full_m_1, next(key_i), label='Boundary (Full)')
    ylabel_key = 'Median Improvement'    
    if pm:
        if key=='cohen':
            plt.xlabel('$r_1$')# \pm 0.075$')
        elif key=='E0':
            plt.xlabel('$E_0$')# \pm 0.075$')             
        else:
            plt.xlabel('R')  
        #suffix = '_pm'
        #if median:
        #    median = 'yes'
        #else:
        #    median = 'no' 
        ylabel_key = 'Median Percentage Improvement'         
    else:
        if key=='cohen':
            plt.xlabel('$r_1 > x$')
        elif key=='E0':
            plt.xlabel('$E_0 > x$')             
        else:
            plt.xlabel('R')
        suffix = ''
    if median == 'yes':
        plt.ylabel(ylabel_key)
        plt.legend(loc='upper right')
        #plt.axes().get_yaxis().set_ticks([])
        plt.savefig('median_radii_1_' + key + suffix)    
    elif median == 'max':
        plt.ylabel('Median Improvement')
        plt.legend(loc='upper right')
        #plt.axes().get_yaxis().set_ticks([])
        plt.savefig('max_radii_1_' + key + suffix)            
    else:
        plt.ylabel('Average Percentage Improvement')
        plt.legend(loc='upper right')
        plt.ylim(0, 30)
        #plt.axes().get_yaxis().set_ticks([])
        plt.savefig('percent_radii_1_' + key + suffix)
        
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    keys = ['g-', 'g-.', 'g:', 'r-', 'r-.', 'r:']
    key_i = iter(keys)
    ax1.plot(radii, p_approx_n_5, next(key_i), label='Single (Appox)')
    ax1.plot(radii, p_approx_d_5, next(key_i), label='Double (Appox)')
    ax1.plot(radii, p_approx_m_5, next(key_i), label='Boundary (Appox)')
    ax1.plot(radii, p_full_n_5, next(key_i), label='Single (Full)')
    ax1.plot(radii, p_full_d_5, next(key_i), label='Double (Full)')
    ax1.plot(radii, p_full_m_5, next(key_i), label='Boundary (Full)') 

    ax1.set_title('a) $\sigma = 0.5$')
    
    if pm:
        if key=='cohen':
            ax1.set_xlabel('$r_1$') #\pm 0.075$')
        elif key=='E0':
            ax1.set_xlabel('$E_0$') #\pm 0.075$')             
        else:
            ax1.set_xlabel('R')  
        #suffix = '_pm'
        #if median:
        #    median = 'yes'
        #else:
        #    median = 'no' 
        ylabel_key = 'Percentage Improvement'         
    else:
        if key=='cohen':
            ax1.set_xlabel('$r_1 > x$')
        elif key=='E0':
            ax1.set_xlabel('$E_0 > x$')             
        else:
            ax1.set_xlabel('R')
        suffix = '' 
        
    if median == 'yes':
        ax1.set_ylabel(ylabel_key)
        #plt.legend(loc='upper right')
        #plt.axes().get_yaxis().set_ticks([])
        ax1.set_ylim(0, 30)        
    elif median == 'max':
        ax1.set_ylabel('Median Improvement')
        #plt.legend(loc='upper right')
        #plt.axes().get_yaxis().set_ticks([])
    else:
        ax1.set_ylabel('Percentage Improvement')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 30)
        #plt.axes().get_yaxis().set_ticks([])
            
    keys = ['g-', 'g-.', 'g:', 'r-', 'r-.', 'r:']
    key_i = iter(keys)
    ax2.plot(radii, p_approx_n_1, next(key_i), label='Single (Approx)')
    ax2.plot(radii, p_approx_d_1, next(key_i), label='Double (Approx)')
    ax2.plot(radii, p_approx_m_1, next(key_i), label='Boundary (Approx)')
    ax2.plot(radii, p_full_n_1, next(key_i), label='Single (Full)')
    ax2.plot(radii, p_full_d_1, next(key_i), label='Double (Full)')
    ax2.plot(radii, p_full_m_1, next(key_i), label='Boundary (Full)')

    ax2.set_title('b) $\sigma = 1.0$')    
    
    if pm:
        if key=='cohen':
            ax2.set_xlabel('$r_1$') #0.075$')
        elif key=='E0':
            ax2.set_xlabel('$E_0$') #0.075$')             
        else:
            ax2.set_xlabel('R')  
        #suffix = '_pm'
        #if median:
        #    median = 'yes'
        #else:
        #    median = 'no' 
        ylabel_key = 'Median Percentage Improvement'         
    else:
        if key=='cohen':
            ax2.set_xlabel('$r_1 > x$')
        elif key=='E0':
            ax2.set_xlabel('$E_0 > x$')             
        else:
            ax2.set_xlabel('R')
        suffix = ''        
    if median == 'yes':
        #plt.ylabel(ylabel_key)

        ax1.legend(loc='upper right')
        ax2.get_yaxis().set_ticks([])
        ax2.set_ylim(0, 30)        
        plt.savefig('consolidated_median_radii_5_and_1_' + key + suffix)    
    elif median == 'max':
        #plt.ylabel('Median Improvement')
        ax2.legend(loc='upper right')
        ax2.get_yaxis().set_ticks([])
        plt.savefig('consolidated_max_radii_5_and_1_' + key + suffix)            
    else:
        #plt.ylabel('Average Percentage Improvement')
        #ax2.legend(loc='upper right')
        ax2.set_ylim(0, 30)
        ax2.get_yaxis().set_ticks([])
        plt.savefig('consolidated_percent_radii_5_and_1_' + key + suffix)        


process_for_key(df_5, df_1, np.linspace(0,2,200), key='cohen')
process_for_key(df_5, df_1, np.linspace(0,2,200), key='cohen', median='yes')
process_for_key(df_5, df_1, np.linspace(0,2,200), key='cohen', median='max')
process_for_key(df_5, df_1, np.linspace(0,1,200), key='E0')
process_for_key(df_5, df_1, np.linspace(0,1,200), key='E0', median='yes')
process_for_key(df_5, df_1, np.linspace(0,1,200), key='E0', median='max')

process_for_key(df_5, df_1, np.linspace(0,2,200), key='cohen', pm=True)
process_for_key(df_5, df_1, np.linspace(0,2,200), key='cohen', median='yes', pm=True)
process_for_key(df_5, df_1, np.linspace(0,1,200), key='E0', pm=True)
process_for_key(df_5, df_1, np.linspace(0,1,200), key='E0', median='yes', pm=True)

##########################################################################################
# Max Improvements
##########################################################################################

keys = ['sim', 'd_sim', 'mod', 'f_sim', 'f_d_sim', 'f_mod']
key_2 = ['0.5', '1.0']
for i, df in enumerate([df_5, df_1]):
    print('Sigma = ' + key_2[i])
    print('Max Cohen Val: ', df['cohen'].max())
    for key in keys:
        print('Max Improvement of : ' + str((df[key] - df['cohen']).max()) + ' for ' + key)


#z = df_1[df_1['E0'] >= 0.9]
#print((100*(z['mod'] - z['cohen'])/z['cohen']).mean(), secondary_condition_radius(df_1, 0.9, 'mod', key='E0'))


plt.clf()
fig, ax = plt.subplots()

a_heights, a_bins = np.histogram(df_5['actual'] - df_5['cohen'])
b_heights, b_bins = np.histogram(df_1['actual'] - df_1['cohen'], bins=a_bins)

width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label='$\sigma = 0.5$')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label='$\sigma = 1.0$')
fig.legend(loc='upper right')
plt.savefig('Hist_diff_actual_' + args.dataset)

plt.clf()
fig, ax = plt.subplots()

a_heights, a_bins = np.histogram(df_5['approx'] - df_5['cohen'])
b_heights, b_bins = np.histogram(df_1['approx'] - df_1['cohen'], bins=a_bins)

width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label='$\sigma = 0.5$')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label='$\sigma = 1.0$')
fig.legend(loc='upper right')
plt.savefig('Hist_diff_approx_' + args.dataset)

################
# Rolling relationship
################


radii = np.linspace(0, 1, 400)

rolling_5_mean_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='E0', mean=True) for rad in radii])
rolling_5_median_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='E0', mean=False) for rad in radii])
rolling_1_mean_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='E0', mean=True) for rad in radii])
rolling_1_median_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='E0', mean=False) for rad in radii])

plt.clf()
plt.plot(radii, rolling_5_mean_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_mean_E0, label='$\sigma = 1.0$')
plt.xlabel('$E_0$') # \pm 0.075$')
plt.ylabel('Mean Improvement')
plt.legend(loc='upper right')
plt.savefig('mean_E0')

plt.clf()
plt.plot(radii, rolling_5_median_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_median_E0, label='$\sigma = 1.0$')
plt.xlabel('$E_0$') #\pm 0.075$')
plt.ylabel('Median Improvement')
plt.legend(loc='upper right')
plt.savefig('median_E0')

radii = np.linspace(0, 1.5, 800)

rolling_5_mean_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='cohen', mean=True) for rad in radii])
rolling_5_median_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='cohen', mean=False) for rad in radii])
rolling_1_mean_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='cohen', mean=True) for rad in radii])
rolling_1_median_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='cohen', mean=False) for rad in radii])

plt.clf()
plt.plot(radii, rolling_5_mean_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_mean_E0, label='$\sigma = 1.0$')
plt.xlabel('$r_1$') #\pm 0.075$')
plt.ylabel('Mean Improvement')
plt.legend(loc='upper right')
#plt.ylim(0,100)
plt.savefig('mean_cohen')

plt.clf()
plt.plot(radii, rolling_5_median_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_median_E0, label='$\sigma = 1.0$')
plt.xlabel('$r_1$') #\pm 0.075$')
plt.ylabel('Median Improvement')
plt.legend(loc='upper right')
#plt.ylim(0,100)
plt.savefig('median_cohen')

## To percentage 


radii = np.linspace(0, 1, 400)

rolling_5_mean_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='E0', mean=True, percentage=True) for rad in radii])
rolling_5_median_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='E0', mean=False, percentage=True) for rad in radii])
rolling_1_mean_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='E0', mean=True, percentage=True) for rad in radii])
rolling_1_median_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='E0', mean=False, percentage=True) for rad in radii])

plt.clf()
plt.plot(radii, rolling_5_mean_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_mean_E0, label='$\sigma = 1.0$')
plt.xlabel('$E_0$') #\pm 0.075$')
plt.ylabel('Mean Percentage Improvement')
plt.legend(loc='upper right')
plt.savefig('mean_E0_percent')

'''radii = np.linspace(0, 1, 400)

rolling_5_mean_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='E0', mean=True, percentage=True) for rad in radii])
rolling_5_median_E0 = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='E0', mean=False, percentage=True) for rad in radii])
rolling_1_mean_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='E0', mean=True, percentage=True) for rad in radii])
rolling_1_median_E0 = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='E0', mean=False, percentage=True) for rad in radii])

plt.clf()
plt.plot(radii, rolling_5_mean_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_mean_E0, label='$\sigma = 1.0$')
plt.xlabel('$E_0 \pm 0.075$')
plt.ylabel('Mean Percentage Improvement')
plt.legend(loc='upper right')
plt.savefig('median_E0_percent_breakdown')'''


plt.clf()
plt.plot(radii, rolling_5_median_E0, label='$\sigma = 0.5$')
plt.plot(radii, rolling_1_median_E0, label='$\sigma = 1.0$')
plt.xlabel('$E_0$') #\pm 0.075$')
plt.ylabel('Median Percentage Improvement')
plt.legend(loc='upper right')
plt.ylim(0,125)
plt.savefig('median_E0_percent')

radii_co = np.linspace(0, 1.5, 800)

rolling_5_mean_co = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='cohen', mean=True, percentage=True) for rad in radii_co])
rolling_5_median_co = np.asarray([at_radius_rolling(df_5, rad, 'actual', key='cohen', mean=False, percentage=True) for rad in radii_co])
rolling_1_mean_co = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='cohen', mean=True, percentage=True) for rad in radii_co])
rolling_1_median_co = np.asarray([at_radius_rolling(df_1, rad, 'actual', key='cohen', mean=False, percentage=True) for rad in radii_co])

plt.clf()
plt.plot(radii_co, rolling_5_mean_co, label='$\sigma = 0.5$')
plt.plot(radii_co, rolling_1_mean_co, label='$\sigma = 1.0$')
plt.xlabel('$r_1$') #\pm 0.075$')
plt.ylabel('Mean Improvement')
plt.legend(loc='upper right')
plt.ylim(0,100)
plt.savefig('mean_cohen_percent')

plt.clf()
plt.plot(radii_co, rolling_5_median_co, label='$\sigma = 0.5$')
plt.plot(radii_co, rolling_1_median_co, label='$\sigma = 1.0$')
plt.xlabel('$r_1$') #\pm 0.075$')
plt.ylabel('Median Improvement')
plt.legend(loc='upper right')
plt.ylim(0,100)
plt.savefig('median_cohen_percent')


########################
# Outperformance of the boundary treatment
#########################

print('For sigma 0.5, Approx Boundary outperforms : {}, Full Boundary outperforms : {}'.format((df_5['mod'] > df_5['d_sim']).mean(), (df_5['f_mod'] > df_5['f_d_sim']).mean()))
print('For sigma 1.0, Approx Boundary outperforms : {}, Full Boundary outperforms : {}'.format((df_1['mod'] > df_1['d_sim']).mean(), (df_1['f_mod'] > df_1['f_d_sim']).mean()))

print('For sigma 0.5, Approx Double outperforms : {}, Full Double outperforms : {}'.format((df_5['d_sim'] > df_5['sim']).mean(), (df_5['f_d_sim'] > df_5['f_sim']).mean()))
print('For sigma 1.0, Approx Double outperforms : {}, Full Double outperforms : {}'.format((df_1['d_sim'] > df_1['sim']).mean(), (df_1['f_d_sim'] > df_1['f_sim']).mean()))
print('For sigma 0.5, Approx Boundary outperforms : {}, Full Boundary outperforms : {}'.format((df_5['mod'] > df_5['f_sim']).mean(), (df_5['f_mod'] > df_5['f_sim']).mean()))
print('For sigma 1.0, Approx Boundary outperforms : {}, Full Boundary outperforms : {}'.format((df_1['mod'] > df_1['f_sim']).mean(), (df_1['f_mod'] > df_1['f_sim']).mean()))



####### Time

def_out = pd.concat([df_5, df_1])
print(def_out[['cohen_t', 'sim_t', 'd_sim_t', 'f_sim_t', 'f_d_sim_t']].mean(), flush=True)

################
# N-Ball relationship, just because I'm curious
################
'''
d = 2000
R = 2

v_2 = 1
v_1 = 2*R

for i in range(2,2001):
    v_0 = (2*np.pi / i)*(R**2)*v_2
    v_2 = v_1
    v_1 = v_0
    if v_0 < 1e-20:
        print(i)
        break
'''

