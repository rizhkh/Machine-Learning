import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

## We are optimizing the click through rates for different users for different ads

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Solving multi arm bandit problem
#implementing the upper confidence bound algorithm

# Step 1: at each round n we consider two numbers for each selection i
d = 10 # d is the number of columns/ads in the data
number_of_selections = [0] * d # number_of_selections is known as Ni(n)
sum_of_rewards = [0] * d # sum_of_rewards is known as Ri(n)

ads_selected = [] #Step 3: ads_selected is from step 3
total_reward = 0 # Step 3 from algorithm description

# Step 2:
N=10000
for n in range(0,N): # this for look is for the 10k indexes
        ad = 0 # Step 3 - this is the index of ADS from dataset
        max_upper_bound = 0 # Step 3
        for i in range(0,d): # this for loop is for the 10 ads that exist
                if( number_of_selections[i] > 0): # this means if ad version of i was selected once this condition is used
                        average_reward = sum_of_rewards[i]/number_of_selections[i] # average_reward is ri(n)
                        delta_i = math.sqrt ( 3/2 *  math.log(n+1) / number_of_selections[i]) # delta_i is delta symbol Δ(n) = sqrt( 3/2 * log(n)/number_of_selections[i] )
                        # n+1 is because the data starts from 1 and here n was 0 so thats why 1 was added
                        upper_bound = average_reward + delta_i  # upper bound is ri(n)+Δ(n)
                else:
                     upper_bound = 1e400 # 10 to the power of 400 -> 10^400
                if upper_bound > max_upper_bound: # Step 3
                        max_upper_bound = upper_bound # Step 3
                        ad = i
        ads_selected.append(ad) # Step 3
        number_of_selections[ad] = number_of_selections[ad]  + 1
        reward = dataset.values[ n, ad ] # n is row from dataset and ad is the column from dataset
        sum_of_rewards[ad] = sum_of_rewards[ad] + reward
        total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Ads Selections')
plt.xlabel('Ads')
plt.ylabel('# of times Ad was selected')
plt.show()
