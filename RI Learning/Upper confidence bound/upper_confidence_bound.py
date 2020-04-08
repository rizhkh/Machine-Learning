import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

## We are optimizing the click through rates for different users for different ads

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\Rizwan\\Desktop\\MachineLearning\\Machine Learning A-Z Template Folder\\Part 6 - Reinforcement Learning\\Section 32 - Upper Confidence Bound (UCB)\\Ads_CTR_Optimisation.csv')

#Solving multi arm bandit problem
#implementing the upper confidence bound algorithm

# Step 1: at each round n we consider two numbers for each selection i
d = 10 # d is the number of columns/ads in the data
number_of_selections = [0] * d # number_of_selections is known as Ni(n)
sum_of_rewards = [0] * d # sum_of_rewards is known as Ri(n)

ads_selected = [] #Step 3: ads_selected is from step 3

# Step 2:
N=10000
for n in range(0,N): # this for look is for the 10k indexes
        
        max_upper_bound = 0 # Step 3
        
        ad = 0 # Step 3
        
        for i in range(0,d): # this for loop is for the 10 ads that exist
                
                average_reward = sum_of_rewards[i]/number_of_selections[i] # average_reward is ri(n)
                
                delta_i = math.sqrt ( 3/2 *  math.log(n+1)/number_of_selections[i]) # delta_i is delta symbol Δ(n) = sqrt( 3/2 * log(n)/number_of_selections[i] )
                # n+1 is because the data starts from 1 and here n was 0 so thats why 1 was added
                
                upper_bound = average_reward + average_reward  # upper bound is ri(n)+Δ(n)
                
                if upper_bound > max_upper_bound: # Step 3
                        
                        max_upper_bound = upper_bound # Step 3
                        ad = i
        





#
## Implementing UCB
#import math
#N = 10000
#d = 10
#ads_selected = []
#numbers_of_selections = [0] * d
#sums_of_rewards = [0] * d
#total_reward = 0
#for n in range(0, N):
#    ad = 0
#    max_upper_bound = 0
#    for i in range(0, d):
#        if (numbers_of_selections[i] > 0):
#            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
#            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
#            upper_bound = average_reward + delta_i
#        else:
#            upper_bound = 1e400
#        if upper_bound > max_upper_bound:
#            max_upper_bound = upper_bound
#            ad = i
#    ads_selected.append(ad)
#    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
#    reward = dataset.values[n, ad]
#    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
#    total_reward = total_reward + reward
#
## Visualising the results
#plt.hist(ads_selected)
#plt.title('Histogram of ads selections')
#plt.xlabel('Ads')
#plt.ylabel('Number of times each ad was selected')
#plt.show()