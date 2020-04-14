# Thompson Sampling

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Solving multi arm bandit problem
#implementing Thomas Sampling

# Step 1:
d = 10 # d is the number of columns/ads in the data
number_of_reward_one = [0] * d # number_of_selections is known as Ni1(n)
number_of_reward_zero = [0] * d # sum_of_rewards is known as Ni0(n)

ads_selected = [] #Step 3: ads_selected is from step 3
total_reward = 0 # Step 3 from algorithm description

# Step 2:
N=10000
for n in range(0,N): # this for look is for the 10k indexes
        ad = 0 # Step 3 - this is the index of ADS from dataset
        max_random = 0 # Step 2: random draw from step 2
        for i in range(0,d): # this for loop is for the 10 ads that exist
                random_Beta = random.betavariate(number_of_reward_one[i]+1 , number_of_reward_zero[i]+1) # Step 2: θi(n) = B( Ni1(n) + 1 , Ni0(n) + 1)
                if random_Beta > max_random: # Step 3
                        max_random = random_Beta # Step 3
                        ad = i
        ads_selected.append(ad) # Step 3
        reward = dataset.values[ n, ad ] # n is row from dataset and ad is the column from dataset
        if reward == 1:
                number_of_reward_one[ad] = number_of_reward_one[ad] + 1
        else:
                number_of_reward_zero[ad] = number_of_reward_zero[ad] + 1
        total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Ads Selections')
plt.xlabel('Ads')
plt.ylabel('# of times Ad was selected')
plt.show()
