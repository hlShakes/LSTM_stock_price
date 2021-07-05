import random
import re
import copy
import numpy as npy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import StockNeuralNet as stk

# Open all pre formatted stock data including: Open, High, Low, Close in that order 
data_file = open("preppedstock.csv", 'r')
# Read all of the lines from the file into memory
data_list = data_file.readlines()
# Close the file (we are done with it)
data_file.close()

def crtIndividual (min, max): # input list of max and mins for each variable to be optimised
    # return a list of containing a value for each variable to be optimised
    # max and min must be integers for non integer requirements multiply out later
    individual = []
    i = 0
    if len(min) != len(max):
        print('there appears to be an error with your inputs to crtIndividual (length min != length max')
        pass
    for x in max:
        individual.append(random.randrange(min[i],x)) # only works for integer numbers 
        i+=1
        pass
    return individual

#test crtIndividual

def ranking (pop, target, results): # order population in order of their results 
    # should return pop[0] is best and pop[end] is worst
    i = 0
    error = []
    for x in results:
        locError = target - x
        error.append(abs(locError))
        i+=1
        pass
    zipped = list(zip(error,pop))
    zipped.sort()
    #print(zipped[0])
    error, pop = zip(*zipped)
    return error, pop

# test ranking function 
#pop = [['a','b','c'],['a1','b1','c1'],['a2','b2','c2'],['a3','b3','c3'],['a4','b4','c4'],['a5','b5','c5']]
#target = 30
#results = [15,75,32,6,42]
#error, pop = ranking (pop, target, results)
#print (pop)
#print(error)

def evolve(pop, target, results, min, max):
    # rank population 
    rankErr, rankPop = ranking (pop, target,results)
    #replace bottom 30% of population
    perc70 = round(len(rankPop)*0.7)
    # breed top 70% with each other, leaving top choice unchanged
    remainingPop = perc70
    rankPop1 = copy.deepcopy(rankPop)
    newPop = list(rankPop1) 
   
    perc25 = 3#round(len(rankPop)*0.25)
    for i in range(1,remainingPop+1): 
        
        parent1 = copy.copy(rankPop[i]) # first populus
        # take at least 1 characteristic from top 25%
        random25 = random.randrange(0,perc25+1) # randomly select top 25 element to take from
        while random25 == i:
            random25 = random.randrange(0,perc25+1)
            pass
        parent2 = copy.copy(rankPop[random25])
        child = parent2
        for x in range(1,len(rankPop[0])):# change 1 to n-1 elements in set
            character = random.randrange(0,len(rankPop[0]))
            child[character] = parent1[character]
            pass
        newPop[i] = child
        pass
    
    for i in range(perc70+1,len(rankPop)):
        newPop[i] = crtIndividual (min, max)
        pass

    return newPop, rankErr


# test ranking function 
#pop = [['a','b','c'],['a1','b1','c1'],['a2','b2','c2'],['a3','b3','c3'],['a4','b4','c4'],['a5','b5','c5']]
#target = 30
#results = [15,75,32,6,42]
##error, pop = ranking (pop, target, results)
##print(error)
#newPop = evolve(pop,target,results,[5,2,1], [30,32,42])
#print(newPop)

pop1 = []
for a in range (1, 15):
    days = a*10
    iterations = a*30
    pop1.append([days,iterations])
    pass
pop = copy.deepcopy(pop1)
print(pop)
min = [5,10]
max = [250,300]
counter = 1
error = []
minerror = 10000
bestCombination = []
bestCombination = copy.deepcopy(pop1[1])
while counter<=100 and minerror >= 15:
    predictions = []
    for i in range(0,len(pop)):
        row = pop[i]
        [training_target,training_data] = stk.dataPrep(row[1],row[0],data_list) # call function to define training data 
        training_shape = training_data.shape
        #print(training_shape)
        target_shape = training_target.shape
        #print(target_shape)
        [data_max, data_min] = stk.maxMin(data_list)
        #print(training_target)
            # normalise data 
        for k in npy.nditer(training_data, op_flags=['readwrite']):
            k[...] = ((k-data_min)/(data_max-data_min))
            pass
        for k in npy.nditer(training_target, op_flags=['readwrite']):
            k[...] = ((k-data_min)/(data_max-data_min))
            pass
       #print(training_data)

        xshape = [training_shape[1],training_shape[2]] 
        xshape = npy.array(xshape)

        model = Sequential()
        model.add(LSTM(64,input_shape=xshape,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64,return_sequences=True))
        model.add(LSTM(64))

        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error',optimizer = 'adamax')
        model.fit (x = training_data, y= training_target, epochs=5)
        training_score = model.evaluate(x = training_data, y= training_target)
        print(training_score)

        #define test parameters
        [test_target,test_data] = stk.colTestData(row[0],data_list) # call function to define training data 

        # normalise data 
        for k in npy.nditer(test_data, op_flags=['readwrite']):
            k[...] = ((k-data_min)/(data_max-data_min))
            pass


        test_score = model.predict(x = test_data)
        print(test_score) 
        predictedCost = test_score*(data_max-data_min)+data_min

        predictions.append(predictedCost)
        print('current pop iteration = ', pop[i])
        print('predicted cost = ', predictedCost)
        pass
    
    error, pop = ranking(pop,actualCost,predictions)
    print('error in order =', error,file=open("output.txt", "a"))
    print('ranked population =', pop,file=open("output.txt", "a"))

    locMinError = error[0]
    if minerror >= locMinError:
        bestCombination = copy.deepcopy(pop[0])
        minerror = copy.deepcopy(locMinError)
        print('New Minimum Error =', minerror, file=open("output.txt", "a"))
        pass
    
    pop, error = evolve(pop,actualCost,predictions,min,max)
    counter +=1
    print('counter =', counter)
    pass
print('Final Best Set =', bestCombination, file=open("output.txt", "a"))
    
   
