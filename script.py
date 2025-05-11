# IMPORTS
import pandas as pd
import math
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Other Class, used in node class, from a5
class OtherKey:
    def __repr__(self):
        return 'OTHER'
OTHER = OtherKey()

# unique class, from a5
def unique(iterable):
    '''
    function that takes an iterable (list or tuple), and returns
    a tuple of two lists. The first is unique items in the iterable
    and second is count of that item.
    Ex. unique([1, 1, 2, 3, 2, 4, 5, 4, 1]) outputs:
    ([1, 2, 3, 4, 5], [3, 2, 1, 2, 1])

    iterable -> tuple
    '''
    items = list(iterable)
    unique_items = list(set(items))
    counts = [items.count(item) for item in unique_items]
    return unique_items, counts

# Node class
class Node:
    def __init__(self, attribute=None, children=None, classification=None):
        '''
        Intializer function
        Takes an attribute for the node, a dictionary of children nodes and a classification
        If it is a leaf node (no children) it will have a classification else None
        '''
        self.attribute = attribute
        self.classification = classification
        if children is None:
            self.children = {}
        else:
            self.children = children

    def classify(self, point):
        '''
        A function used to classify a point. Recursively calls until we reach a classification.
        Taken from a5.

        dict -> classification
        '''
        if self.classification is not None:
            return self.classification
        for child in self.children.keys():
            if point[self.attribute] == child:
                child_node = self.children[child]
                break
            else:
                child_node = self.children[OTHER]

        return child_node.classify(point)

    def train(self, data_points, labels):
        '''
        This will take a list of data points (each element is a dictionary which is a point)
        and a list of labels that goes with each point. Train the decision tree on these points
        and labels.

        Based off of a5

        list[dict], list -> trains decision tree
        '''
        unique_labels, label_counts = unique(labels)
        if len(unique_labels) == 1:
            self.classification = unique_labels[0]
            return
        
        best_attr = None
        best_vals = None
        best_points_by_val = None
        best_labels_by_val = None
        best_ig = None
        
        for attribute in data_points[0].keys():
            vals = set()
            points_by_val = {}
            labels_by_val = {}
            val_freqs = {}
            for point, label in zip(data_points, labels):
                val = point[attribute]

                if not val in vals:
                    vals.add(val)
                    val_freqs[val] = 0
                    points_by_val[val] = []
                    labels_by_val[val] = []

                val_freqs[val] += 1
                points_by_val[val].append(point)
                labels_by_val[val].append(label)

            least_frequent = min(vals, key=val_freqs.get)

            vals.remove(least_frequent)
            vals.add(OTHER)
            points_by_val[OTHER] = points_by_val.pop(least_frequent)
            labels_by_val[OTHER] = labels_by_val.pop(least_frequent)
            val_freqs[OTHER] = val_freqs.pop(least_frequent)

            information_gain = info_gain(labels, labels_by_val, val_freqs)

            if best_attr is None or information_gain > best_ig:
                best_attr = attribute
                best_vals = vals
                best_points_by_val = points_by_val
                best_labels_by_val = labels_by_val
                best_ig = information_gain

        # what if all of the points are the same?
        if len(best_vals) == 1:
            _, self.classification = max(zip(label_counts, unique_labels))
            return

        self.attribute = best_attr
        for val in best_vals:
            child = Node()
            child.train(best_points_by_val[val], best_labels_by_val[val])
            self.children[val] = child

# Entropy Function
def entropy(classifications):
    '''
    Calculate the entropy of an attribute. Takes a list of classifications in the form of: 
    [1, 4, 3, 5, 5, 3, 2, 1, 7] where each element represents the classification
    Works on string classifications too, Ex. ['Top 10', 'Win', 'Top 3']

    Rounded to four decimal points

    list -> float
    '''
    freq = collections.Counter(classifications)
    denom = len(classifications)
    total = 0
    for _, value in freq.items():
        total -= (value / denom) * math.log2(value / denom)
    return round(total, 4)

# Information Gain Function
def info_gain(parent_classifications, classifications_by_val, val_freqs):
    ''' 
    Calculates the information gain of an attribute. Classifications is a dictionary of attribute with 
    keys being a trait and values being the classifications. 
    Ex. Attribute = Top 10 percent fastest lap time?
        {'Yes' : [Top 5, Top 3, Top 10],
        'No' : [Top 15, Top 20, Top 10]}
    
    Parent entropy is the entropy value for the parent. Needed to calculate the information gain.
    
    The classifications dictionary should be the data of the parent node after being split by the attribute
    that the classification dictionary represents.

    Rounds to four decimal places.
                                                            
    dict, float -> float
    '''
    attribute_entropy = 0
    total = sum([item for item in val_freqs.values()])
    for att in classifications_by_val.keys():
        attribute_entropy += entropy(classifications_by_val[att]) * (val_freqs[att] / total)
    
    information_gain = entropy(parent_classifications) - attribute_entropy

    return information_gain

########################
### GETTING OUR DATA ###
########################

# loading constructor data into a dictionary
# this will be used to get the nationality of a constructor
constructors_dict = {}
constructors = pd.read_csv('kaggle-data/archive/constructors.csv')
constructors = constructors.dropna()

for index, row in constructors.iterrows():
    key = row['constructorId']
    constructors_dict[key] = row['nationality']

# loading our race data into a dictionary
# the key will be (raceId, driverId)
# value will be [raceId, driverId, grid, constructor nationality, avg finish over last 5 races, classification]
data_dict = {}

# getting results data
results = pd.read_csv('kaggle-data/archive/results.csv') # results data
results = results[results['grid'] != 0]
results = results.dropna()

for index, row in results.iterrows():
    key_tuple = (row['raceId'], row['driverId']) # tuple for key in data dictionary
    value_list = [row['raceId'], row['driverId']]
    
    # checking for classification
    if row['positionOrder'] == 1: # first place 
        classification = 'First'
    elif row['positionOrder'] <= 5: # top 5
        classification = 'Top 5'
    elif row['positionOrder'] <= 10: # top 10
        classification = 'Top 10'
    else: # outside top 10
        classification = 'Outside Top 10'

    # turn grid position into range
    if row['grid'] <= 3:
        grid = 'Front'
    elif row['grid'] <= 10:
        grid = 'Middle'
    else:
        grid = 'Back'

    value_list.append(grid)
    
    # get last 5 races
    driver_id = row['driverId']
    race_id = row['raceId']

    past_races = results[(results['driverId'] == driver_id) & 
                         (results['raceId'] < race_id)]

    # Sort descending so we get most recent first
    past_races = past_races.sort_values(by='raceId', ascending=False)

    past_positions = past_races['positionOrder'].head(5)

    if len(past_positions) == 5:
        avg_pos = past_positions.mean()
    else:
        avg_pos = None  # not enough history

    # only add if we have data for last 5 races
    if avg_pos is not None:
        if avg_pos <= 5:
            average_finish = 'Great'
        elif avg_pos <= 10:
            average_finish = 'Good'
        elif avg_pos <= 15:
            average_finish = 'Average'
        else:
            average_finish = 'Bad'

        construct = row['constructorId']
        nation = constructors_dict[construct]
        value_list.append(nation)
        value_list.append(average_finish)
        value_list.append(classification)

        data_dict[key_tuple] = value_list

''' 
One data point will be a dictionary like so
Ex. {'grid': grid position,
    'nationality': constructor nationality,
    'avg_finish': average finish}
'''

####################################
### CREATING TRAIN AND TEST SETS ###
####################################

points = []
labels = []

test_points = []
target_labels = []

# 2014 - 2023 will be the training set, we will test on 2024 seasons and some 2025 races
# the raceId for first race of 2014 is 900 to 1120 for last race of 2023 (220 races)
# everything after 1120 is 2024 season
for key, value in data_dict.items():
    point = {}
    point['grid'] = value[2]
    point['nationality'] = value[3]
    point['avg_finish'] = value[4]
    label = value[5] # classification

    # check which set we add it to
    if key[0] >= 900 and key[0] <= 1120: # add to the training set
        points.append(point)
        labels.append(label)
    elif key[0] > 1120: # add to the test set
        test_points.append(point)
        target_labels.append(label)

############################
### TRAINING AND TESTING ###
############################

tree = Node()
tree.train(points, labels)

# function for analysis of results
def analysis(y_pred, y_true):
    '''
    takes a list of predicted points and true labels and uses
    sklearn.metrics to analyze the results.
    '''
    print('\nClassification Report:\n')
    print(classification_report(y_true, y_pred))
    print('\nConfusion Matrix:\n')
    cm = confusion_matrix(y_true, y_pred, labels=['Outside Top 10', 'Top 10', 'Top 5', 'First'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Outside Top 10', 'Top 10', 'Top 5', 'First'])
    disp.plot()
    plt.show()
    return

predictions = []
correct = 0
for point, label in zip(test_points, target_labels):
    prediction = tree.classify(point)
    predictions.append(prediction)
    if prediction == label:
        correct += 1

print('\nAnalysis of results using 2014-2023 season as training set and 2024 Season as testing set')
analysis(predictions, target_labels)
print('\n A small analysis of our results: \n' + 
      'We have a pretty decent precision when it comes to first place winners. When we predict a driver to get first ' +
      '50 percent of the time that driver does get first. On prediction outside top 10, we have 85% precision ' + 
      'So 85 percent of the time we predict a driver to get outside top 10 they do.\n')
print('When it comes to recall for first place winners we correctly predicted over 30 percent of the real winners which ' + 
      'is not bad. Our recall was best for outside top 10 and top 5 which was 83 percent and 78 percent respectively.')

# Predict the saudi grand prix results
# remember one data point is in the form of 
# {grid: starting grid position, nationality: nationality of constructor, avg_finish: average finish over last 5 races}
# predict using tree.calssify(point) and then compare to target label 
# once you get a list of predictions and target labels use analysis(y_pred, y_true) for analysis report


# Today’s Miami Grand Prix features
miami_gp_data = {
    'Max Verstappen':   {'grid':'Front',  'nationality':'NED', 'avg_finish':'Great'},
    'Lando Norris':     {'grid':'Front',  'nationality':'GBR', 'avg_finish':'Great'},
    'Kimi Antonelli':   {'grid':'Front',  'nationality':'ITA', 'avg_finish':'Good'},
    'Oscar Piastri':    {'grid':'Middle', 'nationality':'AUS', 'avg_finish':'Great'},
    'George Russell':   {'grid':'Middle', 'nationality':'GBR', 'avg_finish':'Great'},
    'Carlos Sainz':     {'grid':'Middle', 'nationality':'ESP', 'avg_finish':'Average'},
    'Alexander Albon':  {'grid':'Middle', 'nationality':'GBR', 'avg_finish':'Average'},
    'Charles Leclerc':  {'grid':'Middle', 'nationality':'MON', 'avg_finish':'Good'},
    'Esteban Ocon':     {'grid':'Middle', 'nationality':'FRA', 'avg_finish':'Average'},
    'Yuki Tsunoda':     {'grid':'Middle', 'nationality':'JPN', 'avg_finish':'Average'},
    'Isack Hadjar':     {'grid':'Back',   'nationality':'FRA', 'avg_finish':'Average'},
    'Lewis Hamilton':   {'grid':'Back',   'nationality':'GBR', 'avg_finish':'Good'},
    'Gabriel Bortoleto':{'grid':'Back',   'nationality':'BRA', 'avg_finish':'Bad'},
    'Jack Doohan':      {'grid':'Back',   'nationality':'AUS', 'avg_finish':'Average'},
    'Liam Lawson':      {'grid':'Back',   'nationality':'NZL', 'avg_finish':'Average'},
    'Nico Hulkenberg':  {'grid':'Back',   'nationality':'GER', 'avg_finish':'Average'},
    'Fernando Alonso':  {'grid':'Back',   'nationality':'ESP', 'avg_finish':'Average'},
    'Pierre Gasly':     {'grid':'Back',   'nationality':'FRA', 'avg_finish':'Average'},
    'Lance Stroll':     {'grid':'Back',   'nationality':'CAN', 'avg_finish':'Average'},
    'Oliver Bearman':   {'grid':'Back',   'nationality':'GBR', 'avg_finish':'Average'},
}

# The order they qualified in:
qual_order = [
    'Max Verstappen','Lando Norris','Kimi Antonelli','Oscar Piastri','George Russell',
    'Carlos Sainz','Alexander Albon','Charles Leclerc','Esteban Ocon','Yuki Tsunoda',
    'Isack Hadjar','Lewis Hamilton','Gabriel Bortoleto','Jack Doohan','Liam Lawson',
    'Nico Hulkenberg','Fernando Alonso','Pierre Gasly','Lance Stroll','Oliver Bearman'
]

print("\nMiami GP Predictions (qualifying order):")
preds = []
for pos, driver in enumerate(qual_order, start=1):
    feat = miami_gp_data[driver]
    p = tree.classify(feat)
    preds.append(p)
    print(f"{pos:2d}. {driver:20s} → {p}")


