
# coding: utf-8

# # Spark Assignment
# * Xuyan Xiao, xx2226
# * Junhui Liao, jl4574

# ### Data Preparation

# In[1]:
from pyspark import SparkContext
sc = SparkContext(appName="name")
# user and artist
rawUserArtistData = sc.textFile("s3://aws-logs-546206774149-us-east-1/audio_data/user_artist_data.txt")

# use a sample to train the model
# weights = [.1, .9]
# seed = 42
# rawUserArtistData, someOtherJunk = rawUserArtistData.randomSplit(weights, seed)
# rawUserArtistData.cache()

# print rawUserArtistData.map(lambda l: float(l.split(" ")[0])).stats()
# print rawUserArtistData.map(lambda l: float(l.split(" ")[1])).stats()


# In[2]:

# artist data
rawArtistData = sc.textFile("s3://aws-logs-546206774149-us-east-1/audio_data/artist_data.txt")

def lineSplit(artist):
    line = artist.split("\t")
    if len(line) < 2:
        return []
    else:
        try:
            return [(int(line[0]),line[1].strip())]
        except:
            return []

artistByID = rawArtistData.flatMap(lambda l: lineSplit(l))


# In[3]:

# artist misspelled
rawArtistAlias = sc.textFile("s3://aws-logs-546206774149-us-east-1/audio_data/artist_alias.txt")

def lineSplitAlias(artist):
    line = artist.split("\t",1)
    if line[0] == "":
        return []
    else:
        return [(int(line[0]),int(line[1]))]

artistAlias = rawArtistAlias.flatMap(lambda l: lineSplitAlias(l)).collectAsMap()


# In[14]:




# ### build the first recommendation model

# In[4]:

from pyspark.mllib.recommendation import *
bArtistAlias = sc.broadcast(artistAlias)

def trainDataConstuct(line):
    userID, artistID, count = map(int,line.split(" "))
    finalArtistID = bArtistAlias.value.get(artistID,artistID)
    return Rating(userID, finalArtistID, count)

trainData = rawUserArtistData.map(lambda l: trainDataConstuct(l))
trainData.cache()


# In[5]:

model = ALS.trainImplicit(trainData, 10, 5, 0.01)


# In[11]:

# look at the first line of the model
# ft = model.userFeatures().collect()
# (ft[0][0],)+tuple([it for it in ft[0][1]])

rawArtistsForUser = trainData.filter(lambda u: u.user == 2093760)

# take a look at all the artists the given user has listened to
# t = rawArtistsForUser.collect()
# print t

# find the unique artisits
existingProducts = set(rawArtistsForUser.map(lambda m: int(m.product)).collect())




# In[12]:

# recommendations = model.recommendProducts(1000002, 10) not included in pysprak 1.3.1
recommendations = model.call("recommendProducts", 2093760, 10)
recommendedProductIDs = set([item.product for item in recommendations])


# In[13]:

# match with the artists' names and print out
res = artistByID.filter(lambda l: l[0] in recommendedProductIDs).values().collect()
for artist in res:
    print artist

