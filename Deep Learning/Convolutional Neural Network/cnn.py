from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D # to achieve pooling step : add pooling layers
from keras.layers import Flatten
from keras.layers import Dense # full connection - fully connected layers
from keras.preprocessing.image import ImageDataGenerator # For Part 2

### -----------------------------------------
# from keras.models import Sequential - to initialize the CNN - 2 ways to do it, as sequence of layers or graph
# As CNN is sequence of layers we use Sequential
#from keras.layers import Convolution2D - to add convulational layers - as we are working with images that are 2d we use 2d
# Convolution2D -> Conv2D
# if we use videos we use 3d as videos are 3D
# from keras.layers import MaxPooling2D - to achieve pooling step : add pooling layers
# from keras.layers import Dense - full connection - fully connected layers
### -----------------------------------------
# Steps include:
# initialize the CNN
# Convolution
# Pooling
# Add second layer of CNN
# Add flattening
# Full Connection
# Compile CNN
# THEN -> fit CNN to images

# Instructions for images
# separate it into two sets of folders: Test_set and Training_set
#       -> further in both folders make two separate folders in our case its cat and dog folder in both test_sey and training_set
#       -> this way Keras understands to differntiate the labels
### -----------------------------------------



# Part 1 - Building the CNN - Image classification to predict if image is a cat or a dog
#initializing the CNN
classifier = Sequential()

# CONVOLUTION
# obj.add( here goes the layer instance like Convultion2D or 3D) in keras adds a layers instance on top of the layer stack
#classifier.add( Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu') )

# POOLING
#classifier.add( MaxPooling2D( pool_size = (2,2) ) )

# Now adding a second convolutional layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# This for loop is to add two layers of convol2d
for i in range(0,2):
        if i== 0:
                classifier.add( Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu') )
                # Note: 32 above is the number of feature maps we are building
        else:
                classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add( MaxPooling2D( pool_size = (2,2) ) )

# Flattening
classifier.add( Flatten() ) # tak pooling layer and place them into a large input layer through flattening

# Full Connection
classifier.add( Dense( output_dim = 128, activation = 'relu' ) ) # output_dim is number of nodes in hidden layer
classifier.add( Dense( output_dim = 1, activation = 'sigmoid' ) )

# CNN compiled
classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

## Part 2 - Fitting the CNN to the images

## FOLLOWING CODE SNIPPET IS FROM KERAS DOCS -> PRE PROCESSING -> IMAGE PRPCESSING -> .flow_from_directory()
train_datagen = ImageDataGenerator(rescale=1./255, # all out pixels value are between 0 and 1 but not 0
                                   shear_range=0.2, # random_transactions
                                   zoom_range=0.2, # applying random transformations
                                   horizontal_flip=True) # images flipped horizontally

test_datagen = ImageDataGenerator(rescale=1./255) # rescale pixels of images in test set

training_set = train_datagen.flow_from_directory('dataset/training_set', #directory where images are
                                                target_size=(64, 64), #size of images expected in CNN model
                                                batch_size=32, #size of batches where random samples of imgs are includes including random number of images going through CNN
                                                class_mode='binary') # this determines whether our set class have different categories in our case we have two (cats and dogs)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                           target_size=(64, 64),
                                           batch_size=32,
                                           class_mode='binary')

#classifier.fit_generator(training_set,
#                   steps_per_epoch=8000, # as we have 8000 images in our folders
#                   epochs=25,
#                   validation_data=test_set,
#                   validation_steps=800) # numbers of images in our test set