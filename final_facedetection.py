#importing the required libraries

import sys
import os
import glob
import cv2
import dlib
import itertools
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import math
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

#defining useful variables
BASE_FOLDER_PATH = os.getcwd()
print(f"\ncurrent working directory: {BASE_FOLDER_PATH}\n\n")

FOLDER_PATH = BASE_FOLDER_PATH + '/images/'
DATASET_PATH = BASE_FOLDER_PATH + '/dataset/'
HOG_DATASET_PATH = BASE_FOLDER_PATH + '/hogdataset/'
LOG_PATH = BASE_FOLDER_PATH + '/log/'
MODEL_PATH = BASE_FOLDER_PATH+ '/model/'
MODEL_LABELS_PATH = BASE_FOLDER_PATH + '/labels/'
NEW_IMAGES_PATH = BASE_FOLDER_PATH + '/new images/'
PLOTS_PATH = BASE_FOLDER_PATH + '/plots/'
EPOCHS = 25
BATCH_SIZE = 32
IMAGE_SIZE = 64
FINAL_CELL = 5
FINAL_BLOCK = 3
CELL_SIZE = 5
BLOCK_SIZE = 3
MAKE_DIR = 'mkdir -p -m 777 '

#Function to extract the faces and HOG feature

def create_face_dataSet(data_folder):
    """
    This function finds the face in a given image which is read from the folder path passed as a parameter to the function,
    crops the image to the area that contains face and stores the cropped image into dataset folder.
    """
    hogDetector = dlib.get_frontal_face_detector()   
    if not os.path.exists(DATASET_PATH):
    	os.system(MAKE_DIR+'"'+DATASET_PATH+'"')
    
    total_files_count = len(os.listdir(FOLDER_PATH))
    print(f"\nFile path: {FOLDER_PATH}")
    print(f"\nTotal files in path: {total_files_count}")
    for file in os.listdir(FOLDER_PATH):
        sub_folder = os.path.join(FOLDER_PATH,file)
        no_of_files = len(os.listdir(sub_folder))
        for ind,sub_file in enumerate(os.listdir(sub_folder)):
            f = os.path.join(sub_folder,sub_file)
            image = cv2.imread(f)
            #plt.imshow(image)
            #plt.title('my picture')
            #plt.show()
            #print(f"filename: {f}")
            print_progress(ind, no_of_files, file)
            dataset_subpath = DATASET_PATH + file
            if not os.path.exists(dataset_subpath):
            	os.system(MAKE_DIR+'"'+dataset_subpath+'"')
                            
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
            face = hogDetector(gray_image,1)
            for rect in face:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                
                if x < 0:
                    x=0
                    
                if y<0:
                    y=0
                    
                if h<0:
                    h=0
                    
                if w<0:
                    w=0
                    
                cropped_image = image[y:h, x:w]
                #img = cv2.resize(cropped_image,(IMAGE_SIZE,IMAGE_SIZE))
                cv2.imwrite(os.path.join(dataset_subpath , sub_file), cropped_image)

def create_hog_dataSet(data_folder):
    """
    This function finds the face in a given image which is read from the folder path passed as a parameter to the function,
    crops the image to the area that contains face, extracts the hog features and stores hog features image into dataset folder.
    """
    hogDetector = dlib.get_frontal_face_detector()   
    if not os.path.exists(HOG_DATASET_PATH):
    	os.system(MAKE_DIR+'"'+HOG_DATASET_PATH+'"')

    total_files_count = len(os.listdir(FOLDER_PATH))
    print(f"\nFile path: {FOLDER_PATH}")
    print(f"\nTotal files in path: {total_files_count}")
    for file in os.listdir(FOLDER_PATH):
        sub_folder = os.path.join(FOLDER_PATH,file)
        no_of_files = len(os.listdir(sub_folder))

        for ind,sub_file in enumerate(os.listdir(sub_folder)):
            f = os.path.join(sub_folder,sub_file)
            image = cv2.imread(f)
            #plt.imshow(image)
            #plt.title('my picture')
            #plt.show()
            #print(f"filename: {f}")
            print_progress(ind, no_of_files, file)
            dataset_subpath = HOG_DATASET_PATH + file
            if not os.path.exists(dataset_subpath):
            	os.system(MAKE_DIR+'"'+dataset_subpath+'"')
                            
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
            face = hogDetector(gray_image,1)
            for rect in face:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                
                if x < 0:
                    x=0
                    
                if y<0:
                    y=0
                    
                if h<0:
                    h=0
                    
                if w<0:
                    w=0
                    
                cropped_image = image[y:h, x:w]
                features,hg = hog(cropped_image,orientations=9,pixels_per_cell=(CELL_SIZE,CELL_SIZE),cells_per_block=(BLOCK_SIZE,BLOCK_SIZE), visualize=True, channel_axis=-1)
                hg_rescaled = exposure.rescale_intensity(hg, in_range=(0,10))
                plt.imshow(hg_rescaled, cmap='gray')
                plt.axis('off')
                plt.savefig(os.path.join(dataset_subpath , sub_file), bbox_inches='tight',pad_inches=0)
                #plt.show()
                #cv2.imwrite(os.path.join(dataset_subpath , sub_file), hg_rescaled)
                
#This function is used when the dataset is small to train the neural network, so we generate extra images
def image_augmentation(img):
    """
    This function create 20 extra images from a given image by transforming, changing the rotation of the images.
    It returns the array of images created by augmentation.
    """
    height, width = img.shape[0:2]
    center = (width // 2, height // 2)
    rotation_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
    rotation_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
    rotation_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotation_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
    transform_3 = np.float32([[1, 0, 3], [0, 1, 0]])
    transform_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
    transform_6 = np.float32([[1, 0, 6], [0, 1, 0]])
    transform_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
    transform_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
    transform_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
    transform_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
    transform_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])
    
    imgs = []
    imgs.append(cv2.warpAffine(img, rotation_5, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, rotation_neg_5, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, rotation_10, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, rotation_neg_10, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_3, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_neg_3, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_6, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_neg_6, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_y3, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_neg_y3, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_y6, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, transform_neg_y6, (width, height), borderValue=(255,255,255)))
    imgs.append(cv2.add(img,  10))
    imgs.append(cv2.add(img,  30))
    imgs.append(cv2.add(img, -10))
    imgs.append(cv2.add(img, -30)) 
    imgs.append(cv2.add(img,  15))
    imgs.append(cv2.add(img,  45))
    imgs.append(cv2.add(img, -15))
    imgs.append(cv2.add(img, -45))
    
    return imgs
    
#For printing the progess of a task
def print_progress(current, no_of_files, folder_name):
    """
    Prints the progress of a task
    """
    progressbar_size = 10
    progress = "#"*round((current)*progressbar_size/no_of_files) + " "*round((no_of_files - (current))*progressbar_size/no_of_files)
    if current == 0:
        print("", end = "\n")
    else:
        print("[%s] (%d samples)\t label : %s \t\t" % (progress, current+1, folder_name), end="\r")


#For loading images from the dataset path
def load_images(dataset_path):
    """
    This function is used to load the images from the dataset floder and return the images and class names array
    """
    images=[]
    class_names=[]
    for folder in os.listdir(dataset_path):
        files = os.listdir(os.path.join(dataset_path,folder))
        for ind,name in enumerate(files):
            img = cv2.imread(os.path.join(dataset_path,folder,name))
            img = cv2.resize(img, (50, 50))
            images.append(img)
            class_names.append(folder)
        
            print_progress(ind, len(files), folder)
    return zip(images,class_names)

#To test the augmented images generated
def test_print_augmented_images(images):
    """
    This function takes images array as input and prints the sample augmentation of image
    """
    img_test = images[1]
    augmented_image_test = image_augmentation(img_test)
    plt.figure(figsize=(15,10))
    for i, img in enumerate(augmented_image_test):
        plt.subplot(4,5,i+1)
        plt.imshow(img, cmap="gray")
    plt.savefig(os.path.join(PLOTS_PATH,'Augmented.jpg'))
    plt.show()



#Creates Augmented images
def create_augmented_images(images,class_names):
    new_images = []
    new_class_labels = []
    for ind, image in enumerate(images):
        try :
            new_images.extend(image_augmentation(image))
            new_class_labels.extend([class_names[ind]] * 20)
        except :
            print(ind)
    return zip(new_images,new_class_labels)

#Prints the counts of images
def print_images_counts(class_names):
    """
    This function prints the number of images in each class label
    """
    unique, counts = np.unique(class_names, return_counts = True)
    for item in zip(unique, counts):
        print(item)
        

# Give the pie chart distribution of images
def get_data_distribution(class_names,plt_name):
    """
    This function is used to create a pie chart distribution of the different class labels based on number of images each class contains
    """
    label_name = np.unique(class_names)
    label_distribution = {i:class_names.count(i) for i in class_names}.values()
    
    plt.figure(figsize=(12,6))

    my_circle = plt.Circle( (0,0), 0.7, color='white')
    plt.pie(label_distribution, labels=label_name, autopct='%1.1f%%')
    plt.gcf().gca().add_artist(my_circle)
    plt.savefig(os.path.join(PLOTS_PATH,plt_name+'.jpg'))
    plt.show()


# reduce sample size per-class using numpy random choice
def randc(labels, l, n):
    return np.random.choice(np.where(np.array(labels) == l)[0], n, replace=False)

# Balancing the dataset from each class
def balance_dataset(class_names):
    """
    This funcion balances out the number of images from class to have same number of counts and returns the mask of randomly selected images
    which then used on images array.
    """
    unique, counts = np.unique(class_names, return_counts = True)
    n =  math.floor(min(counts) / 100) * 100
    mask = np.hstack([randc(class_names, l, n) for l in np.unique(class_names)])
    return mask

# This is used to encode the class names to categorical
def encode_label_names(class_names):
    """
    This function converts the encode the classes to categorical vector which is used in training the model
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    name_vect = label_encoder.transform(class_names)
    categorical_name_vect = to_categorical(name_vect)
    
    return categorical_name_vect

# Returns the labels list for class
def get_encode_label_names(class_names):
    """
    This function takes the class names and returns a list of class labels
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    labels_list = label_encoder.classes_
    
    return labels_list

# Creats model
def create_cnn_model(input_shape,model_name,labels_list):
    """
    This function takes model name, labels, input shape as parameters and creates a CNN model and returns the model created
    """
    model = Sequential(name=model_name)
    
    model.add(Conv2D(64,(3,3),padding="valid", activation="relu", input_shape=input_shape))
    model.add(Conv2D(64,(3,3),padding="valid", activation="relu", input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3,3), padding="valid", activation="relu"))
    model.add(Conv2D(128, (3,3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(labels_list)))           # equal to number of classes
    model.add(Activation("softmax"))
    
    model.summary() 
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

# Plots the metrics on graphs
def plot_metrics(hist, x_var,y_var,x_label,y_label,title):
    """
    This function creates the plot between the x and y variable passed as the parameter to the function.
    """
    fig = plt.figure()
    plt.plot(hist.history[x_var], color='teal', label=x_label)
    plt.plot(hist.history[y_var], color='orange', label=y_label)
    fig.suptitle(title, fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(PLOTS_PATH,title+'.jpg'))
    plt.show()

# Pre-processes the images before sending it to model for predicting
def preprocess_new_images(path):
    """
    This function pre-processes the images by detecting the face, cropping the image and reducing the size of the image and then this is passed
    to the model to predict the class this pre-processed image belongs to.
    """
    hogDetector = dlib.get_frontal_face_detector()
    imgs = []
    names = []
    files = os.listdir(path)
    print(f"folder: path contains files: {len(files)}")
    for file in files:
        
        img = cv2.imread(os.path.join(path,file))
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        face = hogDetector(gray_image,1)
        for i,rect in enumerate(face):
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
                
            if x < 0:
                x=0
                    
            if y<0:
                y=0
                    
            if h<0:
                h=0
                    
            if w<0:
                w=0
                    
            cropped_image = img[y:h, x:w]
            resized_image = cv2.resize(cropped_image, (50, 50))
            imgs.append(resized_image)
            name=''
            if '_' in file:
                name = file.split('_')[0]
            else:
                name = file.split('.')[0]
            
            names.append(name)
        
    return zip(imgs,names)

# Prints the model evalution
def get_model_evaluation(model,test_data,test_labels):
    """
    This function takes model, data and labels for the data, performs evalutions and prints the loss and accuracy of the model
    """
    loss, accuracy = model.evaluate(test_data,test_labels,verbose=1)
    print('-------------------------')
    print(f'| Model Loss    | {round(loss,3)} |')
    print('-------------------------')
    print(f'| Model Accuracy| {round(accuracy,3)} |')
    print('-------------------------')

# This function is used for predicting class for new set of images
def new_image_prediction(path,my_model,labels_list):
    """
    This function call the pre-process images function to and shows the actual to predicted class along with the image in the output
    which makes it easy to analyse the output.
    """
    img_test_data,img_name_data = zip(*preprocess_new_images(path))
    files_count = len(os.listdir(path))
    
    catg_names = encode_label_names(img_name_data)

    test_data = np.array(img_test_data, dtype=np.float32)
    test_data_1 = test_data.copy()
    test_name = np.array(catg_names)
    fig=plt.figure(figsize=(12, 12))
    columns = 2
    rows = 5
    #get_model_evaluation(my_model,test_data,test_name)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        rand_n = i-1
        if rand_n == 15:
            break
        print(f'rand_n: {rand_n}')
        plt.imshow(test_data[rand_n][:, :, 0], cmap='gray')
        plt.title(f'Actual:{img_name_data[rand_n]} Predicted: {labels_list[np.argmax(my_model.predict(test_data[rand_n].reshape(-1, 50, 50, 3)))]}');
        plt.axis('off')
    fig.savefig(os.path.join(PLOTS_PATH,'New_Prediction.jpg'))
    get_model_evaluation(my_model,test_data_1,test_name)
    
    class_prediction = my_model.predict(test_data)
    cnf_matrix = confusion_matrix(test_name.argmax(axis=1), class_prediction.argmax(axis=1))
    np.set_printoptions(precision=2)


    # Plot non-normalized confusion matrix
    plot_confusion_map(cnf_matrix, labels_list, normalize=False, title='Confusion Matrix',file_name='New_pred_Confusion Matrix.jpg',cmap=plt.cm.Accent)
	

# This prints the image along with actual and predicted class labels
def plot_image_prediction(x_test,y_test, labels_list,limit=10):
    """
    This function is used for visualising the pridicts made by the model.
    This prints the image along with what class it actually belongs to and what the model has predicted the image class to be
    """
    fig=plt.figure(figsize=(12, 12))
    columns = 2
    rows = 5 #math.ceil(len(y_test)/2)
    print(f'rows: {rows}     columns: {columns}')
    iterations = columns*rows
    for i in range(1, iterations+1):
        fig.add_subplot(rows, columns, i)
        rand_n = np.random.randint(x_test.shape[0])
        plt.imshow(x_test[rand_n][:, :, 0], cmap='gray')
        plt.title(f'Actual:{labels_list[np.argmax(y_test[rand_n])]} Predicted: {labels_list[np.argmax(my_model.predict(x_test[rand_n].reshape(-1, 50, 50, 3)))]}');
        plt.axis('off')
    fig.savefig(os.path.join(PLOTS_PATH,'Prediction.jpg'))
    #print(my_model.evaluate(x_test,y_test))

# This is used to print the heat maps for the model predictions on the test data
def plot_confusion_map(cnf_matrix, label_classes, normalize, title, file_name, cmap=plt.cm.Blues):
    """
    This funcition takes confusion matrix of the model as the input and outputs the heat map between the actual and predicted classes for the test image from the dataset.
    This would give a clear picture of what model had precicted correctly and what the model has predicted wrong.
    """
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 12))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(label_classes))
    plt.xticks(tick_marks, label_classes, rotation=90)
    plt.yticks(tick_marks, label_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(PLOTS_PATH,file_name))
    plt.show()

# Creates the folder that are requires for the program to run properly
def create_prerequisite_folders():
    """
    This function checks and creates the folder that are need for the program to run noramlly by creating these folder in the working directory
    """
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    if not os.path.exists(NEW_IMAGES_PATH):
        os.mkdir(NEW_IMAGES_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.mkdir(PLOTS_PATH)
    if not os.path.exists(MODEL_LABELS_PATH):
        os.mkdir(MODEL_LABELS_PATH)  


######################################## Main Function Starts Here #############################################

arg_1 = sys.argv[1]
arg_2 = sys.argv[2]
arg_3 = sys.argv[3]

#Global Variables
class_labels = []
images = []
new_images = []
new_class_labels = []
combined_images = []
combined_labels = []
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
DATASET_USE = ''

print('*****Program Started*****')

#Creating all the pre-requisite folders
create_prerequisite_folders()

if arg_3 == 'train':
	
	if arg_1 == 'normal':
		#Finding the face from the images and creating the dataset tio train CNN model
		if arg_2 == 'create':
			create_face_dataSet(FOLDER_PATH)
		DATASET_USE = DATASET_PATH

	elif arg_1 =='hog':
		#Finding the face from the images and creating the dataset tio train CNN model
		if arg_2 == 'create':
			create_hog_dataSet(FOLDER_PATH)
		DATASET_USE = HOG_DATASET_PATH

	#Reading the dataset into to variable to convert it into numpy array
	images, class_labels = zip(*load_images(DATASET_USE))


	#sample of what augmentation does to the image
	test_print_augmented_images(images)


	#Printing the number of files in each class
	print_images_counts(class_labels)


	#Since the dataset is not big enough we create more images from same image by applying audmentation to the image
	new_images, new_class_labels = zip(*create_augmented_images(images,class_labels))


	#Adding the newly created images to the actual dataset by appending to the numpy array
	combined_images.extend(images)
	combined_images.extend(new_images)
	combined_labels.extend(class_labels)
	combined_labels.extend(new_class_labels)


	#Printing the number of files in each class after the augmentation
	print_images_counts(combined_labels)


	#Checking if the dataset is balanced by plotting pie chart
	get_data_distribution(combined_labels,'Before Balancing')

	#Balancing the dataset
	mask = balance_dataset(combined_labels)
	combined_images = [combined_images[m] for m in mask]
	combined_labels = [combined_labels[m] for m in mask]

	#Checking if the dataset is balanced by plotting pie chart
	get_data_distribution(combined_labels,'After Balancing')

	#printing the counts of images in each label
	print_images_counts(combined_labels)


	#encoding the labels names to number for the model to train
	categorical_name_vect = encode_label_names(combined_labels)
	labels_list = get_encode_label_names(combined_labels)

	np.save(os.path.join(MODEL_LABELS_PATH,'class_labels.npy'),labels_list)
	#diving the dataset to train and test splits
	x_train, x_test, y_train, y_test = train_test_split(np.array(combined_images, dtype=np.float32),   # input data
                                                    np.array(categorical_name_vect),       # target/output data 
                                                    test_size=0.15,                       # test split percentage
                                                    random_state=42)


	#Creating a cnn model
	input_shape = x_train[0].shape
	model_name = 'Face_Recognition_'+datetime.now().strftime("%H_%M_%S")
	model_subpath = datetime.now().strftime("%m_%d_%Y_%H_%M")
	my_model = create_cnn_model(input_shape,model_name,labels_list)

	#Setting the early stop paramter to stop training the model so as to stop overfitting the model
	early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

	#Training the model on our training dataset with a validation data of 15 percent from training data
	hist = my_model.fit(x_train,y_train, epochs=EPOCHS,batch_size=BATCH_SIZE, shuffle=True, validation_split=0.15,  callbacks=[tensorboard_callback,early_stopping])


	#Plotting the loss of the model
	plot_metrics(hist, x_var='loss',y_var='val_loss',x_label='loss',y_label='val_loss',title='Loss Plot')


	#Plotting the Accuracy of the model
	plot_metrics(hist, x_var='accuracy',y_var='val_accuracy',x_label='accuracy',y_label='val_accuracy',title='Accuracy Plot')


	#Saving the model to drive to directly access the trained model
	my_model.save(os.path.join(MODEL_PATH,model_subpath,model_name+'.h5'))


	#evaluating the model on the test dataset
	#my_model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
	get_model_evaluation(my_model,x_test,y_test)


	#ploting the predictions on the test data
	plot_image_prediction(x_test,y_test,labels_list,limit=10)


	# Compute confusion matrix
	class_prediction = my_model.predict(x_test)
	cnf_matrix = confusion_matrix(y_test.argmax(axis=1), class_prediction.argmax(axis=1))
	np.set_printoptions(precision=2)


	# Plot non-normalized confusion matrix
	plot_confusion_map(cnf_matrix, labels_list, normalize=False, title='Confusion Matrix',file_name='Confusion Matrix.jpg',cmap=plt.cm.Accent)
	
	print('Program Executed Successfully........')

elif arg_3 == 'predict':
	
	#picking the latest model created in the model path
	list_of_files = glob.glob(MODEL_PATH+'*')
	latest_file = max(list_of_files, key=os.path.getctime)
	labels_list = np.load(os.path.join(MODEL_LABELS_PATH,'class_labels.npy'))
	model_file_name = os.listdir(latest_file)[0]
	model_path = os.path.join(latest_file,model_file_name)
	print(f'model_path: {model_path}')
	my_model = load_model(model_path)
	
	#Testing the model with new images from different folder
	new_image_prediction(NEW_IMAGES_PATH,my_model, labels_list)

	print('Program Executed Successfully........')
       
