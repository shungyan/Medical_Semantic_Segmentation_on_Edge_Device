from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda,AveragePooling3D
from keras.optimizers import Adam
from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform' #Try others if you want


################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    #c1 = Dropout(0.1)(c1)
    b1=BatchNormalization()(c1)
    c1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b1)
    b2=BatchNormalization()(c1)
    p1 = AveragePooling3D((2, 2, 2))(b2)
    
    c2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    #c2 = Dropout(0.1)(c2)
    b3=BatchNormalization()(c2)
    c2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b3)
    b4=BatchNormalization()(c2)
    p2 = AveragePooling3D((2, 2, 2))(b4)
     
    c3 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    #c3 = Dropout(0.2)(c3)
    b5=BatchNormalization()(c3)
    c3 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b5)
    b6=BatchNormalization()(c3)
    p3 = AveragePooling3D((2, 2, 2))(b6)
     
    c4 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    #c4 = Dropout(0.2)(c4)
    b7=BatchNormalization()(c4)
    c4 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b7)
    b8=BatchNormalization()(c4)
    p4 = AveragePooling3D(pool_size=(2, 2, 2))(b8)
     
    c5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    #c5 = Dropout(0.3)(c5)
    b9=BatchNormalization()(c5)
    c5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b9)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    #c6 = Dropout(0.2)(c6)
    b10=BatchNormalization()(c6)
    c6 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b10)
    b11=BatchNormalization()(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(b11)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    #c7 = Dropout(0.2)(c7)
    b12=BatchNormalization()(c7)
    c7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b12)
    b13=BatchNormalization()(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(b13)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    #c8 = Dropout(0.1)(c8)
    b14=BatchNormalization()(c1)
    c8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b14)
    b15=BatchNormalization()(c1)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(b15)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    #c9 = Dropout(0.1)(c9)
    b16=BatchNormalization()(c1)
    c9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(b16)
    b17=BatchNormalization()(c1)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(b17)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model

#Test if everything is working ok. 
model = simple_unet_model(128, 128, 128, 3, 4)
print(model.input_shape)
print(model.output_shape)