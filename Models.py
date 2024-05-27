from keras.layers.wrappers import TimeDistributed
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Add,Input, Lambda,Activation,LSTM,Dropout
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers.merge import concatenate
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization

def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)
    result = K.maximum(sum_square, K.epsilon())
    return result

def l1_dist(vect):
    x, y = vect
    return K.abs(x-y)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def slow_fusion_4depth(input_shape,output_shape):
    first_cnn_models = []
    split_size = 4

    split_len = int(input_shape[1]/split_size)
    print("Split Len for SF:" + str(split_len))

    input_layer = Input(shape =( input_shape[0],input_shape[1],input_shape[2] ) )

    # Initialising the first CNN
    for i in range(split_size):
        # input_layer = Input(shape = (64, 64, 2))
        split_input_layer = Lambda(lambda x: x[:,:,i*split_len:(i+1)*split_len,:])(input_layer)
        cnn_layer_1 = Conv2D(64, (3, 3), input_shape = ( input_shape[0], split_len, 2), activation = 'relu')(split_input_layer)
        mp_layer_1 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_1)
        # Adding a second convolutional layer
        cnn_layer_2 = Conv2D(32, (3, 3), activation = 'relu')(mp_layer_1)
        mp_layer_2 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_2)

        # Adding a third convolutional layer
        cnn_layer_3 = Conv2D(16, (3, 3), activation = 'relu')(mp_layer_2)
        mp_layer_3 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_3)

        # Adding a third convolutional layer
        cnn_layer_4 = Conv2D(16, (3, 3), activation = 'relu')(mp_layer_3)
        mp_layer_4 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_4)

        # create the model
        # cnn_model = Model(inputs = split_input_layer, outputs = mp_layer_2)
        # add to the list
        first_cnn_models.append(mp_layer_4)



    #Second CNN 
    second_cnn_model_1_inputs = [first_cnn_models[0], first_cnn_models[1]]
    second_cnn_model_2_inputs = [first_cnn_models[2], first_cnn_models[3]]

    second_cnn_model_1_added_layer = Add()(second_cnn_model_1_inputs)
    second_cnn_1_layer = Conv2D(16, (3, 3), activation = 'relu')(second_cnn_model_1_added_layer)
    second_mp_1_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_1_layer)

    second_cnn_model_2_added_layer = Add()(second_cnn_model_2_inputs)
    second_cnn_2_layer = Conv2D(16, (3, 3), activation = 'relu')(second_cnn_model_2_added_layer)
    second_mp_2_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_2_layer)

    # flat_layer_1 = Flatten()(second_mp_1_layer)
    # flat_layer_2 = Flatten()(second_mp_2_layer)

    # print( flat_layer_1.shape )

    # flat_layer = [flat_layer_1, flat_layer_2]
    
    #lstm = LSTM(128, return_sequences=True, return_state=True)(flat_layer)
    #hidden_dense_layer = lstm( flat_layer)

    # Third CNN
    third_cnn_model_inputs = Add()([second_mp_1_layer,second_mp_2_layer])
    third_cnn_layer = Conv2D(16, (3,3),activation='relu')(third_cnn_model_inputs)
    third_mp_layer = MaxPooling2D(pool_size = (2, 2))(third_cnn_layer)

    Flatten
    flat_layer = Flatten()(third_mp_layer)

    # Fully Connected Feed Forward Neural Nnetwork
    first_dense_layer = Dense(128)(flat_layer)
    hidden_dense_layer = Dense(64)(first_dense_layer)
    output_layer = Dense(output_shape, activation = 'softmax') (hidden_dense_layer)

    # Create the model
    base_model = Model(inputs = input_layer, outputs = output_layer)
    # base_model = Model(inputs = [temp_model.input for temp_model in first_cnn_models], outputs = output_layer)


    #plot_model(base_model, show_shapes=True,to_file='basse_network_model.png')

    return base_model

def slow_fusion_4depthWND(input_shape,output_shape):
    first_cnn_models = []
    split_size = 4

    split_len = int(input_shape[1]/split_size)
    print("Split Len for SF:" + str(split_len))

    input_layer = Input(shape =( input_shape[0],input_shape[1],input_shape[2] ) )

    # Initialising the first CNN
    for i in range(split_size):
        # input_layer = Input(shape = (64, 64, 2))
        split_input_layer = Lambda(lambda x: x[:,:,i*split_len:(i+1)*split_len,:])(input_layer)
        cnn_layer_1 = Conv2D(64, (3, 3), input_shape = ( input_shape[0], split_len, 2), activation = 'relu')(split_input_layer)
        norm_layer_1 = BatchNormalization()(cnn_layer_1)
        mp_layer_1 = MaxPooling2D(pool_size = (2, 2))(norm_layer_1)
        drop_layer_1 = Dropout(0.2)(mp_layer_1)

        # Adding a second convolutional layer
        cnn_layer_2 = Conv2D(32, (3, 3), activation = 'relu')(drop_layer_1)
        norm_layer_2 = BatchNormalization()(cnn_layer_2)
        mp_layer_2 = MaxPooling2D(pool_size = (2, 2))(norm_layer_2)
        drop_layer_2 = Dropout(0.4)(mp_layer_2)


        # Adding a third convolutional layer
        cnn_layer_3 = Conv2D(16, (3, 3), activation = 'relu')(drop_layer_2)
        norm_layer_3 = BatchNormalization()(cnn_layer_3)
        mp_layer_3 = MaxPooling2D(pool_size = (2, 2))(norm_layer_3)
        drop_layer_3 = Dropout(0.4)(mp_layer_3)


        # Adding a third convolutional layer
        cnn_layer_4 = Conv2D(16, (3, 3), activation = 'relu')(drop_layer_3)
        norm_layer_4 = BatchNormalization()(cnn_layer_4)
        mp_layer_4 = MaxPooling2D(pool_size = (2, 2))(norm_layer_4)

        # create the model
        # cnn_model = Model(inputs = split_input_layer, outputs = mp_layer_2)
        # add to the list
        first_cnn_models.append(mp_layer_4)



    #Second CNN 
    second_cnn_model_1_inputs = [first_cnn_models[0], first_cnn_models[1]]
    second_cnn_model_2_inputs = [first_cnn_models[2], first_cnn_models[3]]

    second_cnn_model_1_added_layer = Add()(second_cnn_model_1_inputs)
    second_cnn_1_layer = Conv2D(16, (3, 3), activation = 'relu')(second_cnn_model_1_added_layer)
    second_mp_1_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_1_layer)

    second_cnn_model_2_added_layer = Add()(second_cnn_model_2_inputs)
    second_cnn_2_layer = Conv2D(16, (3, 3), activation = 'relu')(second_cnn_model_2_added_layer)
    second_mp_2_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_2_layer)

    # flat_layer_1 = Flatten()(second_mp_1_layer)
    # flat_layer_2 = Flatten()(second_mp_2_layer)

    # print( flat_layer_1.shape )

    # flat_layer = [flat_layer_1, flat_layer_2]
    
    #lstm = LSTM(128, return_sequences=True, return_state=True)(flat_layer)
    #hidden_dense_layer = lstm( flat_layer)

    # Third CNN
    third_cnn_model_inputs = Add()([second_mp_1_layer,second_mp_2_layer])
    third_cnn_layer = Conv2D(16, (3,3),activation='relu')(third_cnn_model_inputs)
    third_mp_layer = MaxPooling2D(pool_size = (2, 2))(third_cnn_layer)

    Flatten
    flat_layer = Flatten()(third_mp_layer)

    # Fully Connected Feed Forward Neural Nnetwork
    first_dense_layer = Dense(128)(flat_layer)
    hidden_dense_layer = Dense(64)(first_dense_layer)
    output_layer = Dense(output_shape, activation = 'softmax') (hidden_dense_layer)

    # Create the model
    base_model = Model(inputs = input_layer, outputs = output_layer)
    # base_model = Model(inputs = [temp_model.input for temp_model in first_cnn_models], outputs = output_layer)


    #plot_model(base_model, show_shapes=True,to_file='basse_network_model.png')

    return base_model


def slow_fusion_rnn(input_shaper, n_class):

    cnn1 = Sequential()
    cnn1.add(Conv2D(32, (3, 3), input_shape=(input_shaper[1], input_shaper[2], 2)))
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Conv2D(32, (3, 3)))
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Conv2D(16, (3, 3)))
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn1.add(Conv2D(16, (3, 3)))
    # cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn1.add(Conv2D(8, (3, 3)))
    # cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Flatten()) # Not sure if this if the proper way to do this.

    cnn2 = Sequential()
    cnn2.add(Conv2D(32, (3, 3), input_shape=(input_shaper[1], input_shaper[2], 2)))
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Conv2D(32, (3, 3)))
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Conv2D(16, (3, 3)))
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn2.add(Conv2D(16, (3, 3)))
    # cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn2.add(Conv2D(8, (3, 3)))
    # cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Flatten()) # Not sure if this if the proper way to do this.



    # model_concat = concatenate([cnn1.output, cnn2.output], axis=-1)

    # model_concat = LSTM(128, return_sequences=False) (model_concat)
    a = input_shaper[1]/2
    a = int(a)
    main_input1 = Input(shape=(input_shaper[0], a, input_shaper[2], input_shaper[3])) # Data has been reshaped to (718, 16, 64, 64, 2)
    main_input2 = Input(shape=(input_shaper[0], a, input_shaper[2], input_shaper[3])) # Data has been reshaped to (718, 16, 64, 64, 2)

    model1 = TimeDistributed(cnn1)(main_input1) # this should make the cnn 'run' 5 times?
    model2 = TimeDistributed(cnn2)(main_input2)

    model_concat = concatenate([model1, model2], axis=-1)


    rnn = Sequential()

    rnn =  LSTM(128, return_sequences=False) 

    dense = Sequential()
    dense.add(Dense(64))
    dense.add(Dense(32))
    dense.add(Dense(n_class, activation = 'softmax')) # Model output  
    
    model = rnn(model_concat) # combine timedistributed cnn with rnn
    model = dense(model) # add dense
    
    final_model = Model(inputs=[main_input1 , main_input2], outputs=model)
    





    return final_model


def slow_fusion_doubleRnn(input_shaper,n_class):

    a = input_shaper[1]/2
    a = int(a)

    cnn1 = Sequential()
    cnn1.add(Conv2D(32, (3, 3), input_shape=(a, input_shaper[2], 2)))
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Conv2D(32, (3, 3)))
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Conv2D(16, (3, 3)))
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn1.add(Conv2D(16, (3, 3)))
    # cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn1.add(Conv2D(8, (3, 3)))
    # cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Flatten()) # Not sure if this if the proper way to do this.

    cnn2 = Sequential()
    cnn2.add(Conv2D(32, (3, 3), input_shape=(a, input_shaper[2], 2)))
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Conv2D(32, (3, 3)))
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Conv2D(16, (3, 3)))
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn2.add(Conv2D(16, (3, 3)))
    # cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn2.add(Conv2D(8, (3, 3)))
    # cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Flatten()) # Not sure if this if the proper way to do this.



    # model_concat = concatenate([cnn1.output, cnn2.output], axis=-1)

    # model_concat = LSTM(128, return_sequences=False) (model_concat)

    main_input1 = Input(shape=(input_shaper[0], a, input_shaper[2], input_shaper[3])) # Data has been reshaped to (718, 16, 64, 64, 2)
    main_input2 = Input(shape=(input_shaper[0], a, input_shaper[2], input_shaper[3])) # Data has been reshaped to (718, 16, 64, 64, 2)

    model1 = TimeDistributed(cnn1)(main_input1) # this should make the cnn 'run' 5 times?
    model2 = TimeDistributed(cnn2)(main_input2)

    #model_concat = concatenate([model1, model2], axis=-1)


    rnn1 = Sequential()
    rnn1 =  LSTM(128, return_sequences=False) 

    rnn2 = Sequential()
    rnn2 =  LSTM(128, return_sequences=False) 

    dense = Sequential()
    dense.add(Dense(64))
    dense.add(Dense(32))
    dense.add(Dense(n_class, activation = 'softmax')) # Model output  
    
    model_1 = rnn1(model1) # combine timedistributed cnn with rnn
    model_2  =rnn2(model2)

    model_concat = concatenate([model_1, model_2], axis=-1)
    model = dense(model_concat) # add dense
    
    final_model = Model(inputs=[main_input1 , main_input2], outputs=model)
    


    return final_model

def slow_fusion_doubleRnn_wBN(input_shaper,n_class):

    a = input_shaper[1]/2
    a = int(a)

    cnn1 = Sequential()
    cnn1.add(Conv2D(32, (3, 3), input_shape=(a, input_shaper[2], 2)))
    cnn1.add(BatchNormalization())
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Conv2D(32, (3, 3)))
    cnn1.add(BatchNormalization())
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Conv2D(16, (3, 3)))
    cnn1.add(BatchNormalization())
    cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn1.add(Conv2D(16, (3, 3)))
    # cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn1.add(Conv2D(8, (3, 3)))
    # cnn1.add(MaxPooling2D(pool_size = (2, 2)))
    cnn1.add(Flatten()) # Not sure if this if the proper way to do this.

    cnn2 = Sequential()
    cnn2.add(Conv2D(32, (3, 3), input_shape=(a, input_shaper[2], 2)))
    cnn2.add(BatchNormalization())
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Conv2D(32, (3, 3)))
    cnn2.add(BatchNormalization())
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Conv2D(16, (3, 3)))
    cnn2.add(BatchNormalization())
    cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn2.add(Conv2D(16, (3, 3)))
    # cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn2.add(Conv2D(8, (3, 3)))
    # cnn2.add(MaxPooling2D(pool_size = (2, 2)))
    cnn2.add(Flatten()) # Not sure if this if the proper way to do this.



    # model_concat = concatenate([cnn1.output, cnn2.output], axis=-1)

    # model_concat = LSTM(128, return_sequences=False) (model_concat)

    main_input1 = Input(shape=(input_shaper[0], a, input_shaper[2], input_shaper[3])) # Data has been reshaped to (718, 16, 64, 64, 2)
    main_input2 = Input(shape=(input_shaper[0], a, input_shaper[2], input_shaper[3])) # Data has been reshaped to (718, 16, 64, 64, 2)

    model1 = TimeDistributed(cnn1)(main_input1) # this should make the cnn 'run' 5 times?
    model2 = TimeDistributed(cnn2)(main_input2)

    #model_concat = concatenate([model1, model2], axis=-1)


    rnn1 = Sequential()
    rnn1 =  LSTM(128, return_sequences=False) 

    rnn2 = Sequential()
    rnn2 =  LSTM(128, return_sequences=False) 

    dense = Sequential()
    dense.add(Dense(64))
    dense.add(Dense(32))
    dense.add(Dense(n_class, activation = 'softmax')) # Model output  
    
    model_1 = rnn1(model1) # combine timedistributed cnn with rnn
    model_2  =rnn2(model2)

    model_concat = concatenate([model_1, model_2], axis=-1)
    model = dense(model_concat) # add dense
    
    final_model = Model(inputs=[main_input1 , main_input2], outputs=model)



    return final_model

def timeDist_CNN_wRNN(input_shaper,n_class):

    # 4 256 256 
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), input_shape=(input_shaper[1], input_shaper[2], 2)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(16, (3, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(16, (3, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(16, (3, 3)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten()) # Not sure if this if the proper way to do this.
    
    # cnn = Sequential()
    # cnn.add(Conv2D(64, (3, 3), input_shape=(input_shaper[1], input_shaper[2], 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(64, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(32, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(16, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(8, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Flatten()) # Not sure if this if the proper way to do this.

    rnn = Sequential()

    rnn =  LSTM(128, return_sequences=False) 

    dense = Sequential()
    dense.add(Dense(64))
    dense.add(Dense(32))
    dense.add(Dense(n_class, activation = 'softmax')) # Model output  
    
    main_input = Input(shape=(input_shaper[0], input_shaper[1], input_shaper[2], input_shaper[3])) # Data has been reshaped to (800, 5, 120, 60, 1)

    model = TimeDistributed(cnn)(main_input) # this should make the cnn 'run' 5 times?
    model = rnn(model) # combine timedistributed cnn with rnn
    model = dense(model) # add dense
    final_model = Model(inputs=main_input, outputs=model)
    
    return final_model

def timeDist_CNN_wRNN_wBN(input_shaper,n_class):

    # 4 256 256 
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), input_shape=(input_shaper[1], input_shaper[2], 2)))
    cnn.add(BatchNormalization() )
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3)))
    cnn.add(BatchNormalization() )
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(16, (3, 3)))
    cnn.add(BatchNormalization() )
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(16, (3, 3)))
    cnn.add(BatchNormalization() )
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(16, (3, 3)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten()) # Not sure if this if the proper way to do this.
    
    # cnn = Sequential()
    # cnn.add(Conv2D(64, (3, 3), input_shape=(input_shaper[1], input_shaper[2], 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(64, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(32, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(16, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Conv2D(8, (2, 2)))
    # cnn.add(MaxPooling2D(pool_size = (2, 2)))
    # cnn.add(Flatten()) # Not sure if this if the proper way to do this.

    rnn = Sequential()

    rnn =  LSTM(128, return_sequences=False) 

    dense = Sequential()
    dense.add(Dense(64))
    dense.add(Dense(32))
    dense.add(Dense(n_class, activation = 'softmax')) # Model output  
    
    main_input = Input(shape=(input_shaper[0], input_shaper[1], input_shaper[2], input_shaper[3])) # Data has been reshaped to (800, 5, 120, 60, 1)

    model = TimeDistributed(cnn)(main_input) # this should make the cnn 'run' 5 times?
    model = rnn(model) # combine timedistributed cnn with rnn
    model = dense(model) # add dense
    final_model = Model(inputs=main_input, outputs=model)
    
    return final_model


def base_network_wo_slowFusion_LSTM():
    first_cnn_models = []
    split_size = 4

    input_layer = Input(shape = (256, 1024, 2))

    # Initialising the first CNN
    for i in range(split_size):
        # input_layer = Input(shape = (64, 64, 2))
        split_input_layer = Lambda(lambda x: x[:,:,i*256:(i+1)*256,:])(input_layer)
        cnn_layer_1 = Conv2D(64, (3, 3), input_shape = (256, 256, 2), activation = 'relu')(split_input_layer)
        mp_layer_1 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_1)
        # Adding a second convolutional layer
        cnn_layer_2 = Conv2D(64, (3, 3), activation = 'relu')(mp_layer_1)
        mp_layer_2 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_2)

        # Adding a third convolutional layer
        cnn_layer_3 = Conv2D(32, (3, 3), activation = 'relu')(mp_layer_2)
        mp_layer_3 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_3)

        # Adding a third convolutional layer
        cnn_layer_4 = Conv2D(32, (3, 3), activation = 'relu')(mp_layer_3)
        mp_layer_4 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_4)

        flat_layer = Flatten()(mp_layer_4)
        # create the model
        # cnn_model = Model(inputs = split_input_layer, outputs = mp_layer_2)
        # add to the list
        first_cnn_models.append(flat_layer)

    



    # #Second CNN 
    # second_cnn_model_1_inputs = [first_cnn_models[0], first_cnn_models[1]]
    # second_cnn_model_2_inputs = [first_cnn_models[2], first_cnn_models[3]]

    # second_cnn_model_1_added_layer = Add()(second_cnn_model_1_inputs)
    # second_cnn_1_layer = Conv2D(16, (3, 3), activation = 'relu')(second_cnn_model_1_added_layer)
    # second_mp_1_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_1_layer)

    # second_cnn_model_2_added_layer = Add()(second_cnn_model_2_inputs)
    # second_cnn_2_layer = Conv2D(16, (3, 3), activation = 'relu')(second_cnn_model_2_added_layer)
    # second_mp_2_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_2_layer)

    # # flat_layer_1 = Flatten()(second_mp_1_layer)
    # # flat_layer_2 = Flatten()(second_mp_2_layer)

    # # print( flat_layer_1.shape )

    # # flat_layer = [flat_layer_1, flat_layer_2]
    
    # #lstm = LSTM(128, return_sequences=True, return_state=True)(flat_layer)
    # #hidden_dense_layer = lstm( flat_layer)

    # # Third CNN
    # third_cnn_model_inputs = Add()([second_mp_1_layer,second_mp_2_layer])
    # third_cnn_layer = Conv2D(16, (3,3),activation='relu')(third_cnn_model_inputs)
    # third_mp_layer = MaxPooling2D(pool_size = (2, 2))(third_cnn_layer)
    # Flatten
    time_1 =  TimeDistributed(first_cnn_models[0])
    time_2 =  TimeDistributed(first_cnn_models[1])
    time_3 =  TimeDistributed(first_cnn_models[2])
    time_4 =  TimeDistributed(first_cnn_models[3])


    semi_input = []
    #flat_layer =first_cnn_models# Flatten()(third_mp_layer)
    semi_input_1 = Add()(time_1)
    lstm_1 =  LSTM(128, return_sequences=True, return_state=True)(semi_input_1)
    semi_input.append(lstm_1)
    semi_input_2 = Add()(time_2)
    lstm_2 =  LSTM(128, return_sequences=True, return_state=True)(semi_input_2)
    semi_input.append(lstm_2)

    semi_input_3 = Add()(time_3)
    lstm_3=  LSTM(128, return_sequences=True, return_state=True)(semi_input_3)
    semi_input.append(lstm_3)

    semi_input_4 = Add()(time_4)
    lstm_4 =  LSTM(128, return_sequences=True, return_state=True)(semi_input_4)
    semi_input.append(lstm_4)

    dummy = Add()(semi_input)
    # Fully Connected Feed Forward Neural Nnetwork
    first_dense_layer = Dense(128)(dummy)
    hidden_dense_layer = Dense(64)(first_dense_layer)
    output_layer = Dense(5, activation = 'sigmoid') (hidden_dense_layer)
    # Create the model

    base_model = Model(inputs = input_layer, outputs = output_layer)
    # base_model = Model(inputs = [temp_model.input for temp_model in first_cnn_models], outputs = output_layer)


    #plot_model(base_model, show_shapes=True,to_file='basse_network_model.png')

    return base_model



def slow_fusion_2dept(input_shape , n_class):
    first_cnn_models = []
    split_size = 4

    input_layer = Input(shape = (input_shape[0], input_shape[1],input_shape[2]))

    # Initialising the first CNN
    for i in range(split_size):
        # input_layer = Input(shape = (64, 64, 2))
        split_input_layer = Lambda(lambda x: x[:,:,i*256:(i+1)*256,:])(input_layer)
        cnn_layer_1 = Conv2D(32, (3, 3), input_shape = (256, 256, 2), activation = 'relu')(split_input_layer)
        mp_layer_1 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_1)
        # Adding a second convolutional layer
        cnn_layer_2 = Conv2D(32, (3, 3), activation = 'relu')(mp_layer_1)
        mp_layer_2 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_2)
        # create the model
        # cnn_model = Model(inputs = split_input_layer, outputs = mp_layer_2)
        # add to the list
        first_cnn_models.append(mp_layer_2)

    # Second CNN 
    second_cnn_model_1_inputs = [first_cnn_models[0], first_cnn_models[1]]
    second_cnn_model_2_inputs = [first_cnn_models[2], first_cnn_models[3]]

    second_cnn_model_1_added_layer = Add()(second_cnn_model_1_inputs)
    second_cnn_1_layer = Conv2D(32, (3, 3), activation = 'relu')(second_cnn_model_1_added_layer)
    second_mp_1_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_1_layer)

    second_cnn_model_2_added_layer = Add()(second_cnn_model_2_inputs)
    second_cnn_2_layer = Conv2D(32, (3, 3), activation = 'relu')(second_cnn_model_2_added_layer)
    second_mp_2_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_2_layer)

    # Third CNN
    third_cnn_model_inputs = Add()([second_mp_1_layer,second_mp_2_layer])
    third_cnn_layer = Conv2D(32, (3,3),activation='relu')(third_cnn_model_inputs)
    third_mp_layer = MaxPooling2D(pool_size = (2, 2))(third_cnn_layer)

    # Flatten
    flat_layer = Flatten()(third_mp_layer)

    # Fully Connected Feed Forward Neural Nnetwork
    first_dense_layer = Dense(128)(flat_layer)
    hidden_dense_layer = Dense(64)(first_dense_layer)
    output_layer = Dense(n_class, activation = 'sigmoid') (hidden_dense_layer)

    # Create the model
    base_model = Model(inputs = input_layer, outputs = output_layer)
    # base_model = Model(inputs = [temp_model.input for temp_model in first_cnn_models], outputs = output_layer)


    #plot_model(base_model, show_shapes=True,to_file='basse_network_model.png')

    return base_model


def base_network_siam():
    first_cnn_models = []
    split_size = 4

    input_layer = Input(shape = (256, 1024, 2))

    # Initialising the first CNN
    for i in range(split_size):
        # input_layer = Input(shape = (64, 64, 2))
        split_input_layer = Lambda(lambda x: x[:,:,i*256:(i+1)*256,:])(input_layer)
        cnn_layer_1 = Conv2D(32, (3, 3), input_shape = (256, 256, 2), activation = 'relu')(split_input_layer)
        mp_layer_1 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_1)
        # Adding a second convolutional layer
        cnn_layer_2 = Conv2D(32, (3, 3), activation = 'relu')(mp_layer_1)
        mp_layer_2 = MaxPooling2D(pool_size = (2, 2))(cnn_layer_2)
        # create the model
        # cnn_model = Model(inputs = split_input_layer, outputs = mp_layer_2)
        # add to the list
        first_cnn_models.append(mp_layer_2)

    # Second CNN 
    second_cnn_model_1_inputs = [first_cnn_models[0], first_cnn_models[1]]
    second_cnn_model_2_inputs = [first_cnn_models[2], first_cnn_models[3]]

    second_cnn_model_1_added_layer = Add()(second_cnn_model_1_inputs)
    second_cnn_1_layer = Conv2D(32, (3, 3), activation = 'relu')(second_cnn_model_1_added_layer)
    second_mp_1_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_1_layer)

    second_cnn_model_2_added_layer = Add()(second_cnn_model_2_inputs)
    second_cnn_2_layer = Conv2D(64, (3, 3), activation = 'relu')(second_cnn_model_2_added_layer)
    second_mp_2_layer = MaxPooling2D(pool_size = (2, 2))(second_cnn_2_layer)

    # Third CNN
    third_cnn_model_inputs = Add()([second_mp_1_layer,second_mp_2_layer])
    third_cnn_layer = Conv2D(32, (3,3),activation='relu')(third_cnn_model_inputs)
    third_mp_layer = MaxPooling2D(pool_size = (2, 2))(third_cnn_layer)

    # Flatten
    flat_layer = Flatten()(third_mp_layer)

    # Create the model
    base_model = Model(inputs = input_layer, outputs = flat_layer)
    # base_model = Model(inputs = [temp_model.input for temp_model in first_cnn_models], outputs = output_layer)


    plot_model(base_model, show_shapes=True,to_file='basse_network_model.png')

    return base_model


def Siamese_Model():
    input_dim = [64,256,2]
    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

    base = base_network_siam()
    feat_vecs_a = base(img_a)
    feat_vecs_b = base(img_b)

    #distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
    merge_layer = Lambda(euclidean_dist)([feat_vecs_a,feat_vecs_b])
    prediction = Dense(1,activation='sigmoid')(merge_layer)

    siamese_model = Model(input=[img_a, img_b], output=prediction)

    plot_model(siamese_model, show_shapes=True,to_file='siamese_model.png')
    optimizer = SGD(lr = 0.001, momentum = 0.5)
    siamese_model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])

    return siamese_model


