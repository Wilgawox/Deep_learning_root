import numpy as np


def compute_apparition_time_from_2d_time_sequence_by_ramp_model(time_sequence):
    # Input  : a np.array( dim_X, dim_Y, dim_T) with float values indicating probability of background (values near -1) or root (values near 1)
    # Output : a np.array( dim_X, dim_Y ) with integer values indicating for each pixel (x,y) the root apparition time from 1 to max_time, or zero if no_root
    N_times=np.shape(time_sequence)[0]

    # The filter_bank is a list of signal models corresponding to apparition of a root, computed for each target time
    filter_bank=np.array([[[[ (j*2-1) if(j<2) else (-1+2*int( i>(j-2)))  for j in range(N_times+1)] ] ] for i in range(N_times)] )  # mat order = [T X Y F]

    # Here is a visual example of the filter in charge of detection of root appearing at the third timepoint
    print("This filter detects root appearing at the third timepoint : "+str(filter_bank[:,0,0,3]))

    # The broadcasted element-wise dotproduct sum(data-mult-bank filter) try all the filters of the bank for each pixel to estimate the likelihood 
    # of a root apparition at each target time. Then we use argmax function to select the index of the filter which gave the highest response
    return np.argmax(np.sum( np.multiply(time_sequence,filter_bank),axis=0),axis=2)




def compute_apparition_time_from_2d_time_sequence_by_mean_shift(time_sequence,alpha,test=False,testx=-1,testy=-1):
    # Input  : a np.array( dim_X, dim_Y, dim_T) with float values indicating probability of background (values near -1) or root (values near 1)
    # Output : a np.array( dim_X, dim_Y ) with integer values indicating for each pixel (x,y) the root apparition time from 1 to max_time, or zero if no_root
    N_times=np.shape(time_sequence)[0]

    # The filter_bank is a list of signal models corresponding to apparition of a root, computed for each target time
    filter_bank=np.array([[[[ (-1.0/(j+1)) if(i<=j) else (1/(N_times-(j+1)))  for j in range(N_times-1)] ] ] for i in range(N_times)] )  # mat order = [T X Y F], here it is F Y X T
    # Here is a visual example of the filter in charge of detection of root appearing at the third timepoint
    #print("This filter compute the mean change at third point  : "+str(filter_bank[:,0,0,3]))

    # The broadcasted element-wise dotproduct sum(data-mult-bank filter) try all the filters of the bank for each pixel to estimate the likelihood 
    # of a root apparition at each target time. Then we use argmax function to select the index of the filter which gave the highest response
    sum=np.sum( np.multiply(time_sequence,filter_bank),axis=0)
    if(test):
        print("At pixel "+str(testx)+","+str(testy))
        print("Data              : "+str(time_sequence[:,testy,testx,0]))
        print("Result of filters : "+str(sum[testy,testx,:]))
    ar=np.argmax(sum,axis=2)
    ma=np.max(sum,axis=2)
    sel=(ma>alpha).astype(int)
    ret=np.multiply(ar,sel)
    return ret

def test_filter_bank_ramp():
    # Test of the function on a simple root growing downwards on the third column of the hypermatrix
    N_times=5
    data=np.array([[[[ 1 if (col==2 and lig<=tim) else -1 ] for col in range(N_times+2)] for lig in range(N_times+3)] for tim in range(N_times)] )
    print("Data provided : "+str(np.shape(data)))
    data2d=compute_apparition_time_from_2d_time_sequence_by_ramp_model(data)
    print("Result should show a root growing to the south")
    print(data2d)

def test_filter_bank_mean_shift():
    # Test of the function on a simple root growing downwards on the third column of the hypermatrix
    N_times=5
    data=np.array([[[[ 1 if (col==2 and lig<=tim) else -1 ] for col in range(N_times+2)] for lig in range(N_times+3)] for tim in range(N_times)] )
    print("Data provided : "+str(np.shape(data)))
    data2d=compute_apparition_time_from_2d_time_sequence_by_mean_shift(data,0.5)
    print("Result should show a root growing to the south")
    print(data2d)



