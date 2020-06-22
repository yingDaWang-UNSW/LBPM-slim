import numpy as np
import cc3d
import copy


def removeDisconnections(image):
    print('Removing disconnected flow partitions')

    Nx, Ny, Nz = image.shape   
    image = np.invert(image)
    CC = cc3d.connected_components(image, connectivity = 6)
    #print(CC)
    N = np.max(CC)
    print('Identification complete: ', N,' partitions identified')
    print('Filtering...') 
    #remove small blobs
    to_keep = list(range(1, N+1))
    for i in range(1, N+1):
        if(np.count_nonzero(CC == i) < Nz):
            to_keep.remove(i)

    #check blob inlet-outlet hydraulic connectivity
    ps = to_keep.copy()
    in_slice = CC[:,:,0]
    out_slice = CC[:,:,-1]
    for i in ps:
        numIn = np.count_nonzero(in_slice == i)
        numOut = np.count_nonzero(out_slice == i)
        if (numIn == 0 or numOut == 0):
            to_keep.remove(i)
            
    print('Removal complete: ', len(to_keep),' connected partitions identified')

    image = np.isin(CC, to_keep, invert = True) 

    return(image)


