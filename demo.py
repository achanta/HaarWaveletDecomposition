import numpy as np
from timeit import default_timer as timer
from PIL import Image
from haar_wavelets import haar_decomposition



def main():
    ref = Image.open("./bee.png")
    ref = np.asarray(ref)
    print(ref.shape)

    start = timer()
    out = (haar_decomposition(ref[:,:,0])).astype(np.int64)

    # out = 255*(out-out.min())/(out.max()-out.min())
    end = timer()

    out = out.clip(0,255).astype(np.uint8)
    # Image.fromarray(ref[:,:,0]).save("orig.png")
    Image.fromarray(out).save("bee_wavelets.png")
    print("time taken in seconds = ", end-start)

    
if __name__ == "__main__":
    main()
