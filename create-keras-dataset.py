import os
import json
import pickle
import numpy as np
from PIL import Image
from progressbar import ProgressBar

def load_images(base_dir = 'data'):
    arr = []

    image_sets = os.listdir(base_dir + '/images')
    for image_set in image_sets:
        with open(base_dir + '/_annotations/' + image_set + '.json') as json_file:
            print('Loading dataset "' + image_set + '"')

            data = json.load(json_file)

            bar = ProgressBar(maxval=len(data['dots'])).start()
            i = 0

            for image in data['dots']:
                if image['xy'] is not None:
                    image = Image.open(base_dir + '/images/' + image_set + '/' + image['imName'] + '.JPG')

                    width, height = image.size
                    new_dimensions = min((width, height)) - 80

                    left = (width - new_dimensions) / 2
                    top = (height - new_dimensions) / 2
                    right = (width + new_dimensions) / 2
                    bottom = (height + new_dimensions) / 2

                    image = image.crop((left, top, right, bottom))
                    image = image.resize((256, 256), Image.LANCZOS)

                    arr.append(np.array(image, dtype=np.uint8))
                
                i += 1
                bar.update(i)	
            
            print()

    X = np.array(arr)
    X = X.astype('float32')

    X = (X - 127.5) / 127.5
    return X

training_data = load_images()

pickle_out = open('data.pickle', 'wb')
pickle.dump(training_data, pickle_out)
pickle_out.close()
