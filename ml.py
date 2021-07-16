from google.colab import drive
drive.mount('/content/drive')

from fastai.vision import *
from fastai.metrics import error_rate


bs = 16  #batch size: if your GPU is running out of memory, set a smaller batch size, i.e 16 -----16-----32------64---   RESNET101 ----152LAYER RESNET
sz = 256 #image size ------try 229 too ----256 ------512
PATH = '/content/drive/MyDrive/PROJECT-DATASET'

classes = []
for d in os.listdir(PATH):
    if os.path.isdir(os.path.join(PATH, d)) and not d.startswith('.'):
        classes.append(d) 
print ("There are ", len(classes), "classes:\n", classes)            

data  = ImageDataBunch.from_folder(PATH, ds_tfms=get_transforms(), size=sz, bs=bs, valid_pct=0.2).normalize(imagenet_stats)

print ("There are", len(data.train_ds), "training images and", len(data.valid_ds), "validation images." )

learn = cnn_learner(data, models.resnet50, metrics=accuracy)

learn.lr_find();
learn.recorder.plot()

learn.fit_one_cycle(4, max_lr=slice(1e-3,1e-2))

path = './' #The path of your test image
img = open_image(get_image_files(path)[0])
#pred_class,pred_idx,outputs = learn.predict(img)
prediction = learn.predict(img)
print(prediction[0])
withh = 'WITH_MASK'

if str(prediction[0]) == withh :
  print('This person is wearing a face mask')
else:
  print('This person is not wearing a face mask')

img.show()
