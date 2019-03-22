# 3rd STEP --> 102 Flowers
# Create one subfolder for each class
# Olga Valls
#
import pandas as pd
import csv
import os
import shutil

# labels from training an test sets to lists
train_file = 'labels_train.csv'
test_file = 'labels_test.csv'
train_df = pd.read_csv(train_file, header=None, quoting=csv.QUOTE_NONE)
test_df = pd.read_csv(test_file, header=None, quoting=csv.QUOTE_NONE)
tr = train_df.values.tolist()
te = test_df.values.tolist()
# list of list of numbers to list of numbers
y_tr = [item for sublist in tr for item in sublist]
y_te = [item for sublist in te for item in sublist]
print('Labels Train: {}'.format(y_te))
print('Labels Test: {}'.format(y_te))

# create one subfolder for each class in train and test folders
# use class_ before name of the subfolder:
# https://medium.com/difference-engine-ai/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2
for i in range(102):    # 102 classes: 0 to 101
    os.makedirs(os.path.join('train_images', 'class_'+str(i).rjust(3,'0')))
    #os.makedirs(os.path.join('test_images', str(i)))
    os.makedirs(os.path.join('test_images', 'class_'+str(i).rjust(3,'0')))


# move images in train folder to each corresponding class-subfolder
#source = glob.glob('test_images/*.jpg')
# glob.glob llista aleatòriament, desordenat.
# M'interessa tenir-ho ordenat pq. cada jpg va amb el seu label
# ['.ds_store', 'test_00001.jpg', 'test_00002.jpg', ...
list_train = sorted([f.lower() for f in os.listdir('train_images')])   # Convert to lower case
list_test = sorted([f.lower() for f in os.listdir('test_images')])   # Convert to lower case
#eliminar el primer element de la llista pq. és '.ds_store'
#del(list_train[0])

# Eliminar .ds_store i els noms dels 102 subfolders de la llista d'arxius
# només em quedo amb els noms dels jpgs
#target_names = ['.ds_store','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101']
target_names = ['.ds_store','class_000','class_001','class_002','class_003','class_004','class_005','class_006','class_007','class_008','class_009','class_010','class_011','class_012','class_013','class_014','class_015','class_016','class_017','class_018','class_019','class_020','class_021','class_022','class_023','class_024','class_025','class_026','class_027','class_028','class_029','class_030','class_031','class_032','class_033','class_034','class_035','class_036','class_037','class_038','class_039','class_040','class_041','class_042','class_043','class_044','class_045','class_046','class_047','class_048','class_049','class_050','class_051','class_052','class_053','class_054','class_055','class_056','class_057','class_058','class_059','class_060','class_061','class_062','class_063','class_064','class_065','class_066','class_067','class_068','class_069','class_070','class_071','class_072','class_073','class_074','class_075','class_076','class_077','class_078','class_079','class_080','class_081','class_082','class_083','class_084','class_085','class_086','class_087','class_088','class_089','class_090','class_091','class_092','class_093','class_094','class_095','class_096','class_097','class_098','class_099','class_100','class_101']

list_train = [e for e in list_train if e not in target_names]
list_test = [e for e in list_test if e not in target_names]

#eliminar el primer element de la llista pq. és '.ds_store'
#del(list_test[0])
print('list_train: {}'.format(list_train))
print('Length list_train: {}'.format(len(list_train)))
print('list_test: {}'.format(list_test))
print('Length list_test: {}'.format(len(list_test)))


# move train files to class destination folder
for i, filename in enumerate(list_train):
    #num_class = str(y_tr[i])    # de la llista de classes del labels_train
    num_class = 'class_'+str(y_tr[i]).rjust(3,'0')
    destination = os.path.join('train_images', num_class)
    print('Train: i / filename / class / destination: {} / {} / {} / {}'.format(i, filename, num_class, destination))
    shutil.move(os.path.join('train_images', filename),destination)

# move test files to class destination folder
for i, filename in enumerate(list_test):
    num_class = 'class_'+str(y_te[i]).rjust(3,'0')    # de la llista de classes del labels_test
    destination = os.path.join('test_images', num_class)
    print('Test: i / filename / class / destination: {} / {} / {} / {}'.format(i, filename, num_class, destination))
    shutil.move(os.path.join('test_images', filename),destination)

