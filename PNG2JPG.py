import cv2 as cv
import os

path = 'C://Users//User//Desktop//drone//Training_dataset//label_img'
newpath = 'C://Users//User//Desktop//drone//Training_dataset//label_img_converted'

if not os.path.exists(newpath):
    os.mkdir(newpath)
print(newpath)

path_list = os.listdir(path)
path_list.sort()
for filename in path_list:
    portion = os.path.splitext(filename)
    print('convert  ' + filename + '  to ' + portion[0] + '.jpg')
    src = cv.imread(os.path.join(path, filename))
    cv.imwrite(os.path.join(newpath, portion[0] + '.jpg'), src)

print('转换完毕，文件存入 ' + newpath + ' 中')
