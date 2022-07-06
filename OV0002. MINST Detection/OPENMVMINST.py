#导入功能包
import pyb
import sensor, image, time, math
import os, tf

#摄像头传感器配置
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA) # we run out of memory if the resolution is much bigger...
sensor.set_brightness(800)
sensor.skip_frames(time = 20)
sensor.set_auto_gain(False)  # must turn this off to prevent image washout...
sensor.set_auto_whitebal(True,(0,0x80,0))  # must turn this off to prevent image washout...
clock = time.clock()

#导入模型
net_path = "MINST.tflite"                                  # 定义模型的路径
labels = ["1", "2", "3","4","5", "6", "7", "8"]   # 加载标签
net = tf.load(net_path, load_to_fb=True)                                  # 加载模型


while(True):
    #拍摄一张照片
    img = sensor.snapshot()

    #寻找矩形
    for r in img.find_rects(threshold = 25000):
        #矩形画框
        img.draw_rectangle(r.rect(), color = (255, 0, 0))
        #提取矩形中图像
        img1 = img.copy(r.rect())
        #运行模型识别
        for obj in tf.classify(net , img1, min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
            #计算结果
            sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
            #识别准确率大于90%
            if(sorted_list[0][1]>0.22):
                #在图像中画出结果
                img.draw_string(r.rect()[0] + 20, r.rect()[1]-20, sorted_list[0][0],color = (255,0,0), scale = 2,mono_space=False)
                if(r.rect()[0]<50):
                    print("数字在左边")
                if(r.rect()[0]>=50):
                    print("数字在右边")

