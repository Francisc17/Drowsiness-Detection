
# Motivation

Feeling abnormally sleepy or tired during the day is commonly known as drowsiness. Drowsiness may lead to additional symptoms, such as forgetfulness or falling asleep at inappropriate times. A variety of things may cause drowsiness, like the following ones \[1] : 
1. Lifestyle factors
2. Mental state
3. Medical conditions
4. Medications
5. Sleeping disorder

Utilizing artificial intelligence, we have the capability to identify signs of fatigue, enabling us to alert individuals and, where necessary, intervene to mitigate the risk of engaging in tasks demanding high levels of concentration and attention. One area that has attracted a lot of attention in research is using systems to spot when drivers are tired [2-4].

The focus of this work is to apply the YOLO (You Only Look Once) network to detect drowsiness in real-time. I opted for YOLO due to its simplicity, excellent speed, and accuracy, along with its ease of installation.

This work and respective implementation was strongly influenced by: [https://www.youtube.com/watch?v=tFNJGim3FXw](https://www.youtube.com/watch?v=tFNJGim3FXw)

---
# YOLO (You Only Look Once)

YOLOv5 was used because i already had some knowledge using it but in practice YOLOv8 should be a better option and could potentially improve the results obtained. Here we can see the comparison between the two \[5] :

![Drowsiness Detection   1](https://i.imgur.com/FKpjV62.png)


The YOLO (You Only Look Once) model suggests utilizing an end-to-end neural network that predicts bounding boxes and class probabilities simultaneously. This approach diverges from previous object detection methods, which adapted classifiers for detection purposes. 

Following a fundamentally different approach to object detection, YOLO achieved state-of-the-art results, beating other real-time object detection algorithms by a large margin.

## YOLO Architecture

The YOLO algorithm takes an image as input and then uses a simple deep convolutional neural network to detect objects in the image. The architecture of the CNN model that forms the backbone of YOLO is shown below.

![Drowsiness Detection   2 1](https://i.imgur.com/EUMfBDW.png)

YOLO uses an initial pre-training phase on the first 20 convolution layers with ImageNet, then the pre-trained model is converting for detection, since previous research demonstrates that adding convolution and connected layers to a pre-trained network improves performance.

The model divides input images into an S × S grid, assigning grid cells responsibility for detecting objects whose centers fall within them. Each cell predicts B bounding boxes and confidence scores for those boxes, reflecting both object presence and prediction accuracy. During training, YOLO ensures each object is predicted by only one bounding box predictor, promoting specialization among predictors and enhancing recall.

Non-maximum suppression (NMS) is employed post-processing to refine detection accuracy by eliminating redundant or inaccurate bounding boxes, ensuring each object is represented by a single bounding box.

## YOLOv5

YOLO v5 was introduced in 2020 by the same team that developed the original YOLO algorithm as an open-source project and is maintained by Ultralytics. YOLO v5 builds upon the success of previous versions and adds several new features and improvements. Unlike YOLO, YOLO v5 uses a more complex architecture called EfficientDet (architecture shown below), based on the EfficientNet network architecture. Using a more complex architecture in YOLO v5 allows it to achieve higher accuracy and better generalization to a wider range of object categories.

![Drowsiness Detection   3](https://i.imgur.com/N7W2k1p.png)


YOLO v5 contrasts with its predecessor in training data and anchor box generation. YOLO trained on the 20-category PASCAL VOC dataset, while YOLO v5 employs the more expansive D5 dataset with 600 categories. YOLO v5 introduces "dynamic anchor boxes," generated via clustering ground truth boxes for better alignment with detected objects. Additionally, it implements "spatial pyramid pooling" (SPP) to enhance small object detection, refining the SPP architecture for superior performance compared to YOLO v4.

The YOLOv5 series consists of four variants: YOLOv5s, YOLOv5m, YOLOv5l, and YOLOv5x, each offering different trade-offs in terms of speed and accuracy \[5] :
- **YOLOv5s** is the smallest and fastest version, suitable for real-time applications with slightly lower accuracy.
- **YOLOv5m** provides a balance between speed and accuracy, making it a versatile choice for various applications.
- **YOLOv5l** offers improved accuracy at the expense of speed, making it suitable for tasks demanding higher precision. (**Used on this work**)
- **YOLOv5x** is the largest and most accurate variant, suitable for applications where high accuracy is paramount, although it presents slower inference times. 

![Drowsiness Detection   4](https://i.imgur.com/tqouOGh.png)


After YOLOv5, we already had v6, v7 and v8 but that will be explained in more depth on future work.

**To learn more about YOLO and its architecture, I recommend the following references: \[6-9]**

---
# Dataset

YOLOv5 pre-trained on the COCO dataset was used. This dataset has 80 different classes which can be consulted [here](https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt).

The custom and smaller dataset that we use to train YOLO was made by me, by manually collecting images for each class (Drowsiness and awake) and labelling them.

Collect images for each class:
```Python
cap = cv2.VideoCapture(0) # 0 indicates the standard system associated cam

for label in labels:

    print('Collecting images for {}'.format(label))

    # Time to get ready for taking x images for each label
    time.sleep(10)

    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        ret, frame = cap.read()
        
        # using uuid to create a unique image file name
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')

        # Save image
        cv2.imwrite(imgname, frame)

        # Show image taken
        cv2.imshow('Image Collection', frame)

        # Wait for 3 seconds before taking the next picture
        time.sleep(3)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

The manual labelling was made using the [LabelImg package](https://pypi.org/project/labelImg/) that is a graphical image annotation tool and label object bounding boxes in images.

![Drowsiness Detection   6](https://i.imgur.com/urAIa6S.png)

**At the end of this process, the dataset consists of 424 images, 212 from the Drowsiness class and 212 from the awake class.**

---
# Training and Inference

To train a custom model, I used the technique known as transfer learning. This consists of using knowledge previous obtained from training in COCO dataset and use that to help in the new and different task. In practice, this is done by using the weights of the pre-trained network and re-training it on the new data for the new objective.

![Drowsiness Detection   5](https://i.imgur.com/cIysLSy.png)

The YOLOv5l was used here with the custom dataset:

```Python
# The cloning yolov5 project already has a train.py file to be used for train
# --img parameter defines the picture size (320x320 in this case)
# --batch defines the batch size, determining how many images are preocessed in each train iteration
# --epochs defines the number of epochs that the model will be trained
# --data defines the data where models will be trained (data.yaml points for the actual data)
# --weights are the network weights. Here we use transfer learning from the large yolov5 version
# --workers pecifies the number of CPU workers used for data loading during training.

!cd yolov5 && python train.py --img 320 --batch 16 --epochs 200 --data data.yaml --weights yolov5l.pt --workers 6
```

After training, we first need to load the custom model:

```Python
# load model with custom weights (the previous trained ones)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp31//weights/last.pt', force_reload=True)
```

And after that we can pass some images to test it:

```Python
# Two diferent pictures in order to test the model
img_test1 = "data/test/test1.jpg"
img_test2 = "data/test/test2.jpg"
```

```Python
# test model on awake

results = model(img_test1)

%matplotlib inline
plt.imshow(np.squeeze(results.render()))
plt.show()
```

![Drowsiness Detection   awake test](https://i.imgur.com/CRJWofC.png)

```Python
# Test model on drowsy
results = model(img_test2)

%matplotlib inline
plt.imshow(np.squeeze(results.render()))

plt.show()
```

![Drowsiness Detection   drowsiness](https://i.imgur.com/mBr5Z7i.png)


We can, of course, use the model to predict in real-time:

```Python
# Same real time detection used before but with our trained yolo model

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---
# Conclusion

This was a very interesting piece of work to do in order to better understand how the YOLO network works, how images are normally collected and labelled for Computing Vision tasks. Although it's a simple piece of work, it's something that can serve as a basis for building a more robust system with great practical applicability.
### Strengths
- Good performance, with the system being able to identify most cases correctly.
- Use of a specific dataset with my own data, allowing me to explore labelling tools.
- Good work to get a better understanding of the YOLO network and how to adapt and apply it to a specific problem.

### Limitations
- Training data from just one person with photos taken in a wide variety of environments. This can cause a bias in the model and it may only be able to predict when it's me and certain background conditions.
- Perhaps YOLOv8, for example, is capable of delivering better results, but it hasn't been considered at the moment.

### Future work
- Creating a larger dataset, with a greater diversity of people, environments, different tones, lighting and in general trying to ensure that the model is able to maintain performance regardless of the person and environment it is in.
- Don't just tick a general bounding box in the face, but show explicitly why it is arriving at the result. For example, show that it's because of the eyes that are closed or because of the position of the mouth.

---
# References:

\[1]: https://www.healthline.com/health/drowsiness
\[2]: https://ieeexplore.ieee.org/document/9596413
\[3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9482962/
\[4]: https://www.mdpi.com/2313-433X/9/5/91
\[5]: https://www.augmentedstartups.com/blog/yolov8-vs-yolov5-choosing-the-best-object-detection-model
\[6]: https://www.v7labs.com/blog/yolo-object-detection
\[7]: https://arxiv.org/abs/1506.02640
\[8]: https://dataphoenix.info/a-guide-to-the-yolo-family-of-computer-vision-models/
\[9]: https://blog.roboflow.com/guide-to-yolo-models/

