#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# Pat a Mat:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

import numpy as np
import tensorflow as tf

import bboxes_utils
from svhn_dataset import SVHN

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")



def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    svhn = SVHN()

    train = svhn.train.map(lambda x: (x["image"], x["classes"], x["bboxes"]))
    #train = svhn.train.map(lambda x: (tf.image.resize(x["image"], [224, 224]),
    #                                  x["classes"], tf.math.round(x["bboxes"]) * (224 / x["image"].shape[0])))
    dev = svhn.dev.map(lambda x: (x["image"], x["classes"], x["bboxes"]))
    test = svhn.test.map(lambda x: (x["image"]))

    #for example in train:
    #    print(example)

    def resize_all2(example):
        scale = 224 / example["image"].shape[0]
        return tf.image.resize(example["image"], [224, 224]), example["classes"], tf.math.round(example["bboxes"] * scale)

    def resize_all(img, cls, box):
        #print(img, cls, box)
        #print(img[2].shape)
        scale = tf.cast(224 / tf.shape(img)[0], dtype=tf.float32)
        return tf.image.resize(img, [224, 224]), cls, tf.math.round(box * scale)

    test_scale_vector = []

    def resize_test(img):
        #scale = tf.cast(224 / tf.shape(img)[0], dtype=tf.float32) #tf.float32
        scale = 224 / tf.shape(img)[0]
        #print(type(224 / int(tf.shape(img)[0])))
        #scale = np.cast(224 / int(tf.shape(img)[0]),)
        test_scale_vector.append(scale)
        return tf.image.resize(img, [224, 224])

    train = train.map(resize_all)
    dev = dev.map(resize_all)
    test = test.map(resize_test)


    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    #input_tensor=tf.keras.Input([224, 224, 3])
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, input_tensor=tf.keras.Input([224, 224, 3]))

    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
             "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    backbone.trainable = False

    # TODO: Create the model and train it

    # 32*i+16,32*j+16

    # TOP: int = 0
    # LEFT: int = 1
    # BOTTOM: int = 2
    # RIGHT: int = 3

    A = 1

    anchors = []
    for i in range(7):
        for j in range(7):
            anchors.append(tf.constant([max(32*i - 16, 0), 32*j, min(32*(i+1) + 16, 223), 32*(j+1)]))

    anchors = tf.stack(anchors)

    def prepare_data(img, cls, box):
        anchor_classes, anchor_bboxes = tf.numpy_function(
            bboxes_utils.bboxes_training, [anchors, cls, box, 0.5], (tf.int64, tf.double)) #(tf.int32, tf.float32)
        anchor_classes = tf.ensure_shape(anchor_classes, [len(anchors)])
        anchor_bboxes = tf.ensure_shape(anchor_bboxes, [len(anchors), 4])

        #keep = tf.where(anchor_classes >= 1)
        #return example["image"], tf.one_hot(anchor_classes[keep], depth=10), anchor_bboxes[keep]

        return img, (tf.one_hot(anchor_classes - 1, depth=10, axis=-1), anchor_bboxes)

    train = train.map(prepare_data)
    dev = dev.map(prepare_data)

    take_count = 13
    train = train.take(take_count)
    dev = dev.take(take_count)
    test = test.take(take_count)

    train = train.batch(args.batch_size)
    dev = dev.batch(args.batch_size)
    test = test.batch(args.batch_size)

    class_head = tf.keras.layers.Conv2D(kernel_size=3, strides=1, padding="same", filters=1280)(backbone.outputs[0])
    class_head = tf.keras.layers.BatchNormalization()(class_head)
    class_head = tf.keras.activations.relu(class_head)
    class_head = tf.keras.layers.Conv2D(kernel_size=3, strides=1, padding="same", filters=10 * A, activation=tf.math.sigmoid)(class_head)
    class_head = tf.keras.layers.Reshape((49, 10*A), input_shape=(7, 7, 10*A), name="class_head")(class_head)
    #class_head = tf.keras.layers.Reshape((49*10 * A,1), input_shape=(7, 7, 10 * A), name="class_head")(class_head)

    box_head = tf.keras.layers.Conv2D(kernel_size=3, strides=1, padding="same", filters=1280)(backbone.outputs[0])
    box_head = tf.keras.layers.BatchNormalization()(box_head)
    box_head = tf.keras.activations.relu(box_head)
    box_head = tf.keras.layers.Conv2D(kernel_size=3, strides=1, padding="same", filters=4 * A)(box_head)
    box_head = tf.keras.layers.Reshape((49, 4 * A), input_shape=(7, 7, 4 * A), name="box_head")(box_head)
    #box_head = tf.keras.layers.Flatten(name="box_head")(box_head)



    def huber_loss(truth, our):
        if truth[0] == 0 and truth[2] == 0:
            return 0
        else:
            res = 0
            for i in range(4):
                res += tf.losses.Huber(truth[i], our[i])

            return res

    def the_loss(truth_cl, truth_reg, our_cl, our_reg):
        cl_loss = tf.losses.BinaryFocalCrossentropy()(truth_cl, our_reg)

        #keep = tf.reduce_all(tf.equal(temp1, temp2))

    def binloss(label, our):
        print("binloss label:",label.shape)
        print("binloss our:",our.shape)
        return tf.losses.BinaryFocalCrossentropy()(label, our)

    def loss_man(label, our):
        print(" label:",label.shape)
        print(" our:",our.shape)
        return tf.losses.Huber()(label, our)

    def metric_man(label, our):
        print("metric label:",label.shape)
        print("metric our:",our.shape)
        return tf.metrics.BinaryAccuracy(name="accuracy")(label, our)

    def weighted_huber(gold, prediction):
        keep = tf.where(gold[..., 0] != gold[..., 2])
        new_gold = tf.gather_nd(gold, keep)
        new_prediction = tf.gather_nd(prediction, keep)
        return tf.losses.Huber()(new_gold, new_prediction)



    model = tf.keras.Model(inputs=backbone.inputs, outputs=[class_head, box_head])

    optimizer = tf.optimizers.experimental.AdamW()
    #optimizer.exclude_from_weight_decay(var_names=['bias'])

    model.compile(
        optimizer=optimizer,
        #loss=[tf.losses.BinaryFocalCrossentropy(), huber_loss],
        #loss={"box_head": huber_loss},
        #loss={"class_head": tf.losses.BinaryFocalCrossentropy(), "box_head": tf.losses.Huber()},
        #loss=[tf.losses.BinaryFocalCrossentropy(), tf.losses.Huber()],
        #loss=[the_loss],
        #loss={"class_head": binloss, "box_head": loss_man},
        #metrics=[svhn.evaluate],
        #metrics=[tf.metrics.binary_accuracy()],
        #metrics={"class_head": tf.metrics.BinaryAccuracy(name="accuracy")},
        #metrics={"class_head": metric_man}
        #metrics=[metric_man]
        loss=[tf.losses.BinaryFocalCrossentropy(), weighted_huber],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

    logs = model.fit(
        train,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback],  # maybe delete
    )

    #dev_pred = model.predict(dev)


    test_pred = model.predict(test)
    test_pred_class = test_pred[0]
    test_pred_box = test_pred[1]
    test_pred_class = np.array(test_pred_class)
    test_pred_box = np.array(test_pred_box)

    #print(test_pred_box[0][0])
    #print(test_pred_class)
    #print(test_pred_class.shape, test_pred_box.shape)

    max_matrix = np.max(test_pred_class, axis=-1)
    argmax_matrix = np.argmax(test_pred_class, axis=-1)

    keep = np.where(max_matrix >= 0.05)

    test_noBG_class = argmax_matrix[keep[0], keep[1]]
    test_score = max_matrix[keep[0], keep[1]]

    test_noBG_box_rel = test_pred_box[keep[0], keep[1]]
    test_noBG_anchors = np.array(anchors)[keep[1]]

    test_noBG_box_abs = bboxes_utils.bboxes_from_fast_rcnn(test_noBG_anchors, test_noBG_box_rel)
    #print(test_noBG_box_abs)
    scales = tf.stack(test_scale_vector[1:])
    scales = scales.numpy()
    scales_for_imgs = scales[keep[0]]

    stacked_scales = np.stack([scales_for_imgs, scales_for_imgs, scales_for_imgs, scales_for_imgs], axis=-1)
    test_bboxes_real_abs = np.round(test_noBG_box_abs / stacked_scales)

    #print(test_bboxes_real_abs)

    out = []

    for i in range(scales.shape[0]):
        for_img = np.where(keep[0] == i)[0]

        out_indices = []

        bboxes_for_img = test_bboxes_real_abs[for_img]
        class_for_img = test_noBG_class[for_img]
        score = test_score[for_img]


        for num in np.unique(class_for_img):
            for_num = np.where(class_for_img == num)[0]
            b_num = bboxes_for_img[for_num]
            #c_num = class_for_img[for_num]
            s_num = score[for_num]

            tf_box_num = tf.convert_to_tensor(b_num)
            #tf_class = tf.convert_to_tensor(c_num)

            sc = tf.cast(tf.convert_to_tensor(s_num), dtype=tf.float32)
            bx = tf.cast(tf_box_num, dtype=tf.float32)

            selected_indices = tf.image.non_max_suppression(
                bx, sc, max_output_size=tf.constant(3))
            selected_boxes = tf.gather(for_num, selected_indices)
            #print(selected_boxes)
            out_indices.append(selected_boxes.numpy())

        if out_indices != []:
            out_indices = np.concatenate(out_indices, axis=-1)
            out_indices.sort()

            final_boxes = bboxes_for_img[out_indices]
            final_classes = class_for_img[out_indices]

            out.append((final_classes, final_boxes))
        else:
            out.append((np.array([0]), np.array([[0, 1, 0, 1]])))

        '''out_indices = np.concatenate(out_indices, axis=-1)
        print("out_ind:", out_indices)



        tf_box = tf.convert_to_tensor(bboxes_for_img)
        tf_class = tf.convert_to_tensor(class_for_img)

        sc = tf.cast(tf.convert_to_tensor(score), dtype=tf.float32)
        bx = tf.cast(tf_box, dtype=tf.float32)

        print(class_for_img)


        selected_indices = tf.image.non_max_suppression(
            bx, sc, max_output_size=tf.constant(5))
        selected_boxes = tf.gather(tf_box, selected_indices)
        selected_classes = tf.gather(tf_class, selected_indices)

        final_boxes = selected_boxes.numpy()
        final_classes = selected_classes.numpy()

        #print(final_boxes.shape, final_classes.shape)

        out.append((final_classes, final_boxes))'''



    '''print("keep2", keep2.shape)
    print(keep2)
    aaaaa = np.take_along_axis(test_pred_class, keep2, axis=1)
    print(aaaaa.shape)
    test_noback_class = np.argmax(test_pred_class[keep2,:], axis=-1)
    print(test_noback_class)'''

    #print("test_pred class size:", len(test_pred[0]), len(test_pred[0][0]), len(test_pred[0][0][0]))
    #print("test_pred boxes size:", len(test_pred[1]), len(test_pred[1][0]), len(test_pred[1][0][0]))
    #print(test_pred[1][0][0])
    #test_pred_class = []
    #test_pred_boxes = []
    #for i in range(len(test_pred[0])):
    #    test_pred_class.append([])
    #    for j in range(len(test_pred[0][0])):
    #        test_pred_class[i][j] = np.argmax(test_pred[0][i][j])


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;

        for predicted_classes, predicted_bboxes in out:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
