import numpy as np
import tensorrt as trt
import time
import argparse
from data_processing import PreprocessYOLO, PostprocessYOLO
import sys,os
import pycuda.driver as cuda
import pycuda.autoinit
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from PIL import ImageDraw
TRT_LOGGER = trt.Logger()

class YOLO_TINY:

    def __init__(self, ):
        self.engine_file_path = "model/yolov3-tiny-car-detector.trt"
        self.input_resolution_yolo3_HW = (416,416)

        self.preprocessor = PreprocessYOLO(self.input_resolution_yolo3_HW)

        self.postprocessor = self._load_postprocessor()
        self._load_engine()

    def _load_postprocessor(self):
        postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)],
                          "obj_threshold": 0.5, 
                          "nms_threshold": 0.35,
                          "yolo_input_resolution": self.input_resolution_yolo3_HW}

        return PostprocessYOLO(**postprocessor_args)

    def _load_engine(self):
        engine_file = open(self.engine_file_path, "rb")
        runtime = trt.Runtime(TRT_LOGGER) 
        self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        self.context = self.engine.create_execution_context()
        print('Edge side yolo tiny detector loaded...')

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def _inference(self, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        self.context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def inference(self, image_bytes):
        start = time.time()

        image_raw, image = self.preprocessor.process(image_bytes)
        shape_orig_HW = image_raw.size

        output_shapes = [(1, 18, 13, 13), (1, 18, 26, 26)]

        trt_outputs = []

        inputs, outputs, bindings, stream = self._allocate_buffers()
        
        # Do inference
        # print('Running inference on image {}...'.format(input_image_path))
        # print('Running inference on image')
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = self._inference(bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

        boxes, classes, scores = self.postprocessor.process(trt_outputs, (shape_orig_HW))
        end = time.time()
        # print(end-start)
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        #obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ['car'])
        #output_image_path = './test.png'
        #obj_detected_img.save(output_image_path, 'PNG')
        #print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
        return len(boxes)


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='image/cameras/jackson_test_30_1462.jpg', help='source image file')
    opt = parser.parse_args()
    print(opt)

    model = YOLO_TINY()
    time.sleep(10)
    fps = 10
    for _ in range(fps):
        image_buffer = open(opt.input_image_path, 'rb')
        count = model.inference(image_buffer.read())
        print(count)
        time.sleep(1/fps)

if __name__ == "__main__":
    main()
