import onnx
import onnxruntime

import onnx_inference
from onnx import helper
import numpy as np
import onnx_graphsurgeon as gs

def main():


    # load model
    model = onnx.load("lighttrack_neck_head.onnx")
    # create node
    graph = gs.import_onnx(model)
    tmap = graph.tensors()
    graph.outputs = [tmap['448'].to_variable(dtype=np.float32), tmap['454'].to_variable(dtype=np.float32)]

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "test1.onnx")

    input2 = np.random.randn(1, 96, 18, 18).astype(np.float32)
    input1 = np.random.randn(1, 96, 8, 8).astype(np.float32)
    session = onnxruntime.InferenceSession('test1.onnx')
    output = session.run([], input_feed={'input1': input1, 'input2': input2})
    print(output)




if __name__ == "__main__":
    main()

