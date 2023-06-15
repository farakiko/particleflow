import pickle
import sys
import time

import numpy as np
import onnxruntime
import pynvml
import yaml

# pip install only onnxruntime_gpu, not onnxruntime!

if __name__ == "__main__":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    EP_list = ["CUDAExecutionProvider"]
    #     EP_list = ["CPUExecutionProvider"]

    time.sleep(5)

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_initial = mem.used / 1000 / 1000
    print("mem_initial", mem_initial)

    onnx_sess = onnxruntime.InferenceSession(f"{sys.argv[1]}/model.onnx", providers=EP_list)
    time.sleep(5)

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_onnx = mem.used / 1000 / 1000
    print("mem_onnx", mem_initial)

    input_dim = 17
    # get bin size from config
    with open(f"{sys.argv[1]}/config.yaml", "r") as stream:
        bin_size = yaml.safe_load(stream)["parameters"]["combined_graph_layer"]["bin_size"]

    out = {}

    batch_size = int(sys.argv[2])
    # for num_elems in [bin_size * i for i in range(50)][1:]:  # skip the first element which is 0
    for num_elems in [100, 200, 300, 400, 500]:
        out[num_elems] = {}
        times = []
        mem_used = []

        # average over 100 events
        for i in range(100):
            # allocate array in system RAM
            X = np.array(np.random.randn(batch_size, num_elems, input_dim), np.float32)
            X1 = np.array(np.random.randn(batch_size, 1, 16), np.float32)
            X2 = np.array(np.random.randn(batch_size, 1, 16), np.float32)

            # transfer data to GPU, run model, transfer data back
            t0 = time.time()
            pred_onx = onnx_sess.run(
                None,
                {
                    "x:0": X,
                    "pf_net_dense/normalization/sub/y:0": X1,
                    "pf_net_dense/normalization/Sqrt/x:0": X2,
                },
            )
            t1 = time.time()
            dt = t1 - t0
            times.append(dt)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used.append(mem.used / 1000 / 1000)

        print(
            "Nelem={} mean_time={:.2f} ms stddev_time={:.2f} ms mem_used={:.0f} MB".format(
                num_elems,
                1000.0 * np.mean(times),
                1000.0 * np.std(times),
                np.max(mem_used),
            )
        )
        time.sleep(5)

        out[num_elems]["mean"] = 1000.0 * np.mean(times)
        out[num_elems]["std"] = 1000.0 * np.std(times)
        out[num_elems]["mem_used"] = np.max(mem_used)

        with open(f"out_bs{batch_size}.pkl", "wb") as f:
            pickle.dump(out, f)
