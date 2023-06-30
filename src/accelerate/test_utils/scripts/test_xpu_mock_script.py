import os
import torch
from mock import patch
from accelerate import Accelerator
from accelerate.test_utils import RegressionModel4XPU


def test_mock_prepare_ipex():
    if os.environ.get("ACCELERATE_XPU_MOCK_TEST", "false") == "true":
        with patch("accelerate.Accelerator._prepare_ipex") as p:
            p.return_value = "return"
            accelerator = Accelerator()
            device = accelerator.device
            print("device is {}".format(device))
            model = RegressionModel4XPU()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            model = accelerator.prepare(model, optimizer)
            assert p.call_count == 1, "xpu mock test fail."
            print("xpu mock test is done")


if __name__ == "__main__":
    test_mock_prepare_ipex()
