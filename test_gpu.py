import torch

def test_gpu() -> None:
        print("\nis cuda available?", torch.cuda.is_available())
        print("currently", torch.cuda.device_count(), "devices are available")
        print("using device", torch.cuda.current_device())
        print("which is", torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__=="__main__":
        test_gpu()