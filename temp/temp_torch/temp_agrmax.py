import torch

if __name__ == '__main__':
    v_raw_predict = torch.randn(32, 10)    
    v_list_predict = v_raw_predict.argmax(dim=1)
    print(v_list_predict.shape)