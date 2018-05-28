class Env(object):
    ratio = 1 # 訓練集比例
    batch_size = 50 # batch size
    shuffle = True # 打亂data
    core = 1 # 使用核心數
    epochs = 50 # epoch
    use_gpu = False # 是否使用GPU
    g_emb_dim = 32 # generator embedding dimension
    g_hid_dim = 32 # generator hidden dimension