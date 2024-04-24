from train import train

def main(dataset='xray'):
    modelConfig = None
    if dataset == 'xray':
        modelConfig = {
            "state": "train",  # train or test
            # train_set
            "epoch": 400,
            "batch_size": 4,
            "dataset": 'xray',
            # img
            "img_size": 256,
            "img_channel": 3,
            # report
            "report_channel": 1,
            "report_seq_length": 64,
            "report_embedding_dim": 768,
            # img_latent
            "latent_channel": 4,
            "latent_dim": 16,
            # time_steps
            "num_time_steps": 4,
            # optimizer
            "lr_g": 0.001,
            "lr_d": 0.0003,
            "beta_min": 0.5,
            "beta_max": 0.999,
            "use_geometric": False,
            # codebook
            'num_codebook_vectors': 1024,
            'beta': 0.1,
            # gpt2
            # vit
            'work_channel': 330,  # 320+10
            'vit_dim': 512,
            'vit_channel': 64,
            'vit_depth': 8,
            'vit_transf_qkv_dim': 2048,
            'vit_transf_dim': 2048,

            # file
            'dataset_root': 'D:/Code/data/Xray/Xray',
            "save_weight_dir": "./Checkpoints_xray/",
            "save_img": "./save_images_2",
            "device": "cuda",
            "training_load_weight": None,
            "labels": ("persistent stable right basilar atelectasis . low lung volumes and patient rotation . given "
                       "differences in technique , heart size within normal limits . persistent right basilar opacity "
                       ", atelectasis . no suspicious pulmonary opacity , pneumothorax or definite pleural effusion . "
                       "mild degenerative change of the thoracic spine ."
                       "CXR562_IM-2163-1001.png	no acute or active cardiac , pulmonary or pleural disease . "
                       "frontal and lateral views of the chest show normal size and configuration of the cardiac "
                       "silhouette . normal mediastinal contour , pulmonary and vasculature , central airways and "
                       "lung volumes . no pleural effusion ."),
            "test_load_weight": "ckpt_180_.pt", }
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        pass


if __name__ == '__main__':
    main(dataset='xray')
