python macer_main.py --task test --dataset mnist --root ./datasets/mnist5 --resume_ckpt ./checkpoints/mnist5/440.pth --sigma 0.50 --name m5 > M_catch
python macer_main.py --task test --dataset mnist --root ./datasets/mnist1 --resume_ckpt ./checkpoints/mnist1/440.pth --sigma 1.00 --name m1 >> M_catch
python macer_main.py --task test --dataset cifar10 --root ./datasets/c105 --resume_ckpt ./checkpoints/0.50.pth --sigma 0.5 --name c105 > C1cert_catch
python macer_main.py --task test --dataset cifar10 --root ./datasets/c110 --resume_ckpt ./checkpoints/1.00.pth --sigma 1.00 --name c101 >> C1cert_catch

