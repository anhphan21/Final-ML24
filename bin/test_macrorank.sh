python test.py --model GClassifier  --dataset PlainClusterSet --checkp checkp/GClassifier_[1]_0/last.pth --base_model EGNN_DENSE --label 1  --group 0 --egnn_layers 4 --pos_encode 4
# python main.py --model GClassifier  --dataset PlainClusterSet --base_model EGNN_DENSE --label 2  --group 0 --egnn_layers 4 --pos_encode 4
# python main.py --model GClassifier  --dataset PlainClusterSet --base_model EGNN_DENSE --label 3  --group 0 --egnn_layers 5 --pos_encode 4
# python main.py --model GClassifier  --dataset PlainClusterSet --base_model EGNN_DENSE --label 1  --group 1 --egnn_layers 4 --pos_encode 4
# python main.py --model GClassifier  --dataset PlainClusterSet --base_model EGNN_DENSE --label 2  --group 1 --egnn_layers 4 --pos_encode 4

# python test.py --model GClassifier  --dataset PlainClusterSet --checkp chekcp/GClassifier_[1]_0/last.pth --base_model EGNN_DENSE --label 1  --group 0 --egnn_layers 4 --pos_encode 4
# python test.py --model GClassifier  --dataset PlainClusterSet --checkp chekcp/GClassifier_[2]_0/last.pth --base_model EGNN_DENSE --label 2  --group 0 --egnn_layers 4 --pos_encode 4
# python test.py --model GClassifier  --dataset PlainClusterSet --checkp chekcp/GClassifier_[3]_0/last.pth --base_model EGNN_DENSE --label 3  --group 0 --egnn_layers 5 --pos_encode 4
# python test.py --model GClassifier  --dataset PlainClusterSet --checkp chekcp/GClassifier_[1]_1/last.pth --base_model EGNN_DENSE --label 1  --group 1 --egnn_layers 4 --pos_encode 4
# python test.py --model GClassifier  --dataset PlainClusterSet --checkp chekcp/GClassifier_[2]_1/last.pth --base_model EGNN_DENSE --label 2  --group 1 --egnn_layers 4 --pos_encode 4
# python test.py --model GClassifier  --dataset PlainClusterSet --checkp chekcp/GClassifier_[3]_1/last.pth --base_model EGNN --label 3  --group 1 --egnn_layers 4 --pos_encode 0