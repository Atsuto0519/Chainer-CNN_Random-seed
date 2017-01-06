# Chainer-CNN_Random-seed

## modify関連
##### 画面に配列データを画像として表示する。

## train_img32.py
##### 使うためには、~/.chainer/dataset/pfnet/chainer/imgの中にtrain.npzとtest.npzを入れておく。
##### npzファイルを作るには、train_img32.pyのあるディレクトリの下に"train/"と"test/"を作って、その中にまた数字で識別番号として振ったディレクトリの中に画像を入れ、_make_npz_img32()を実行。