# pytorch
import torch
class AbstractRecurrentEncoder():
    def forward(self, image, hidden=None, done=None):
        assert image.ndim == 5
        image = image.div(255)
        image = image.permute(0, 1, 4, 2, 3) # [バッチ、ステップ、高さ、横幅、チャンネル] から [バッチ、ステップ、チャンネル、高さ、横幅] まで
        batches, steps, channels, height, width = image.shape
        image_feature = self.vision(image.reshape(batches*steps, channels, height, width)).reshape(batches, steps, -1)
        if done is None:
            recurrent_feature, hidden = self.recurrent(image_feature, hidden)
        else:
            assert done.shape == (batches, steps)
            recurrent_feature = image_feature.new_zeros(batches, steps, self.hidden_features)
            for t in range(steps):
                recurrent_feature[:, t:t+1, :], hidden = self.recurrent(image_feature[:, t:t+1, :], hidden)
                hidden = hidden*done[None, :, t:t+1].logical_not()
        feature = torch.cat((image_feature, recurrent_feature), 2)
        # feature.shape == [バッチ、ステップ、特徴]
        return feature, hidden