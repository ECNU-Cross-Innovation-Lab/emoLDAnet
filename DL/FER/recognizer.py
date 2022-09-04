import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from FER import FERtransforms as transforms
from skimage.transform import resize
from FER.FERmodels import *


class Recognizer(object):

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def new_recognize(self, raw_imgs, model):

        if model.startswith('VGG'):
            net = VGG(model.split('_')[0])
        elif model == 'resnet18':
            net = ResNet18()
        else:
            net = ssyNet(model.split('_')[0])
        # checkpoint = torch.load(os.path.join('FER', 'FERckpts', model + '.t7'), map_location={'cuda:1':'cuda:0'})
        checkpoint = torch.load(("" + model + "/PrivateTest_model.t7"), map_location={'cuda:1':'cuda:0'})
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()

        cut_size = 44

        recscores = []

        for raw_img in raw_imgs:

            transform_test = transforms.Compose([
                transforms.TenCrop(cut_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            ])
            gray = self.rgb2gray(raw_img)
            gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

            img = gray[:, :, np.newaxis]

            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = transform_test(img)

            ncrops, c, h, w = np.shape(inputs)

            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = net(inputs)

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            score = F.softmax(outputs_avg, dim=0)
            _, predicted = torch.max(outputs_avg.data, 0)

            recscores.append(score.data.cpu().numpy())

        return recscores

    def recognize(self, raw_img):

        cut_size = 44

        transform_test = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        gray = self.rgb2gray(raw_img)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        net = VGG('VGG19')
        # checkpoint = torch.load(os.path.join('FER', 'FERckpts', 'PrivateTest_model.t7'), map_location='cpu')
        checkpoint = torch.load(os.path.join('FER', 'FERckpts', 'PrivateTest_model.t7'))
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)

        recscore = score.data.cpu().numpy()

        self.raw_img = raw_img
        self.score = score
        return recscore
