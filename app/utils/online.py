import pickle, torch, socket, cv2, os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch.nn as nn
from torchvision import transforms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_actv(actv):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'prelu':
        return nn.PReLU(init=0.2)
    elif actv == 'elu':
        return nn.ELU(inplace=True)
    elif actv is None:
        return nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid activation: {:s}'.format(actv)
        )


def make_norm(norm, dim):
    if norm == 'layer':
        return nn.LayerNorm(dim)
    elif norm is None:
        return nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid normalization: {:s}'.format(norm)
        )


class FiLMModel(nn.Module):
    def __init__(self, feats_dim, resrc_dim, hid_dim, film_dim, out_dim,
                 norm=None, actv='leaky_relu', dropout=0):
        super(FiLMModel, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(feats_dim, hid_dim),
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
        )

        self.film = nn.Sequential(
            nn.Linear(resrc_dim, film_dim),
            make_norm(norm, film_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(film_dim, film_dim),
            make_norm(norm, film_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(film_dim, hid_dim * 4),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
        )
        self.out_fc = nn.Linear(hid_dim, out_dim)

    def forward(self, feats, resrc):
        x = self.stem(feats)
        h = self.film(resrc)
        g1, b1, g2, b2 = h.split(h.size(-1) // 4, -1)
        x = self.fc1(g1 * x + b1)
        x = self.fc2(g2 * x + b2)
        x = self.out_fc(x)
        return x


class FeatureToVecOneHeadOnline():
    def __init__(self, filename, mask=np.ones((1036,)).astype(bool), tv_version=None):

        torch.set_default_dtype(torch.float32)
        self.mask = mask
        self.model_acc = FiLMModel(1284, 22, 256, 128, 200, 'layer', 'leaky_relu', 0)
        if not tv_version:
            socket_name = socket.gethostname()
            version = {"xv3": "0.8.1", "tx2-1": "0.5.0", "tx2-2": "0.5.0"}[socket_name]
        else:
            version = tv_version
        self.model_fe = torch.hub.load('pytorch/vision:v{}'.format(version),
                                       'mobilenet_v2', pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        mydict = torch.load(filename)
        self.model_acc.load_state_dict(mydict["net"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_acc.to(self.device)
        self.model_fe.to(self.device)
        self.model_fe.eval()
        self.model_acc.eval()

        # ordinal encoding for latency requirements
        self.lat_levels = {
            33.3: [0, 0, 0, 1],
            50: [0, 0, 1, 1],
            100: [0, 1, 1, 1],
            200: [1, 1, 1, 1],
        }

        # ordinal encoding for CPU contention levels
        self.cpu_levels = {
            0: [0, 0, 0, 0, 0, 0, 1],
            1: [0, 0, 0, 0, 0, 1, 1],
            2: [0, 0, 0, 0, 1, 1, 1],
            3: [0, 0, 0, 1, 1, 1, 1],
            4: [0, 0, 1, 1, 1, 1, 1],
            5: [0, 1, 1, 1, 1, 1, 1],
            6: [1, 1, 1, 1, 1, 1, 1],
        }

        # ordinal encoding for GPU contention levels
        self.gpu_levels = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            10: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            20: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            30: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            40: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            50: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            60: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            70: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            80: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            90: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }

    def _encode_lat_req(self, lat_req):
        return np.array([self.lat_levels[l] for l in lat_req])  # [n, 4]

    def _encode_cpu_cont(self, cpu_cont):
        return np.array([self.cpu_levels[c] for c in cpu_cont])  # [n, 7]

    def _encode_gpu_cont(self, gpu_cont):
        return np.array([self.gpu_levels[g] for g in gpu_cont])  # [n, 11]

    def fe(self, img_pil):
        # (1, 3, 224, 224) torch array
        img_pre = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        # (1280,) torch array
        feature_heavy = torch.mean(self.model_fe.features(img_pre)[0], dim=(1, 2))
        # (1280,) numpy array
        feature_heavy = feature_heavy.cpu().detach().numpy()
        return feature_heavy

    def predict(self, feature, lat_req, cpu_cont, gpu_cont):
        # lat_req, cpu_cont, gpu_cont are (batch_size,) numpy arrays
        # output is a (1036,) numpy array
        feature = torch.from_numpy(np.array(feature)).float().to(self.device)  # [n, 4+1280]
        lat_req = torch.from_numpy(self._encode_lat_req(lat_req)).float().to(self.device)
        cpu_cont = torch.from_numpy(self._encode_cpu_cont(cpu_cont)).float().to(self.device)
        gpu_cont = torch.from_numpy(self._encode_gpu_cont(gpu_cont)).float().to(self.device)
        resrc = torch.cat([lat_req, cpu_cont, gpu_cont], dim=1)  # [n, 22]
        output = self.model_acc.forward(feature, resrc)  # [n,4+1280] Tensor --> [n,200] Tensor
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = np.exp(output)
        return output_ret


class BaselineAccuracyPredictorOnline:
    def __init__(self, filename):
        self.accuracy = pickle.load(open(filename, "rb"))

    def predict(self):
        return self.accuracy


class NN_residual(torch.nn.Module):
    # Architecture of the accuracy predictor
    #   first project (light, heavy) features to (256, 256) shape, 
    #   then concatenate and proceed with 5-layer fully connected layers
    #   1s in the mask determines the number of branches we predict on
    def __init__(self, input_dim, mask=np.ones((1036,)).astype(bool)):

        # dim = 4, #neurons = [4] --> [256, 256, 1036]
        # dim = X, #neurons = [4, X] --> [256+256=256, 256, 256, 256, 256, 1036], X = {768, 5400, 1280}
        super(NN_residual, self).__init__()
        self.input_dim = input_dim
        self.N_branch = sum(mask)
        if input_dim == 4:
            self.project0 = torch.nn.Linear(4, 256)  # [n,4]     -> [n,256]
            self.fc1 = torch.nn.Linear(256, 256)  # [n,256]   -> [n,256]
            self.fc2 = torch.nn.Linear(256, self.N_branch)  # [n,256]   -> [n,self.N_branch]
        else:
            self.project0 = torch.nn.Linear(4, 256)
            self.project1 = torch.nn.Linear(input_dim - 4, 256)
            self.fc1 = torch.nn.Linear(256, 256)  # [n,256]   -> [n,256]
            self.fc2 = torch.nn.Linear(256, 256)  # [n,256]   -> [n,256]
            self.fc3 = torch.nn.Linear(256, 256)  # [n,256]   -> [n,256]
            self.fc4 = torch.nn.Linear(256, 256)  # [n,256]   -> [n,256]
            self.fc5 = torch.nn.Linear(256, self.N_branch)  # [n,256]   -> [n,self.N_branch]

    def forward(self, feature):
        # dim = 4, activation = relu, sigmoid
        # dim = X, activation = relu, relu, relu, relu, sigmoid
        x = feature.double()
        if self.input_dim == 4:
            x = torch.nn.functional.relu(self.project0(x))  # [n,4]             -> [n,256]
            x = x + torch.nn.functional.relu(self.fc1(x))  # [n,256] + [n,256] -> [n,256]
            x = torch.sigmoid(self.fc2(x))  # [n,256]           -> [n,self.N_branch]
        else:
            xl = x[:, 0:4]
            xl = torch.nn.functional.relu(self.project0(xl))  # [n,4]             -> [n,256]
            xh = x[:, 4:self.input_dim]
            xh = torch.nn.functional.relu(self.project1(xh))  # [n,X]             -> [n,256]
            x = xl + xh  # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc1(x))  # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc2(x))  # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc3(x))  # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc4(x))  # [n,256] + [n,256] -> [n,256]
            x = torch.sigmoid(self.fc5(x))  # [n,256]           -> [n,self.N_branch]
        return x


class FeatureToVecOnline():
    def __init__(self, feature, filename, mask):
        torch.set_default_dtype(torch.float64)
        input_dim = {"light": 4, "HoC": 4 + 768, "HoG": 4 + 5400,  # "MobileNetV2": 4+62720,
                     "MobileNetV2Pool": 4 + 1280, "RPN": 4 + 1024, "CPoP": 4 + 31}[feature]
        self.model = NN_residual(input_dim, mask=mask)
        mydict = torch.load(filename)
        self.model.load_state_dict(mydict["model"])
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.mask = mask

    def predict(self, feature):
        # input feature is a (batch_size, feature_dim) numpy array
        # output accuracy prediction is a (batch_size, 1036) numpy array
        feature = torch.from_numpy(feature).double().to(device=self.device)
        output = self.model.forward(feature)
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = output
        return output_ret


class FeatureToVecJointOnline():
    def __init__(self, filename, mask, trainable_fe, tv_version=None):

        torch.set_default_dtype(torch.float64)
        self.mask = mask
        input_dim = 4 + 1280  # Support "MobileNetV2Pool" only
        self.model_acc = NN_residual(input_dim, mask=mask)
        if not tv_version:
            socket_name = socket.gethostname()
            version = {"xv3": "0.8.1", "tx2-1": "0.5.0", "tx2-2": "0.5.0"}[socket_name]
        else:
            version = tv_version
        self.model_fe = torch.hub.load('pytorch/vision:v{}'.format(version),
                                       'mobilenet_v2', pretrained=True)

        mydict = torch.load(filename)
        if trainable_fe:
            self.model_acc.load_state_dict(mydict["model_acc"])
            self.model_fe.load_state_dict(mydict["model_fe"])
        else:
            self.model_acc.load_state_dict(mydict["model"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_acc.to(self.device)
        self.model_fe.to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def fe(self, img_pil):

        # (1, 3, 224, 224) torch array
        img_pre = self.preprocess(img_pil).double().unsqueeze(0).to(self.device)
        # (1280,) torch array
        feature_heavy = torch.mean(self.model_fe.features(img_pre)[0], dim=(1, 2))
        # (1280,) numpy array
        feature_heavy = feature_heavy.cpu().detach().numpy()
        return feature_heavy

    def predict(self, feature):

        # output_ret is a (1036,) numpy array
        feature = torch.from_numpy(np.array(feature)).double().to(self.device)  # [n, 4+1280]
        output = self.model_acc.forward(feature)  # [n,4+1280] Tensor --> [n,1036] Tensor
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = output
        return output_ret


class TrackerPredictor:
    def __init__(self, model_file):

        self.list_tracker = [("medianflow", 4), ("medianflow", 2),
                             ("medianflow", 1), ("kcf", 4),
                             ("csrt", 4), ("bboxmedianfixed", 4)]
        self.sis = [1, 2, 4, 8, 20, 50, 100]
        self.transform = PolynomialFeatures(2)
        self.init_models, self.tr_models = {}, {}
        self.init_coeff, self.init_intercept = [], []
        self.tracking_coeff, self.tracking_intercept = [], []

        all_models = pickle.load(open(model_file, 'rb'))
        for tracker, ds in self.list_tracker:
            key = "{}_ds{}_init".format(tracker, ds)
            if key in all_models:
                self.init_models[(tracker, ds)] = all_models[key]
                self.init_coeff.append(all_models[key].coef_)
                self.init_intercept.append(all_models[key].intercept_)
            else:
                print("Error in loading accuracy prediction model.")
                return

            key = "{}_ds{}_tracking".format(tracker, ds)
            if key in all_models:
                self.tr_models[(tracker, ds)] = all_models[key]
                self.tracking_coeff.append(all_models[key].coef_)
                self.tracking_intercept.append(all_models[key].intercept_)
            else:
                print("Error in loading accuracy prediction model.")
                return

        self.init_coeff = np.array(self.init_coeff)
        self.init_intercept = np.array(self.init_intercept)
        self.tracking_coeff = np.array(self.tracking_coeff)
        self.tracking_intercept = np.array(self.tracking_intercept)

    # faster implementation
    def batch_prediction(self, num_obj, avg_size, width, height, core=0):

        feature = [num_obj, avg_size, width, height, core]
        X = np.array(feature).reshape(1, -1)  # (1, 5) shape
        X = self.transform.fit_transform(X).squeeze()
        init_time = np.dot(self.init_coeff, X) + self.init_intercept
        tr_time = np.dot(self.tracking_coeff, X) + self.tracking_intercept
        return init_time, tr_time  # both in (6,) shape

    def predict(self, num_obj, avg_size, width, height, core=0):
        accuracy_init, accuracy_tr = self.batch_prediction(num_obj, avg_size,
                                                           width, height, core=core)

        # Construct the tracker accuracy array if si > 1, in (1440,) shape
        lat_init, lat_tr = np.zeros((6, 40, 6)), np.zeros((6, 40, 6))
        for idx in range(6):
            lat_init[:, :, idx] = accuracy_init[idx]
            lat_tr[:, :, idx] = accuracy_tr[idx]
        for idx, si in enumerate(self.sis[1:]):
            lat_init[idx, :, :] *= (1 / si)
            lat_tr[idx, :, :] *= ((si - 1) / si)
        detector_only = np.zeros((40,))
        treacker_accuracy = (lat_init + lat_tr).flatten()
        return np.concatenate((detector_only, treacker_accuracy))  # (1480,) shape


class DNNPredictor:
    def __init__(self, model_file, version="v2b"):
        def filter_det(lat):
            while len(lat) > 3:
                mean, std = np.mean(lat), np.std(lat)
                new_lat = [l for l in lat if abs(l - mean) <= 1.5 * std]
                if len(new_lat) == len(lat):
                    lat = new_lat
                    break
                else:
                    lat = new_lat
            return lat

        self.version = version
        self.fshapes = [224, 320, 448, 576]
        self.nprops = [1, 3, 5, 10, 20, 50, 100]
        self.yshapes = list(range(224, 577, 32))
        self.sis = [1, 2, 4, 8, 20, 50, 100]
        self.ker_sh_np = [('FRCNN', s, n) for s in self.fshapes for n in self.nprops]
        self.ker_sh_np += [('YOLO', s, -1) for s in self.yshapes]
        # map any (h, w) in val/test to 15 profiled (h, w) in val
        self.hwm = {(576, 1280): (720, 1280), (360, 480): (360, 480),
                    (360, 608): (360, 640), (358, 480): (360, 480),
                    (180, 320): (240, 320), (720, 1270): (720, 1280),
                    (720, 1280): (720, 1280), (144, 192): (240, 320),
                    (360, 636): (360, 640), (480, 872): (480, 640),
                    (480, 640): (480, 640), (352, 640): (358, 640),
                    (240, 426): (240, 426), (360, 376): (360, 450),
                    (720, 1278): (720, 1280), (424, 640): (424, 640),
                    (358, 640): (358, 640), (360, 490): (360, 480),
                    (304, 540): (320, 568), (270, 480): (270, 480),
                    (1080, 1920): (1080, 1920), (180, 240): (240, 320),
                    (360, 472): (360, 480), (360, 600): (360, 640),
                    (360, 524): (360, 540), (360, 534): (360, 540),
                    (320, 568): (320, 568), (720, 960): (720, 960),
                    (288, 512): (270, 480), (288, 384): (240, 426),
                    (360, 450): (360, 450), (360, 492): (360, 480),
                    (264, 396): (240, 426), (360, 640): (360, 640),
                    (264, 480): (270, 480), (360, 426): (360, 450),
                    (488, 640): (480, 640), (720, 406): (720, 406),
                    (240, 320): (240, 320), (816, 1920): (1080, 1920),
                    (160, 208): (240, 320), (360, 540): (360, 540)}

        data = pickle.load(open(model_file, "rb"))
        self.LUT, self.gl_seen = {}, set()
        for key, arr in data.items():
            dataset, kernel, shape, nprop, height, width, gl = key
            if dataset == 'test':
                continue
            self.gl_seen.add(gl)
            arr = filter_det(arr)
            self.LUT[(kernel, shape, nprop, height, width, gl)] = np.mean(arr)

        # Construct cache to speed up
        self.gs = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 99 not allowed
        self.cache = {}
        for height, width in self.hwm:
            for g in self.gs:
                self.cache[(height, width, g)] = self.predict(height, width, 0, g)

    def predict(self, height, width, cpu_contention=0, gpu_contention=0):
        if self.version == 'v2b' and (height, width, gpu_contention) in self.cache:
            return self.cache[(height, width, gpu_contention)]
        if self.version == "v2b":  # baseline v2: LUT for each s, n, h, w, g
            if (height, width) in self.hwm:
                height, width = self.hwm[(height, width)]
            else:
                height, width = 720, 1280
            if not gpu_contention in self.gl_seen:
                print("gpu_contention not seen in offline data, set it to 90.")
                gpu_contention = 90
            ans_det = [self.LUT[(k, s, n, height, width, gpu_contention)] \
                       for k, s, n in self.ker_sh_np]  # (40,) list
            ans_det = np.array(ans_det)  # (40,) np
            ans = np.array([ans_det / si for si in self.sis[1:]])  # (6, 40) np
            ans = np.repeat(ans[:, :, np.newaxis], 6, axis=2)  # (6, 40, 6) np
            return np.concatenate((ans_det.flatten(), ans.flatten()))  # (1480,) shape
        elif self.version == 'v2':
            features = [height, width, gpu_contention]  # (3,)
            features = np.array(features).reshape(1, -1)
            features = self.transform.fit_transform(features).squeeze()  # (10,)
            ans_det = np.dot(self.coeff_array, features) + self.bias_array
            # now_time = now_time.reshape(4, 7)  # (shape, nprop), arranged (4, 7) shape
            # (si, shape, nprop) arranged , (7, 4, 7) shape
            ans = np.array([ans_det / si for si in self.sis[1:]])  # (6, 40) shape
            ans = np.repeat(ans[:, :, np.newaxis], 6, axis=2)  # (6, 40, 6) shape
            # (si, shape, nprop, tracker) arranged, (7, 4, 7, 6) shape
            return np.concatenate((ans_det.flatten(), ans.flatten()))  # (1480,) shape
        else:  # v1: contains model for FRCNN only
            features = [height, width, cpu_contention, gpu_contention]  # (4,)
            features = np.array(features).reshape(1, -1)
            features = self.transform.fit_transform(features).squeeze()  # (15,)
            now_time = np.dot(self.coeff_array, features) + self.bias_array
            now_time = now_time.reshape(4, 7)  # (shape, nprop), arranged (4, 7) shape
            # (si, shape, nprop) arranged , (7, 4, 7) shape
            with_ds_array = np.stack([now_time / self.sis[i] for i in range(7)], axis=0)
            # (si, shape, nprop, tracker) arranged, (7, 4, 7, 6) shape
            final_array = np.repeat(with_ds_array[:, :, :, np.newaxis], 6, axis=3)
            return final_array.flatten()


class Predictor:
    def __init__(self, dlp_model="models/ApproxDet_LatDet_1228.pb",
                 tlp_model="models/ApproxDet_LatTr_1227.pb"):
        self.dlp = DNNPredictor(model_file=dlp_model, version='v2b')
        self.tlp = TrackerPredictor(model_file=tlp_model)

    def convert1480to1036(self, vec1480):
        vec40, vec1440 = vec1480[:40], vec1480[40:]
        vec6_40_6 = vec1440.reshape(6, 40, 6)
        vec6_28_6 = vec6_40_6[:, :28, :]
        vec28 = vec40[:28]
        vec1036 = np.concatenate((vec28, vec6_28_6.flatten()))
        return vec1036

    def predict(self, height=720, width=1280, nobj=1, objsize=220, cl=0, gl=0, FRCNN_only=True):
        per_branch_DNN_accuracy = self.dlp.predict(height=height, width=width, gpu_contention=gl)
        per_branch_tracker_accuracy = self.tlp.predict(nobj, objsize, width, height, core=cl)
        per_branch_accuracy = per_branch_DNN_accuracy + per_branch_tracker_accuracy
        if FRCNN_only:  # convert the (1480,) array to a (1036,) one
            per_branch_accuracy = self.convert1480to1036(per_branch_accuracy)
        return per_branch_accuracy


class FeatureExtractor:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        if feature_name == 'HoG':
            self.extractor = self.hog_extractor
        elif feature_name == 'HoC':
            self.extractor = self.hoc_extractor

    def hog_extractor(self, input_image):
        winSize = (320, 480)
        input_image = cv2.resize(input_image, winSize)
        blockSize = (80, 80)  # 105
        blockStride = (80, 80)
        cellSize = (16, 16)
        Bin = 9  # 3780
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, Bin)
        return hog.compute(input_image)[:, 0]

    def hoc_extractor(self, input_image):
        # input_image: (h, w, 3) numpy image in BGR order
        # output: 768 x 1 dimension vector
        h, w, _ = input_image.shape
        hist_b = cv2.calcHist([input_image], [0], None, [256], [0, 255])
        hist_b = hist_b / np.linalg.norm(hist_b, ord=2)
        hist_g = cv2.calcHist([input_image], [1], None, [256], [0, 255])
        hist_g = hist_g / np.linalg.norm(hist_g, ord=2)
        hist_r = cv2.calcHist([input_image], [2], None, [256], [0, 255])
        hist_r = hist_r / np.linalg.norm(hist_r, ord=2)
        return np.concatenate((hist_b, hist_g, hist_r), axis=0)[:, 0]

    def extract(self, input_image):
        return self.extractor(input_image)


def output_dict_to_bboxes_single_img(output_dict):
    # Output translation, in (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
    # all outputs are float32 numpy arrays, so convert types as appropriate
    N = int(output_dict['num_detections'][0])
    boxes = [(cls - 1, sc, box[0], box[1], box[2], box[3]) for cls, box, sc in \
             zip(output_dict['detection_classes'][0].astype(np.int64)[:N],
                 output_dict['detection_boxes'][0][:N],
                 output_dict['detection_scores'][0][:N])]
    return boxes
