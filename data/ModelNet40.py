import numpy as np
import os
import torch
import torch.utils.data as data

type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}


class ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.augment_data = cfg['augment_data']
        self.part = part

        self.data = []
        labels = os.listdir(self.root)
        labels.sort()
        for label_index, label in enumerate(labels):
            type_root = os.path.join(os.path.join(self.root, label), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), label_index))

    def __getitem__(self, i):
        path, label = self.data[i]
        data = np.load(path)
        centers = data['centers'] # [face, 3]
        corners = data['corners'] # [face, vertice, 3]
        normals = data['normals'] # [face, 3]
        neighbors_index = data['neighbors_index'] # [face, ?]
        
        num_point = len(centers)
        # fill for n < 1024
        if num_point < 1024:
            chosen_indexes = np.random.randint(0, num_point, size=(1024 - num_point))
            centers = np.concatenate((centers, centers[chosen_indexes]))
            corners = np.concatenate((corners, corners[chosen_indexes]))
            normals = np.concatenate((normals, normals[chosen_indexes]))
            neighbors_index = np.concatenate((neighbors_index, neighbors_index[chosen_indexes]))
            
            # choose 3 neighbors
            new_neighbors_index = np.empty([1024, 3], dtype=np.int64)
            for idx in range(1024):
                neighbors = neighbors_index[idx]
                if len(neighbors) > 3:
                    new_neighbors_index[idx] = np.random.choice(neighbors, 3, replace=False)
                else:
                    new_neighbors_index[idx] = np.concatenate((neighbors, [idx]*(3-len(neighbors))))

            neighbors_index = new_neighbors_index
        else:
            chosen_indexes = np.random.choice(num_point, size=1024, replace=False)
            centers = centers[chosen_indexes]
            corners = corners[chosen_indexes]
            normals = normals[chosen_indexes]
            neighbors_index = neighbors_index[chosen_indexes]
            # remove unlinkable index and choose 3 neighbors
            new_neighbors_index = np.empty([1024, 3], dtype=np.int64)
            for idx in range(1024):
                mask = np.in1d(neighbors_index[idx], chosen_indexes)
                neighbors = np.array(neighbors_index[idx])[mask]
                if len(neighbors) > 3:
                    new_neighbors_index[idx] = np.random.choice(neighbors, 3, replace=False)
                else:
                    new_neighbors_index[idx] = np.concatenate((neighbors, [chosen_indexes[idx]]*(3-len(neighbors))))

            # re-index the neighbor
            invert_index = {value: key for key, value in enumerate(chosen_indexes)}
            neighbors_index = np.vectorize(invert_index.get, cache=True)(new_neighbors_index)

        # data augmentation
        if self.augment_data and self.part == 'train':
            centers = self.__augment__(centers)
            corners = self.__augment__(corners)

        # make corner relative to center
        corners = corners - centers[:, np.newaxis, :]
        corners = corners.reshape([-1, 9])

        # to tensor
        centers = torch.from_numpy(centers).float().permute(1, 0).contiguous()
        corners = torch.from_numpy(corners).float().permute(1, 0).contiguous()
        normals = torch.from_numpy(normals).float().permute(1, 0).contiguous()
        neighbors_index = torch.from_numpy(neighbors_index).long()
        target = torch.tensor(label, dtype=torch.long)

        return centers, corners, normals, neighbors_index, target

    def __augment__(self, data):
        sigma, clip = 0.01, 0.05
        jittered_data = np.clip(sigma * np.random.randn(*data.shape), -clip, clip)
        return data + jittered_data

    def __getitem__o(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < 1024
        num_point = len(face)
        if num_point < 1024:
            fill_face = []
            fill_neighbor_index = []
            for i in range(1024 - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = ModelNet40({'data_root': '../meshnet_data/ModelNet40/', 'augment_data': True})
    dataset[0]